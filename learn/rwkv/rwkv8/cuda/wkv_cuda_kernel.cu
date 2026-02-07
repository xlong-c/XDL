/*
 * RWKV WKV CUDA Kernel Implementation
 * 
 * 提供高效的 WKV 计算，支持并行训练模式和递推推理模式
 */

 #include <torch/extension.h>
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <vector>
 
 // CUDA 错误检查宏
 #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
 #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
 #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
 
 ///////////////////////////////////////////////////////////////////////////////
 // 工具函数
 ///////////////////////////////////////////////////////////////////////////////
 
 __device__ __forceinline__ float sigmoidf(float x) {
     return 1.0f / (1.0f + expf(-x));
 }
 
 __device__ __forceinline__ float reluf(float x) {
     return fmaxf(0.0f, x);
 }
 
 ///////////////////////////////////////////////////////////////////////////////
 // Kernel 1: 并行 WKV 前向传播 (用于训练)
 ///////////////////////////////////////////////////////////////////////////////
 
 template <typename scalar_t>
 __global__ void wkv_forward_parallel_kernel(
     const scalar_t* __restrict__ r,      // [B, T, C]
     const scalar_t* __restrict__ k,      // [B, T, C]
     const scalar_t* __restrict__ v,      // [B, T, C]
     const scalar_t* __restrict__ w,      // [C]
     const scalar_t* __restrict__ u,      // [C]
     scalar_t* __restrict__ out,          // [B, T, C]
     int B, int T, int C
 ) {
     // 每个线程处理一个 (batch, channel) 对
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     int total = B * C;
     
     if (idx >= total) return;
     
     int b = idx / C;
     int c = idx % C;
     
     // 加载参数
     float wc = -fabsf(static_cast<float>(w[c]));  // 确保为负
     float uc = static_cast<float>(u[c]);
     float exp_w = expf(wc);
     
     // 累积变量
     float num = 0.0f;
     float den = 0.0f;
     
     // 遍历时间步
     for (int t = 0; t < T; t++) {
         int idx_btc = ((b * T) + t) * C + c;
         
         float rt = static_cast<float>(r[idx_btc]);
         float kt = static_cast<float>(k[idx_btc]);
         float vt = static_cast<float>(v[idx_btc]);
         
         float exp_k = expf(kt);
         
         // 更新累积
         num = exp_k * vt + num * exp_w;
         den = exp_k + den * exp_w;
         
         // Bonus 项
         float exp_k_u = expf(kt + uc);
         float num_bonus = num + exp_k_u * vt;
         float den_bonus = den + exp_k_u;
         
         // 计算输出
         float wkv = rt * num_bonus / (den_bonus + 1e-6f);
         out[idx_btc] = static_cast<scalar_t>(wkv);
     }
 }
 
 
 ///////////////////////////////////////////////////////////////////////////////
 // Kernel 2: 递推 WKV 前向传播 (用于推理)
 ///////////////////////////////////////////////////////////////////////////////
 
 template <typename scalar_t>
 __global__ void wkv_forward_recursive_kernel(
     const scalar_t* __restrict__ r,      // [B, 1, C] - 单步
     const scalar_t* __restrict__ k,      // [B, 1, C]
     const scalar_t* __restrict__ v,      // [B, 1, C]
     const scalar_t* __restrict__ w,      // [C]
     const scalar_t* __restrict__ u,      // [C]
     scalar_t* __restrict__ out,          // [B, 1, C]
     scalar_t* __restrict__ state_out,    // [B, 2, C] - 输出新状态 (num, den)
     const scalar_t* __restrict__ state_in, // [B, 2, C] - 输入状态 (num, den)
     int B, int C
 ) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx >= B * C) return;
     
     int b = idx / C;
     int c = idx % C;
     
     // 加载参数
     float wc = -fabsf(static_cast<float>(w[c]));
     float uc = static_cast<float>(u[c]);
     float exp_w = expf(wc);
     
     // 加载输入
     float rt = static_cast<float>(r[b * C + c]);
     float kt = static_cast<float>(k[b * C + c]);
     float vt = static_cast<float>(v[b * C + c]);
     
     // 加载状态
     float num, den;
     if (state_in != nullptr) {
         num = static_cast<float>(state_in[(b * 2 + 0) * C + c]);
         den = static_cast<float>(state_in[(b * 2 + 1) * C + c]);
     } else {
         num = 0.0f;
         den = 0.0f;
     }
     
     // 计算
     float exp_k = expf(kt);
     float num_new = exp_k * vt + num * exp_w;
     float den_new = exp_k + den * exp_w;
     
     // Bonus
     float exp_k_u = expf(kt + uc);
     float num_bonus = num_new + exp_k_u * vt;
     float den_bonus = den_new + exp_k_u;
     
     // 输出
     float wkv = rt * num_bonus / (den_bonus + 1e-6f);
     out[b * C + c] = static_cast<scalar_t>(wkv);
     
     // 保存新状态
     state_out[(b * 2 + 0) * C + c] = static_cast<scalar_t>(num_new);
     state_out[(b * 2 + 1) * C + c] = static_cast<scalar_t>(den_new);
 }
 
 
 ///////////////////////////////////////////////////////////////////////////////
 // PyTorch 接口
 ///////////////////////////////////////////////////////////////////////////////
 
 torch::Tensor wkv_forward_parallel_cuda(
     torch::Tensor r,
     torch::Tensor k,
     torch::Tensor v,
     torch::Tensor w,
     torch::Tensor u
 ) {
     CHECK_INPUT(r);
     CHECK_INPUT(k);
     CHECK_INPUT(v);
     CHECK_INPUT(w);
     CHECK_INPUT(u);
     
     int B = r.size(0);
     int T = r.size(1);
     int C = r.size(2);
     
     auto out = torch::zeros_like(r);
     
     const int threads = 256;
     const int blocks = (B * C + threads - 1) / threads;
     
     AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "wkv_forward_parallel", ([&] {
         wkv_forward_parallel_kernel<scalar_t><<<blocks, threads>>>(
             r.data_ptr<scalar_t>(),
             k.data_ptr<scalar_t>(),
             v.data_ptr<scalar_t>(),
             w.data_ptr<scalar_t>(),
             u.data_ptr<scalar_t>(),
             out.data_ptr<scalar_t>(),
             B, T, C
         );
     }));
     
     return out;
 }
 
 std::vector<torch::Tensor> wkv_forward_recursive_cuda(
     torch::Tensor r,
     torch::Tensor k,
     torch::Tensor v,
     torch::Tensor w,
     torch::Tensor u,
     c10::optional<torch::Tensor> state
 ) {
     CHECK_INPUT(r);
     CHECK_INPUT(k);
     CHECK_INPUT(v);
     CHECK_INPUT(w);
     CHECK_INPUT(u);
     
     int B = r.size(0);
     int C = r.size(2);
     
     auto out = torch::zeros_like(r);
     auto state_out = torch::zeros({B, 2, C}, r.options());
     
     const int threads = 256;
     const int blocks = (B * C + threads - 1) / threads;
     
     const float* state_ptr = nullptr;
     if (state.has_value()) {
         CHECK_INPUT(state.value());
         state_ptr = state.value().data_ptr<float>();
     }
     
     AT_DISPATCH_FLOATING_TYPES(r.scalar_type(), "wkv_forward_recursive", ([&] {
         wkv_forward_recursive_kernel<scalar_t><<<blocks, threads>>>(
             r.data_ptr<scalar_t>(),
             k.data_ptr<scalar_t>(),
             v.data_ptr<scalar_t>(),
             w.data_ptr<scalar_t>(),
             u.data_ptr<scalar_t>(),
             out.data_ptr<scalar_t>(),
             state_out.data_ptr<scalar_t>(),
             state_ptr,
             B, C
         );
     }));
     
     return {out, state_out};
 }
 
 
 ///////////////////////////////////////////////////////////////////////////////
 // Python 绑定
 ///////////////////////////////////////////////////////////////////////////////
 
 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
     m.doc() = "RWKV WKV CUDA Kernels";
     
     m.def("wkv_forward_parallel", &wkv_forward_parallel_cuda, 
           "WKV forward (parallel version for training)");
     
     m.def("wkv_forward_recursive", &wkv_forward_recursive_cuda,
           "WKV forward (recursive version for inference)");
 }
