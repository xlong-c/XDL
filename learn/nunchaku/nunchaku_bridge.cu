// nunchaku → torch 桥接层
// 将 torch::Tensor 转换为 nunchaku kernel 所需的原始指针格式

#include "nunchaku_bridge.h"
#include "zgemm.h"
#include "gemm_w4a4.cuh"
#include "gemm_w4a4_launch.cuh"
#include "gemm_w8a8.cuh"
#include "attention.cuh"
#include "epilogues.cuh"

#include <cuda_runtime_api.h>

namespace nunchaku_bridge {

using namespace nunchaku::kernels;

// === 辅助: torch scalar type → nunchaku 判断 ===
static bool is_bf16(torch::ScalarType t) { return t == torch::kBFloat16; }

// ============================================================
// W4A4 GEMM
// ============================================================
void gemm_w4a4(torch::Tensor act, torch::Tensor wgt, torch::Tensor out,
               torch::Tensor ascales, torch::Tensor wscales,
               torch::Tensor bias) {
    // act:  [M, K/2]  packed INT4 (存为 INT8)
    // wgt:  [N, K/2]  packed INT4
    // out:  [M, N]    FP16/BF16
    int M = act.size(0);
    int N = wgt.size(0);
    int K = act.size(1) * 2;  // INT4: 2 elements per byte

    bool bf16 = is_bf16(out.scalar_type());

    auto launch = [&]<typename Config, bool USE_FP4>() {
        using GEMM = GEMM_W4A4<Config>;
        using Epilogue = typename GEMM::EpilogueDefault;

        dim3 grid(ceilDiv(M, GEMM::BLOCK_M), ceilDiv(N, GEMM::BLOCK_N));
        dim3 block(GEMM::WARP_SIZE * GEMM::NUM_WARPS);

        bool swapBlockMN = M > N * 2;
        if (swapBlockMN) std::swap(grid.x, grid.y);

        typename Epilogue::Arguments args;
        args.out     = out.data_ptr<typename GEMM::half_t>();
        args.actualM = M;
        args.actualN = N;

        auto kernel = invoke_kernel<typename GEMM::template gemm_w4a4_kernel<Epilogue, false>,
                                     const typename GEMM::packed_act_t*,
                                     const typename GEMM::packed_wgt_t*,
                                     const typename GEMM::packed_ascale_t*,
                                     const typename GEMM::packed_wscale_t*,
                                     int, int, int,
                                     typename Epilogue::Arguments, bool, bool>;

        kernel<<<grid, block>>>(
            reinterpret_cast<const typename GEMM::packed_act_t*>(act.data_ptr<int8_t>()),
            reinterpret_cast<const typename GEMM::packed_wgt_t*>(wgt.data_ptr<int8_t>()),
            reinterpret_cast<const typename GEMM::packed_ascale_t*>(ascales.data_ptr()),
            reinterpret_cast<const typename GEMM::packed_wscale_t*>(wscales.data_ptr()),
            M, N, K, args, swapBlockMN, false);

        checkCUDA(cudaGetLastError());
    };

    dispatchBool(false /* no FP4 */, [&]<bool USE_FP4>() {
        if (bf16) {
            launch.template operator()<GEMMConfig_W4A4_BF16, USE_FP4>();
        } else {
            launch.template operator()<GEMMConfig_W4A4_FP16, USE_FP4>();
        }
    });
}

// ============================================================
// W8A8 GEMM
// ============================================================
void gemm_w8a8(torch::Tensor act, torch::Tensor wgt, torch::Tensor out,
               torch::Tensor ascales, torch::Tensor wscales,
               torch::Tensor bias) {
    using GEMM = GEMM_W8A8;

    int M = act.size(0);
    int N = wgt.size(0);
    int K = act.size(1);

    dim3 grid(ceilDiv(M, GEMM::BLOCK_M), ceilDiv(N, GEMM::BLOCK_N));
    dim3 block(GEMM::WARP_SIZE * GEMM::NUM_WARPS);

    bool swapBlockMN = M > N * 2;
    if (swapBlockMN) std::swap(grid.x, grid.y);

    typename GEMM::EpilogueDefault::Arguments args;
    args.out     = out.data_ptr<GEMM::half_t>();
    args.actualM = M;
    args.actualN = N;

    auto kernel = invoke_kernel<GEMM::template gemm_w8a8_kernel<GEMM::EpilogueDefault>,
                                 const GEMM::packed_act_t*,
                                 const GEMM::packed_wgt_t*,
                                 const GEMM::packed_ascale_t*,
                                 const GEMM::packed_wscale_t*,
                                 int, int, int,
                                 typename GEMM::EpilogueDefault::Arguments, bool, bool>;

    kernel<<<grid, block>>>(
        reinterpret_cast<const GEMM::packed_act_t*>(act.data_ptr<int8_t>()),
        reinterpret_cast<const GEMM::packed_wgt_t*>(wgt.data_ptr<int8_t>()),
        reinterpret_cast<const GEMM::packed_ascale_t*>(ascales.data_ptr()),
        reinterpret_cast<const GEMM::packed_wscale_t*>(wscales.data_ptr()),
        M, N, K, args, swapBlockMN, false);

    checkCUDA(cudaGetLastError());
}

// ============================================================
// Flash Attention
// ============================================================
void attention_fp16(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                    torch::Tensor o, float scale) {
    int B    = q.size(0);
    int H    = q.size(1);
    int NQ   = q.size(2);
    int D    = q.size(3);
    int NKV  = k.size(2);

    bool bf16_out = is_bf16(o.scalar_type());

    auto launch = [&]<bool bf16out>() {
        using ATTN = Attention<AttentionFP16Config<bf16out>>;
        using GEMM_ = typename ATTN::GEMM;

        dim3 grid(ceilDiv(NQ, ATTN::BLOCK_M), H, B);

        typename GEMM_::EpilogueDefault::Arguments args;
        args.out     = o.data_ptr<typename GEMM_::half_t>();
        args.actualM = B * NQ;
        args.actualN = H * D;

        auto kernel = invoke_kernel<typename ATTN::template attention_fp16_kernel<typename GEMM_::EpilogueDefault>,
                                     const typename ATTN::packed_q_t*,
                                     const typename ATTN::packed_k_t*,
                                     const typename ATTN::packed_v_t*,
                                     float, int, int,
                                     typename GEMM_::EpilogueDefault::Arguments, bool>;

        kernel<<<grid, GEMM_::WARP_SIZE * GEMM_::NUM_WARPS>>>(
            reinterpret_cast<const typename ATTN::packed_q_t*>(q.data_ptr()),
            reinterpret_cast<const typename ATTN::packed_k_t*>(k.data_ptr()),
            reinterpret_cast<const typename ATTN::packed_v_t*>(v.data_ptr()),
            scale * 1.44269504f,  // M_LOG2E: exp2 代替 exp
            NQ, NKV, args, false);

        checkCUDA(cudaGetLastError());
    };

    if (bf16_out) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
}

// ============================================================
// 量化: W4A4 Activation
// ============================================================
void quantize_w4a4_act(torch::Tensor input, torch::Tensor output, torch::Tensor oscales) {
    int M = input.size(0);
    int K = input.size(1);

    bool bf16 = is_bf16(input.scalar_type());

    auto launch = [&]<typename Config, bool USE_FP4>() {
        using GEMM = GEMM_W4A4<Config>;
        using Kernel = typename GEMM::quantize_w4a4_act_kernel;

        dim3 grid(M / GEMM::WARP_M, ceilDiv(K, GEMM::WARP_K));
        dim3 block(GEMM::WARP_SIZE);

        auto kernel = invoke_kernel<Kernel, const GEMM::half_t*,
                                     GEMM::packed_act_t*, GEMM::packed_ascale_t*, int>;

        kernel<<<grid, block>>>(
            input.data_ptr<GEMM::half_t>(),
            reinterpret_cast<GEMM::packed_act_t*>(output.data_ptr<int8_t>()),
            reinterpret_cast<GEMM::packed_ascale_t*>(oscales.data_ptr()),
            K);

        checkCUDA(cudaGetLastError());
    };

    dispatchBool(false, [&]<bool USE_FP4>() {
        if (bf16) launch.template operator()<GEMMConfig_W4A4_BF16, USE_FP4>();
        else      launch.template operator()<GEMMConfig_W4A4_FP16, USE_FP4>();
    });
}

// ============================================================
// 量化: W4A4 Weight
// ============================================================
void quantize_w4a4_wgt(torch::Tensor input, torch::Tensor output, torch::Tensor oscales) {
    int N = input.size(0);
    int K = input.size(1);

    bool bf16 = is_bf16(input.scalar_type());

    auto launch = [&]<typename Config, bool USE_FP4>() {
        using GEMM = GEMM_W4A4<Config>;
        using Kernel = typename GEMM::quantize_w4a4_wgt_kernel;

        dim3 grid(N / GEMM::WARP_N, ceilDiv(K, GEMM::WARP_K));
        dim3 block(GEMM::WARP_SIZE);

        auto kernel = invoke_kernel<Kernel, const GEMM::half_t*,
                                     GEMM::packed_wgt_t*, GEMM::packed_wscale_t*, int>;

        kernel<<<grid, block>>>(
            input.data_ptr<GEMM::half_t>(),
            reinterpret_cast<GEMM::packed_wgt_t*>(output.data_ptr<int8_t>()),
            reinterpret_cast<GEMM::packed_wscale_t*>(oscales.data_ptr()),
            K);

        checkCUDA(cudaGetLastError());
    };

    dispatchBool(false, [&]<bool USE_FP4>() {
        if (bf16) launch.template operator()<GEMMConfig_W4A4_BF16, USE_FP4>();
        else      launch.template operator()<GEMMConfig_W4A4_FP16, USE_FP4>();
    });
}

// ============================================================
// 量化: W8A8 Activation
// ============================================================
void quantize_w8a8_act(torch::Tensor input, torch::Tensor output, torch::Tensor oscales, bool fuse_glu) {
    using GEMM = GEMM_W8A8;

    int M = input.size(0);
    int K = input.size(1);

    auto launch = [&]<bool FUSE_GLU>() {
        using Kernel = GEMM::template quantize_w8a8_act_kernel<FUSE_GLU>;

        int K2 = FUSE_GLU ? K / 2 : K;
        dim3 grid(M / GEMM::WARP_M);
        dim3 block(GEMM::NUM_WARPS * 32);

        auto kernel = invoke_kernel<Kernel, const GEMM::half_t*,
                                     GEMM::packed_act_t*, GEMM::packed_ascale_t*, int, bool>;

        size_t smem = Kernel::smemSize(M, K);
        if (smem > 24 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        }

        kernel<<<grid, block, smem>>>(
            input.data_ptr<GEMM::half_t>(),
            reinterpret_cast<GEMM::packed_act_t*>(output.data_ptr<int8_t>()),
            reinterpret_cast<GEMM::packed_ascale_t*>(oscales.data_ptr()),
            K, false);

        checkCUDA(cudaGetLastError());
    };

    if (fuse_glu) launch.template operator()<true>();
    else          launch.template operator()<false>();
}

}  // namespace nunchaku_bridge
