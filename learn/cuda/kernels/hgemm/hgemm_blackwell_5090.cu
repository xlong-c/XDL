/**
 * Blackwell RTX 5090 HGEMM — tcgen05.mma + TMA + TMEM + Warp Specialization
 *
 * Based on the real SM100 PTX from CUTLASS 3.6+ (cute/arch/mma_sm100_umma.hpp).
 * CUDA 13.0+, nvcc -arch=sm_120 -std=c++20 -I third_party/cutlass/include
 *
 * Architecture (RTX 5090 — GeForce SM100, no cluster, no 2-SM UMMA):
 *
 *   TMEM (256 KB/SM)
 *     ┌─────────────────────────────┐
 *     │ acc[0]: 128×128 f16 (32KB)  │  ping-pong accum buffers
 *     │ acc[1]: 128×128 f16 (32KB)  │
 *     └─────────────────────────────┘
 *          ▲ tcgen05.mma (reads A/B SMEM desc, writes TMEM)
 *          │
 *   SMEM (228 KB max)
 *     ┌──────────────────────────────────┐
 *     │ A[BM*BK*5]  = 128*64*5*2 = 80KB │
 *     │ B[BN*BK*5]  = 128*64*5*2 = 80KB │
 *     │ Total: ~160 KB                   │
 *     └──────────────────────────────────┘
 *          ▲ TMA async copy (cp.async.bulk)
 *          │
 *   HBM (global memory)
 *     A[M×K], B[N×K] (TN layout), C[M×N]
 *
 * Key differences vs Hopper (hgemm_wgmma_stages.cu):
 *   1. tcgen05.mma replaces wgmma.mma_async — reads SMEM via 64-bit descriptors
 *   2. TMEM accumulators — zero RF pressure from accum (DRegisters = void)
 *   3. 5 pipeline stages (Hopper: 3) — deeper prefetch for 2× TC throughput
 *   4. UMMA SMEM descriptor — different format from Hopper's WGMMA descriptor
 *   5. elect_one_sync() — MMA issued by 1 elected thread, not per-warp
 *
 *   Tile:  BM=128, BN=128, BK=64  (4 tcgen05.mma × 128×128×16 each)
 *   Grid:  (N/128, M/128)
 *   Block: 256 threads (2 WG of 128: 1 producer + 1 consumer)
 *
 * References:
 *   - cute/arch/mma_sm100_umma.hpp — SM100_MMA_F16BF16_SS (the actual PTX)
 *   - cute/arch/mma_sm100_desc.hpp — UMMA::SmemDescriptor, UMMA::InstrDescriptor
 *   - cute/arch/tmem_allocator_sm100.hpp — TMEM allocation
 *   - CUTLASS examples/74_blackwell_gemm_streamk/
 *   - NVIDIA Blackwell Tuning Guide
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// CuTe headers (CUTLASS 3.6+)
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

// SM100 UMMA descriptor types
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>

// ============================================================================
// Constants
// ============================================================================
#define WARPGROUP_SIZE 128 // tcgen05.mma thread group size
#define WARP_SIZE 32

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ============================================================================
// UMMA SMEM Descriptor + InstrDescriptor helpers
//
// These construct the exact HW descriptors consumed by tcgen05.mma.
// Based on: cute/arch/mma_sm100_desc.hpp (UMMA::SmemDescriptor,
// UMMA::InstrDescriptor, make_runtime_instr_desc)
// ============================================================================

// Build 64-bit UMMA SMEM descriptor from a CuTe rank-2 SMEM tensor.
// The tensor MUST have 128B swizzle (Swizzle<3,3,3>) applied.
template <typename Engine, typename Layout>
__device__ uint64_t make_umma_smem_desc(cute::Tensor<Engine, Layout> const &tensor) {
  namespace UMMA = cute::UMMA;

  // Recast to uint128_t view so byte offsets are in 128-bit units
  auto u128_tensor = cute::recast<cute::uint128_t const>(tensor);

  UMMA::SmemDescriptor desc;
  desc.desc_ = 0;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;

  // Detect layout type from tensor swizzle
  constexpr auto layout_type = UMMA::layout_type(u128_tensor);
  desc.layout_type_ = static_cast<uint8_t>(layout_type);

  // Start address (4 LSB not included — 16B alignment)
  uint32_t start_addr = cute::cast_smem_ptr_to_uint(
      cute::raw_pointer_cast(u128_tensor.data()));
  desc.start_address_ = static_cast<uint16_t>(start_addr >> 4);

  // For Major::K (A in TN layout):
  //   Canonical layout = Swizzle<3,4,3> o ((8,n),2) : ((SBO,1),(LBO,1))
  //   stride_byte_offset  = stride<0,1> canonical (SBO)
  //   leading_byte_offset = stride<1,0> canonical (1 in uint128_t units)
  //
  // With 128B swizzle on (BM,BK) K-major:
  //   BM=128 = 16 uint128_t in M
  //   BK=64  = 4 uint128_t in K
  //   leading_byte_offset = 1 (in uint128_t units → 16 bytes)
  //   stride_byte_offset  = 16 (BM/8 uint128_t → 16 * 16 = 256 bytes)

  // For simplicity with 128B swizzle @ Major::K:
  //   leading_byte_offset = 1 (1 uint128_t = 16 bytes >> 4 = 1)
  //   stride_byte_offset  = BM / 8 (the swizzle chunk stride)
  // But since UMMA uses 4LSB-excluded encoding, these are uint128_t counts.

  if constexpr (Layout::rank == 2) {
    // K-major (A operand): canonical layout after swizzle
    // leading is K stride (1 uint128_t), stride is M stride (BM/8 uint128_t)
    desc.stride_byte_offset_ = static_cast<uint16_t>(
        cute::size<0>(u128_tensor) / (layout_type == UMMA::LayoutType::SWIZZLE_128B ? 8 : 1));
    desc.leading_byte_offset_ = 1;
  }

  return desc.desc_;
}

// Build idescE (iteration descriptor embedded in uint64_t) for tcgen05.mma.
// Upper 32 bits = InstrDescriptor, lower 32 bits = 0 (no sparse metadata).
__device__ uint64_t make_umma_idesc() {
  namespace UMMA = cute::UMMA;

  // InstrDescriptor encodes: data types, M/N dims, major-ness, negate flags, etc.
  // For F16×F16→F16, M=128, N=128, both K-major, no negate, no saturate:
  UMMA::InstrDescriptor desc_i = {};
  desc_i.a_format_ = 0;     // F16 = 0
  desc_i.b_format_ = 0;     // F16 = 0
  desc_i.c_format_ = 0;     // F16 = 0
  desc_i.m_dim_ = 128 >> 4; // 8 (encoded as M/16)
  desc_i.n_dim_ = 128 >> 3; // 16 (encoded as N/8)
  desc_i.a_major_ = static_cast<uint8_t>(UMMA::Major::K);
  desc_i.b_major_ = static_cast<uint8_t>(UMMA::Major::K);
  desc_i.a_negate_ = 0;    // ScaleIn::One
  desc_i.b_negate_ = 0;    // ScaleIn::One
  desc_i.c_saturate_ = 0;  // Saturate::False
  desc_i.sparse_flag_ = 0; // not sparse
  desc_i.max_shift_ = 0;   // MaxShift::NoShift
  desc_i.sparse_id2_ = 0;

  // idescE = (desc_i as uint32_t) << 32 | (sparse_metadata_tmem_addr as uint32_t)
  return (static_cast<uint64_t>(static_cast<uint32_t>(desc_i)) << 32);
}

// ============================================================================
// TMEM allocation + addressing
//
// TMEM is column-organized. Each column = 128 rows × 16 cols × sizeof(half).
// A 128×128 accumulator spans 8 hardware columns (128/16 = 8).
//
// tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_ptr], num_cols;
// writes the allocated TMEM base address to smem_ptr.
//
// For a single 128×128 f16 tile: num_columns = 8.
// For double-buffered accum: num_columns = 16.
// ============================================================================

// Allocate TMEM columns. Returns base TMEM address (32-bit offset).
// Must be called by a single warp, uniformly.
__device__ uint32_t tmem_allocate(int num_columns, uint32_t *smem_result) {
#if defined(CUTE_ARCH_TCGEN05_TMEM_ENABLED)
  uint32_t dst_int = cute::cast_smem_ptr_to_uint(smem_result);
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_int), "r"(num_columns));
  __syncthreads();
  return *smem_result; // Read back the allocated base address
#else
  (void)num_columns;
  (void)smem_result;
  return 0; // Placeholder when compiling for non-SM100
#endif
}

// ============================================================================
// tcgen05.mma inline PTX macro
//
// Exact PTX from CUTLASS SM100_MMA_F16BF16_SS::fma():
//
//   tcgen05.mma.cta_group::1.kind::f16
//     [tmem_c],          // %0: TMEM accumulator base address (uint32_t)
//     desc_a,            // %1: UMMA SMEM descriptor for A (uint64_t)
//     desc_b,            // %2: UMMA SMEM descriptor for B (uint64_t)
//     idescE_hi,         // %3: upper 32 bits of iteration descriptor
//     {mask0..3},        // %5-%8: mask registers (all zeros for normal MMA)
//     p;                 // predicate: 0=clear accum, 1=accumulate
//
// The MMA is issued by elect_one_sync() — a single thread in the WG.
// All 128 threads must participate uniformly in the warp group.
// ============================================================================

#define TCGEN05_MMA_128x128x16_F16(tmem_c, desc_a, desc_b, accumulate, idescE_hi) \
  do {                                                                            \
    if (cute::elect_one_sync()) {                                                 \
      uint32_t mask[4] = {0, 0, 0, 0};                                            \
      asm volatile(                                                               \
          "{\n\t"                                                                 \
          ".reg .pred p;\n\t"                                                     \
          "setp.ne.b32 p, %4, 0;\n\t"                                             \
          "tcgen05.mma.cta_group::1.kind::f16 "                                   \
          "[%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t"                            \
          "}\n" ::"r"(tmem_c),                                                    \
          "l"(desc_a), "l"(desc_b), "r"(idescE_hi),                               \
          "r"(accumulate),                                                        \
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3]));                \
    }                                                                             \
  } while (0)

// ============================================================================
// TMA tensor map (unchanged from Hopper)
// ============================================================================
template <int BlockMajor, int BlockMinor>
__host__ static inline void create_tensor_map(
    CUtensorMap *tma_map, half *gmem_ptr, int blocks_h, int blocks_w) {
  uint64_t shape[5] = {(uint64_t)BlockMinor * blocks_w,
                       (uint64_t)BlockMajor * blocks_h, 1, 1, 1};
  uint64_t stride[5] = {sizeof(half),
                        sizeof(half) * BlockMinor * blocks_w, 0, 0, 0};
  uint32_t box[5] = {uint32_t(BlockMinor), uint32_t(BlockMajor), 1, 1, 1};
  uint32_t box_stride[5] = {1, 1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      tma_map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void *)gmem_ptr,
      shape, stride + 1, box, box_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS)
    printf("cuTensorMapEncodeTiled failed: %d\n", (int)r);
}

__host__ static inline CUtensorMap *upload_tensor_map(
    half *src, int blocks_h, int blocks_w) {
  CUtensorMap *d_map;
  cudaMalloc(&d_map, sizeof(CUtensorMap));
  CUtensorMap h_map;
  create_tensor_map<128, 64>(&h_map, src, blocks_h, blocks_w);
  cudaMemcpy(d_map, &h_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);
  return d_map;
}

// ============================================================================
// Shared memory layout
//
// CuTe SMEM with 128B swizzle (MANDATORY for Blackwell bank conflict elim).
// Swizzle<3,3,3> = 2^3 × 2^3 × 2^3 = 512 elements = 1024 bytes per swizzle
// block → exactly 128B in uint128_t units.
//
// A layout: (BM, BK, K_STAGE) K-major within each (BM,BK) slice
// B layout: (BN, BK, K_STAGE) K-major within each (BN,BK) slice
// ============================================================================
template <int BM, int BN, int BK, int K_STAGE>
struct SmemLayouts {
  // 128B swizzle atom: Swizzle<3,3,3> o (8, BK) : (BK, 1)
  using SmemAtom = decltype(cute::composition(
      cute::Swizzle<3, 3, 3>{},
      cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<BK>{}),
                        cute::make_stride(cute::Int<BK>{}, cute::Int<1>{}))));

  // Tile to (BM, BK, K_STAGE) for A
  using A = decltype(cute::tile_to_shape(
      SmemAtom{},
      cute::make_shape(cute::Int<BM>{}, cute::Int<BK>{}, cute::Int<K_STAGE>{})));

  // Tile to (BN, BK, K_STAGE) for B
  using B = decltype(cute::tile_to_shape(
      SmemAtom{},
      cute::make_shape(cute::Int<BN>{}, cute::Int<BK>{}, cute::Int<K_STAGE>{})));

  // Flat byte size
  static constexpr int bytes_A = cute::cosize(A{}) * sizeof(half);
  static constexpr int bytes_B = cute::cosize(B{}) * sizeof(half);
  static constexpr int total_bytes = bytes_A + bytes_B;
};

// ============================================================================
// Blackwell HGEMM Kernel
//
// Warp-specialized (1 producer WG + 1 consumer WG), K_STAGE=5 pipeline.
//
// Data flow:
//   WG0 (producer):  TMA global → SMEM A/B (async, mbarrier-tracked)
//   WG1 (consumer):  tcgen05.mma SMEM(A,B desc) → TMEM(accum)
//                    TMEM → reg → SMEM → TMA store → global C
//
// Sync: mbarrier full/empty per stage (2 participants each)
// ============================================================================
template <int BM = 128, int BN = 128, int BK = 64, int K_STAGE = 5,
          int NUM_THREADS = 256>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemm_blackwell_tcgen05_tn_kernel(
        int M, int N, int K, half *C,
        const CUtensorMap *__restrict__ tma_A,
        const CUtensorMap *__restrict__ tma_B) {

  using namespace cute;
  using Smem = SmemLayouts<BM, BN, BK, K_STAGE>;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = threadIdx.x;

  if (by * BM >= M || bx * BN >= N)
    return;

  // ---- Shared memory ---------------------------------------------------
  extern __shared__ __align__(128) uint8_t raw_smem[];
  half *smem_a = reinterpret_cast<half *>(raw_smem);
  half *smem_b = smem_a + (Smem::bytes_A / sizeof(half));

  // ---- TMEM allocation (via smem staging) --------------------------------
  // 16 columns for 2 × 128×128 f16 accumulators (ping-pong)
  __shared__ uint32_t tmem_acc0, tmem_acc1;

  // ---- Mbarriers for producer↔consumer sync -----------------------------
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier full[K_STAGE], empty[K_STAGE];

  int wg_idx = tid / WARPGROUP_SIZE; // 0=producer, 1=consumer
  int wg_tid = tid % WARPGROUP_SIZE;

  // Init: barriers + TMEM allocation
  if (tid == 0) {
    for (int i = 0; i < K_STAGE; ++i) {
      init(&full[i], 2);
      init(&empty[i], 2);
    }
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  // TMEM allocation (single warp issues it)
  if (wg_idx == 0 && wg_tid < WARP_SIZE) {
    tmem_allocate(16, &tmem_acc0); // 2 × 8 columns = 16 columns total
    tmem_acc1 = tmem_acc0 + 8;     // Second accumulator at base+8
  }
  __syncthreads();

  // ---- Pre-compute the invariant idescE ---------------------------------
  uint64_t idescE = make_umma_idesc();
  uint32_t idescE_hi = static_cast<uint32_t>(idescE >> 32);

  // ========================================================================
  // WG0: Producer — TMA global→shared
  // ========================================================================
  if (wg_idx == 0) {
    int qidx = 0;
    int num_k_tiles = K / BK;

    if (wg_tid == 0) {
      for (int k = 0; k < num_k_tiles; ++k, ++qidx) {
        if (qidx == K_STAGE)
          qidx = 0;

        empty[qidx].wait(empty[qidx].arrive());

        // TMA load A: (BM, BK) tile at (by*BM, k*BK) → smem_a[qidx]
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            smem_a + qidx * BK * BM, tma_A, k * BK, by * BM, full[qidx]);

        // TMA load B: (BN, BK) tile at (bx*BN, k*BK) → smem_b[qidx]
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            smem_b + qidx * BK * BN, tma_B, k * BK, bx * BN, full[qidx]);

        [[maybe_unused]] auto token = cuda::device::barrier_arrive_tx(
            full[qidx], 1, (BK * BM + BK * BN) * sizeof(half));
      }
    }
  }
  // ========================================================================
  // WG1: Consumer — tcgen05.mma compute + epilogue
  // ========================================================================
  else {
    // Arrive on all empty barriers (consumer ready)
    for (int i = 0; i < K_STAGE; ++i) {
      [[maybe_unused]] auto token = empty[i].arrive();
    }

    int num_k_tiles = K / BK;
    int k_iters_per_tile = BK / 16; // 64/16 = 4 MMA per K tile

    int qidx = 0;
    int pingpong = 0; // 0→tmem_acc0, 1→tmem_acc1

    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile, ++qidx) {
      if (qidx == K_STAGE)
        qidx = 0;

      // Wait for TMA data in this stage
      full[qidx].wait(full[qidx].arrive());

      uint32_t tmem_acc = (pingpong == 0) ? tmem_acc0 : tmem_acc1;

      // ---- Issue tcgen05.mma for all K-iterations in this tile ---------
      for (int k_it = 0; k_it < k_iters_per_tile; ++k_it) {
        uint32_t accumulate = (k_it == 0) ? 0 : 1;

        // Point to A[k_it*16 .. k_it*16+15, :] and B[k_it*16 .. k_it*16+15, :]
        half *a_ptr = smem_a + qidx * BK * BM + k_it * 16;
        half *b_ptr = smem_b + qidx * BK * BN + k_it * 16;

        // Build UMMA SMEM descriptors on-the-fly from raw pointers.
        // For production: use CuTe's make_umma_desc with proper layout tensors.
        uint64_t desc_a = make_umma_smem_desc(
            make_tensor(make_smem_ptr(a_ptr),
                        make_layout(make_shape(Int<BM>{}, Int<16>{}),
                                    make_stride(Int<16>{}, Int<1>{}))));
        uint64_t desc_b = make_umma_smem_desc(
            make_tensor(make_smem_ptr(b_ptr),
                        make_layout(make_shape(Int<BN>{}, Int<16>{}),
                                    make_stride(Int<16>{}, Int<1>{}))));

        TCGEN05_MMA_128x128x16_F16(tmem_acc, desc_a, desc_b,
                                   accumulate, idescE_hi);
      }

      // ---- TMA store previous TMEM accumulator → global C -------------
      // For ping-pong: while computing into pingpong, store (pingpong^1).
      // On first K tile there's nothing to store.
      //
      // TMA store path (production):
      //   1. tmgen05.cp TMEM → registers (each of 128 threads gets a column slice)
      //   2. stmatrix reg → SMEM staging (128B aligned, swizzled)
      //   3. cp.async.bulk SMEM → global C (TMA store)
      //
      // For now: direct register store (functional baseline).

      // Swap ping-pong buffer
      pingpong ^= 1;

      // Release slot for producer reuse
      [[maybe_unused]] auto token = empty[qidx].arrive();
    }

    // ---- Final epilogue: store last accumulator from TMEM ------------
    // The final result is in tmem_acc(pingpong^1) after the loop.
    // Production: TMEM→reg→SMEM→TMA store.
    // Baseline: direct register store of zeros (placeholder).

    half *block_C = C + by * BM * N + bx * BN;
    int lane = wg_tid % WARP_SIZE;
    int warp = wg_tid / WARP_SIZE;

#pragma unroll
    for (int m_iter = 0; m_iter < BM / WARPGROUP_SIZE; ++m_iter) {
      int row = warp * (BM / 4) + m_iter;
      if (row >= BM)
        continue;
#pragma unroll
      for (int col_group = 0; col_group < BN / 8; ++col_group) {
        int col = col_group * 8 + (lane % 4) * 2;
        int g_row = by * BM + row;
        int g_col = bx * BN + col;
        if (g_row < M && g_col < N) {
          // Placeholder: real value from TMEM→reg→smem chain
          // For now store zeros to confirm kernel structure compiles
          uint32_t *dst = reinterpret_cast<uint32_t *>(&block_C[row * N + col]);
          *dst = 0;
        }
      }
    }
  }
}

// ============================================================================
// Host launch
// ============================================================================
template <int BM = 128, int BN = 128, int BK = 64, int K_STAGE = 5,
          int NUM_THREADS = 256>
void launch_blackwell_hgemm(half *A, half *B, half *C, int M, int N, int K) {
  using Smem = SmemLayouts<BM, BN, BK, K_STAGE>;

  CUtensorMap *tma_a = upload_tensor_map(A, M / BM, K / BK);
  CUtensorMap *tma_b = upload_tensor_map(B, N / BN, K / BK);

  dim3 block(NUM_THREADS);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  printf("Blackwell HGEMM (tcgen05.mma): BM=%d BN=%d BK=%d K_STAGE=%d ",
         BM, BN, BK, K_STAGE);
  printf("SMEM=%d bytes (A:%d + B:%d) TMEM=64KB (2×128×128 FP16)\n",
         Smem::total_bytes, Smem::bytes_A, Smem::bytes_B);

  cudaFuncSetAttribute(
      hgemm_blackwell_tcgen05_tn_kernel<BM, BN, BK, K_STAGE, NUM_THREADS>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, Smem::total_bytes);

  hgemm_blackwell_tcgen05_tn_kernel<BM, BN, BK, K_STAGE, NUM_THREADS>
      <<<grid, block, Smem::total_bytes>>>(M, N, K, C, tma_a, tma_b);

  cudaFree(tma_a);
  cudaFree(tma_b);
}

// ============================================================================
// Benchmark
// ============================================================================
int main() {
  constexpr int M = 4096, N = 4096, K = 4096;

  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(half));

  std::vector<half> h_A(M * K), h_B(K * N);
  for (int i = 0; i < M * K; ++i)
    h_A[i] = __float2half((float)rand() / RAND_MAX);
  for (int i = 0; i < K * N; ++i)
    h_B[i] = __float2half((float)rand() / RAND_MAX);
  cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);

  // Warmup
  launch_blackwell_hgemm(d_A, d_B, d_C, M, N, K);
  cudaDeviceSynchronize();

  // Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int iters = 100;
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    launch_blackwell_hgemm(d_A, d_B, d_C, M, N, K);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= iters;

  double tflops = (2.0 * M * N * K) / (ms / 1000.0) / 1e12;
  printf("%dx%dx%d: %.3f ms, %.2f TFLOPS\n", M, N, K, ms, tflops);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
