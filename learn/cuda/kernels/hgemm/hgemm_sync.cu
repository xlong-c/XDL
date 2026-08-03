#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <vector_types.h>

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CA(dst, src, bytes) asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes) asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes)) //

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_8x8_slice_k16_pack_dbuf(half *A, half *B, half *C,
                                              int M, int N, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = tx + ty * blockDim.x;

  __shared__ half s_a[2][BK][BM]; // 双缓冲
  __shared__ half s_b[2][BK][BN];

  half r_a[TM]; // 数据迁移缓存
  half r_b[TN];
  half r_a_buf[TM]; // 数据计算缓存
  half r_b_buf[TN];
  half r_c[TM][TN] = {CUDART_ZERO_FP16}; // 8x8

  int idx_s_am = tid / 2; // 在块内，A的行索引, BK是16, 一次加载8个, 一行2个线程
  int idx_s_ak = (tid & 1) << 3;
  int idx_s_bn = tid / 16; // 在块内，B的列索引, BN是128, 一次加载8个, 一列16个线程
  int idx_s_bk = (tid & 15) << 3;

  int idx_g_am = idx_s_am + by * BM;
  int idx_g_bn = idx_s_bn + bx * BN;
  if (idx_g_am >= M || idx_g_bn >= N) {
    return;
  }
  // 双缓冲
  {
    int idx_g_ak = idx_s_ak; //预加载第一块, bk = 0
    int idx_g_bk = idx_s_bk;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_ak * N + idx_g_bn;
    LDST128BITS(r_a[0]) = LDST128BITS(A[addr_a]);
    LDST128BITS(s_b[0][idx_s_bk][idx_s_bn]) = LDST128BITS(B[addr_b]);
#pragma unroll
    for (int i = 1; i < 8; i++) {
      s_a[0][idx_s_ak + i][idx_s_am] = r_a[i];
    }
    __syncthreads();
  }

  for (int bk = 1; bk < K / BK; bk++) {
    int sel_smem = (bk - 1) & 1; // 当前块的SMEM选择
    int sel_buf = bk & 1;        // 当前块的缓存选择
    int idx_g_ak = idx_s_ak + bk * BK;
    int idx_g_bk = idx_s_bk + bk * BK;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_ak * N + idx_g_bn;
    LDST128BITS(r_a[0]) = LDST128BITS(A[addr_a]);
    LDST128BITS(r_b[0]) = LDST128BITS(B[addr_b]);

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_a_buf[0]) = LDST128BITS(s_a[sel_smem][tk][ty * TM]); // 读出A
      LDST128BITS(r_b_buf[0]) = LDST128BITS(s_b[sel_smem][tk][tx * TN]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_a_buf[tm], r_b_buf[tn], r_c[tm][tn]); // 矩阵乘法
        }
      }
    }

#pragma unroll
    for (int i = 1; i < 8; i++) {
      s_a[sel_buf][idx_s_ak + i][idx_s_am] = r_a[i];
    }
    LDST128BITS(s_b[sel_buf][idx_s_bk][idx_s_bn]) = LDST128BITS(r_b[0]);

    __syncthreads();
  }
  {
#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_a_buf[0]) = LDST128BITS(s_a[1][tk][ty * TM]); // 0,1,0,1 双数最后一个块肯定是1
      LDST128BITS(r_b_buf[0]) = LDST128BITS(s_b[1][tk][tx * TN]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_a_buf[tm], r_b_buf[tn], r_c[tm][tn]);
        }
      }
    }
  } // 最后一个块的计算
#pragma unroll
  for (int tm = 0; tm < TM; tm++) {
    int idx_g_cm = by * BM + ty * TM + tm;
    int idx_g_cn = bx * BN + tx * TN;
    int addr_c = idx_g_cm * N + idx_g_cn;
    LDST128BITS(C[addr_c]) = LDST128BITS(r_c[tm][0]); // r_c 8x8 128 = 8x16,刚好一行一个线程
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_8x8_slice_k16_pack_dbuf_async(half *A, half *B, half *C,
                                                    int M, int N, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = tx + ty * blockDim.x;
  __shared__ half s_a[2][BK][BM];
  __shared__ half s_b[2][BK][BN];
  half r_a_load[TM];
  half r_b_load[TN];
  half r_a_buf[TM];
  half r_b_buf[TN];
  half r_c[TM][TN] = {CUDART_ZERO_FP16};
  int idx_s_am = tid / 2;
  int idx_s_ak = (tid & 1) << 3;
  int idx_s_bn = tid / 16;
  int idx_s_bk = (tid & 15) << 3;
  int idx_g_am = idx_s_am + by * BM;
  int idx_g_bn = idx_s_bn + bx * BN;
  if (idx_g_am >= M || idx_g_bn >= N) {
    return;
  }
  {
    int idx_g_ak = idx_s_ak;
    int idx_g_bk = idx_s_bk;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_ak * N + idx_g_bn;
    uint32_t addr_b_u32 = __cvta_generic_to_shared(&s_b[0][idx_s_bk][idx_s_bn]);
    CP_ASYNC_CG(addr_b_u32, &B[addr_b], 16);
    CP_ASYNC_COMMIT_GROUP();

    LDST128BITS(r_a_load[0]) = LDST128BITS(A[addr_a]);
#pragma unroll
    for (int i = 1; i < 8; i++) {
      s_a[0][idx_s_ak + i][idx_s_am] = r_a_load[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();
  for (int bk = 1; bk < K / BK; bk++) {
    int sel_smem = (bk - 1) & 1; // 当前块的SMEM选择,错位,当前bk是1,但是是计算第0块的SMEM
    int sel_buf = bk & 1;        // 当前块的缓存选择,就是下一块
    int idx_g_ak = idx_s_ak + bk * BK;
    int idx_g_bk = idx_s_bk + bk * BK;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_ak * N + idx_g_bn; // 这些都是下一块的
    uint32_t addr_b_u32 = __cvta_generic_to_shared(&s_b[sel_buf][idx_s_bk][idx_s_bn]);
    CP_ASYNC_CG(addr_b_u32, &B[addr_b], 16); // 异步加载到s_b
    CP_ASYNC_COMMIT_GROUP();

#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_a_buf[0]) = LDST128BITS(s_a[sel_smem][tk][ty * TM]);
      LDST128BITS(r_b_buf[0]) = LDST128BITS(s_b[sel_smem][tk][tx * TN]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_a_buf[tm], r_b_buf[tn], r_c[tm][tn]);
        }
      }
    }
    LDST128BITS(r_a_load[0]) = LDST128BITS(A[addr_a]);
    LDST128BITS(r_b_load[0]) = LDST128BITS(B[addr_b]);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      s_a[sel_buf][idx_s_ak + i][idx_s_am] = r_a_load[i];
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  {
#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_a_buf[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
      LDST128BITS(r_b_buf[0]) = LDST128BITS(s_b[1][tk][tx * TN]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_a_buf[tm], r_b_buf[tn], r_c[tm][tn]);
        }
      }
    }
  }
  for (int tm = 0; tm < TM; tm++) {
    int idx_g_cm = by * BM + ty * TM + tm;
    int idx_g_cn = bx * BN + tx * TN;
    int addr_c = idx_g_cm * N + idx_g_cn;
    LDST128BITS(C[addr_c]) = LDST128BITS(r_c[tm][0]);
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 32,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_8x8_slice_k32_pack_dbuf(half *A, half *B, half *C,
                                              int M, int N, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = tx + ty * blockDim.x;
  __shared__ half s_a[2][BK][BM]; // 2 x 32 x 128
  __shared__ half s_b[2][BK][BN];
  half r_a_load[16];
  half r_a_buf[TM];
  half r_b_buf[TN];
  half r_c[TM][TN] = {CUDART_ZERO_FP16};
  // 现在每个线程需要加载的数据是多少个? 128x32=4096, 4096/((128/8)*(128*8))=4096/256=16
  int idx_s_am = tid / 2; // 32, 每次加载16个数据,一行32,需要2个线程
  int idx_s_ak = (tid & 1) << 4;
  int idx_s_bn = tid / 8; // 128, 128bit / 8bit = 16, 128/16=8
  int idx_s_bk = (tid & 7) << 4;
  int idx_g_am = idx_s_am + by * BM;
  int idx_g_bn = idx_s_bn + bx * BN;
  if (idx_g_am >= M || idx_g_bn >= N) {
    return;
  }
  {
    int idx_g_ak = idx_s_ak;
    int idx_g_bk = idx_s_bk;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_ak * N + idx_g_bn;
    LDST128BITS(r_a_load[0]) = LDST128BITS(A[addr_a + 0]);
    LDST128BITS(r_a_load[8]) = LDST128BITS(A[addr_a + 8]);
    uint32_t addr_b_u32 = __cvta_generic_to_shared(&s_b[0][idx_s_bk][idx_s_bn]);
    CP_ASYNC_CA(addr_b_u32 + 0, &B[addr_b + 0], 16); // 异步加载到s_b
    CP_ASYNC_CA(addr_b_u32 + 16, &B[addr_b + 8], 16);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      s_a[0][idx_s_ak + i][idx_s_am] = r_a_load[i];
    }
    CP_ASYNC_COMMIT_GROUP();
  }
  __syncthreads();
#pragma unroll
  for (int bk = 1; bk < K / BK; bk++) {
    int sel_smem = (bk - 1) & 1; // 当前块的SMEM选择,错位,当前bk是1,但是是计算第0块的SMEM
    int sel_buf = bk & 1;
    int idx_g_ak = idx_s_ak + bk * BK;
    int idx_g_bk = idx_s_bk + bk * BK;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_ak * N + idx_g_bn;
    uint32_t addr_b_u32 = __cvta_generic_to_shared(&s_b[sel_buf][idx_s_bk][idx_s_bn]);
    CP_ASYNC_CA(addr_b_u32 + 0, &B[addr_b + 0], 16);
    CP_ASYNC_CA(addr_b_u32 + 16, &B[addr_b + 8], 16);
    CP_ASYNC_COMMIT_GROUP();
#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_a_buf[0]) = LDST128BITS(s_a[sel_smem][tk][ty * TM]);
      LDST128BITS(r_b_buf[0]) = LDST128BITS(s_b[sel_smem][tk][tx * TN]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_a_buf[tm], r_b_buf[tn], r_c[tm][tn]);
        }
      }
    }
    LDST128BITS(r_a_load[0]) = LDST128BITS(A[addr_a + 0]);
    LDST128BITS(r_a_load[8]) = LDST128BITS(A[addr_a + 8]);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      s_a[sel_buf][idx_s_ak + i][idx_s_am] = r_a_load[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();
  {
#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_a_buf[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
      LDST128BITS(r_b_buf[0]) = LDST128BITS(s_b[1][tk][tx * TN]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_a_buf[tm], r_b_buf[tn], r_c[tm][tn]);
        }
      }
    }
  } // 循环结束
#pragma unroll
  for (int tm = 0; tm < TM; tm++) {
    int idx_g_cm = by * BM + ty * TM + tm;
    int idx_g_cn = bx * BN + tx * TN;
    int addr_c = idx_g_cm * N + idx_g_cn;
    LDST128BITS(C[addr_c]) = LDST128BITS(r_c[tm][0]);
  }
}

template <const int BM = 128, const int BN = 128, const int BK = 32,
          const int TM = 16, const int TN = 8, const int OFFSET = 0>
__global__ void hgemm_16x8_slice_k32_pack_dbuf_sync(half *A, half *B, half *C,
                                                    int M, int N, int K) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tid = tx + ty * blockDim.x;
  __shared__ half s_a[2][BK][BM + OFFSET];
  __shared__ half s_b[2][BK][BN + OFFSET];
  half r_a_load[32]; // 线程数量128/16=8 128/8=16 8*16=128个 BK*BM=32*128=4096 4096/128=32
  half r_a_buf[TM];
  half r_b_buf[TN];
  half r_c[TM][TN] = {CUDART_ZERO_FP16};
  int idx_s_am = tid;
  int idx_s_ak = 0;
  int idx_s_bn = tid / 4;
  int idx_s_bk = (tid & 7) << 5;
  int idx_g_am = idx_s_am + by * BM;
  int idx_g_bn = idx_s_bn + bx * BN;
  if (idx_g_am >= M || idx_g_bn >= N) {
    return;
  }
  {
    int idx_g_ak = idx_s_ak;
    int idx_g_bk = idx_s_bk;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_ak * N + idx_g_bn;
    uint32_t addr_b_u32 = __cvta_generic_to_shared(&s_b[0][idx_s_bk][idx_s_bn]);
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
      CP_ASYNC_CA(addr_b_u32 + i, &B[addr_b + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
      LDST128BITS(r_a_load[i]) = LDST128BITS(A[addr_a + i]);
    }
#pragma unroll
    for (int i = 0; i < 32; i++) {
      s_a[0][idx_s_ak + i][idx_s_am] = r_a_load[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();
  for (int bk = 1; bk < K / BK; bk++) {
    int sel_smem = (bk - 1) & 1;
    int sel_buf = bk & 1;
    int idx_g_ak = idx_s_ak + bk * BK;
    int idx_g_bk = idx_s_bk + bk * BK;
    int addr_a = idx_g_am * K + idx_g_ak;
    int addr_b = idx_g_ak * N + idx_g_bn;
    // 开始搬运b到缓存,这样在计算的时候,b可以异步的读取到缓存中
    uint32_t addr_b_u32 = __cvta_generic_to_shared(&s_b[sel_buf][idx_s_bk][idx_s_bn]);
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
      CP_ASYNC_CA(addr_b_u32 + i, &B[addr_b + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP(); // 提交
#pragma unroll               // 开始计算
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_a_buf[0]) = LDST128BITS(s_a[sel_smem][tk][ty * TM]);
      LDST128BITS(r_a_buf[8]) = LDST128BITS(s_a[sel_smem][tk][ty * TM + 8]); // 这里TM变成16了,要两次加载
      LDST128BITS(r_b_buf[0]) = LDST128BITS(s_b[sel_smem][tk][tx * TN]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_a_buf[tm], r_b_buf[tn], r_c[tm][tn]);
        }
      }
    }
    // 开始搬运a到缓存
#pragma unroll
    for (int i = 0; i < 32; i += 8) {
      LDST128BITS(r_a_load[i]) = LDST128BITS(A[addr_a + i]); //因为s_a的存储不连续,这里异步搬运a不合适
    }
#pragma unroll
    for (int i = 0; i < 32; i++) {
      s_a[sel_buf][idx_s_ak + i][idx_s_am] = r_a_load[i];
    }
    CP_ASYNC_WAIT_GROUP(0);
  }
  __syncthreads();
  { // 计算最后的块
#pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      LDST128BITS(r_a_buf[0]) = LDST128BITS(s_a[1][tk][ty * TM]);
      LDST128BITS(r_a_buf[8]) = LDST128BITS(s_a[1][tk][ty * TM + 8]);
      LDST128BITS(r_b_buf[0]) = LDST128BITS(s_b[1][tk][tx * TN]);
#pragma unroll
      for (int tm = 0; tm < TM; tm++) {
#pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          r_c[tm][tn] = __hfma(r_a_buf[tm], r_b_buf[tn], r_c[tm][tn]);
        }
      }
    }
  }
#pragma unroll
  for (int tm = 0; tm < TM; tm++) {
    int idx_g_cm = by * BM + ty * TM + tm;
    int idx_g_cn = bx * BN + tx * TN;
    int addr_c = idx_g_cm * N + idx_g_cn;
    LDST128BITS(C[addr_c]) = LDST128BITS(r_c[tm][0]);
  }
}
