#include "utils.h"
#include <vector_types.h>

template <
    const int KHeadDim,
    const int kMmaAtomM,
    const int kMmaAtomN,
    const int kMmaAtomK,
    const int kMmaTileSeqLenQ,
    const int kMmaTileSeqLenK,
    const int kMmaTileSeqLenP,
    const int kMmaTileHeadDimV,
    const int kWarpTileSeqLenQ,
    const int kWarpTileSeqLenK,
    const int kWarpTileSeqLenP,
    const int kWarpTileHeadDimV,
    const int kStage, const int kPad>
__global__ __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
void flash_attn_mma_stages_split_kv_kernel(
    half *Q, half *K, half *V, half *O,
    int QKV_seqlen, int QKV_head) {}
