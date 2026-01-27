#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#define WARP_SIZE = 32
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
