#include "utils.h"

void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    int stages);

void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_v2(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    int stages);

void flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_v3(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    int stages);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(
      flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv);
  TORCH_BINDING_COMMON_EXTENSION(
      flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_v2);
  TORCH_BINDING_COMMON_EXTENSION(
      flash_attn_mma_stages_split_q_tiling_qkv_acc_f32_swizzle_qkv_v3);
}
