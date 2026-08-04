#!/usr/bin/env bash
# 4x32GB 启动 SeFi 5B LoRA (select_cloth).
# 依赖:
#   - 已 clone 并装好 SeFi-Image (setup_sefi_env.sh)
#   - 已生成 /root/autodl-tmp/data/sefi_select_cloth
#   - HF 已 Agree Base + SemVAE, 且 HF_TOKEN 有效

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEFI_ROOT="${SEFI_ROOT:-/root/autodl-tmp/SeFi-Image}"
DEFAULT_CONFIG_A="${SCRIPT_DIR}/configs/lora_5b_select_cloth_4x32g.yaml"
DEFAULT_CONFIG_B="/root/autodl-tmp/lora_5b_select_cloth_4x32g.yaml"
if [[ -z "${CONFIG:-}" ]]; then
  if [[ -f "${DEFAULT_CONFIG_A}" ]]; then
    CONFIG="${DEFAULT_CONFIG_A}"
  else
    CONFIG="${DEFAULT_CONFIG_B}"
  fi
fi
NPROC="${NPROC:-4}"
SMOKE="${SMOKE:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/outputs/cache/triton}"
mkdir -p "${TRITON_CACHE_DIR}"

if [[ ! -d "${SEFI_ROOT}" ]]; then
  echo "missing SeFi repo: ${SEFI_ROOT} (run setup_sefi_env.sh first)"
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "missing config: ${CONFIG}"
  exit 1
fi
if [[ ! -d /root/autodl-tmp/data/sefi_select_cloth/data ]]; then
  echo "missing dataset; run: python ${SCRIPT_DIR}/build_select_cloth_dataset.py"
  exit 1
fi
echo "config=${CONFIG}"
echo "sefi_root=${SEFI_ROOT}"
echo "nproc=${NPROC} smoke=${SMOKE}"

cd "${SEFI_ROOT}"

EXTRA_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi

torchrun --standalone --nproc_per_node="${NPROC}" \
  demo/dit_finetune/train.py \
  --config "${CONFIG}" \
  "${EXTRA_ARGS[@]}"
