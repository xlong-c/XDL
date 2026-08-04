#!/usr/bin/env bash
# 在 AutoDL 上安装 SeFi 官方 DiT LoRA 训练依赖.
# 在 /root/autodl-tmp 执行:
#   bash setup_sefi_env.sh

set -euo pipefail

ROOT="${SEFI_ROOT:-/root/autodl-tmp/SeFi-Image}"
WORKDIR="${WORKDIR:-/root/autodl-tmp}"

cd "${WORKDIR}"

if [[ ! -d "${ROOT}/.git" ]]; then
  git clone https://github.com/jmliu206/SeFi-Image.git "${ROOT}"
else
  echo "SeFi-Image already present: ${ROOT}"
fi

cd "${ROOT}"

python -m pip install -U pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || \
  python -m pip install torch torchvision

python -m pip install \
  transformers accelerate safetensors huggingface_hub omegaconf pillow \
  peft datasets pyarrow deepspeed

# 官方 smoke 锁定的 diffusers commit (提供 Flux2 / SeFi transformer 类)
python -m pip install \
  "git+https://github.com/huggingface/diffusers.git@277e3055898dd98c89cafb7df7c1d359b554df76"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p outputs/cache/triton
mkdir -p /root/autodl-tmp/outputs/cache/huggingface/{models,datasets}

echo "SeFi env ready at ${ROOT}"
echo "Next:"
echo "  1) export HF_TOKEN=...  # 浏览器已 Agree Base + SemVAE"
echo "  2) python build_select_cloth_dataset.py"
echo "  3) SMOKE=1 bash run_lora_5b_4x32g.sh"
echo "  4) bash run_lora_5b_4x32g.sh"
