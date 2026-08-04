# SeFi 5B LoRA (select_cloth, 4x32GB)

走官方 [SeFi-Image](https://github.com/jmliu206/SeFi-Image) `demo/dit_finetune`,
**不要**再用 XDL 的 `sefi_image_outfit_lora_finetune.py` + `Flux2Pipeline`.

## 同步到 AutoDL

把本目录文件拷到 `/root/autodl-tmp/`:

```text
build_select_cloth_dataset.py
lora_5b_select_cloth_4x32g.yaml   # 从 configs/ 拷出并改名, 或保持路径
setup_sefi_env.sh
run_lora_5b_4x32g.sh
```

或整个目录:

```bash
scp -r examples/sefi_dit root@autodl:/root/autodl-tmp/
```

## 一步步

```bash
cd /root/autodl-tmp

# 0) HF: 浏览器 Agree 以下仓库 (同一账号)
#    SeFi-Image/SeFi-Image-5B-Base
#    SeFi-Image/SeFi-Image-SemVAE
#    facebook/dinov2-with-registers-large (通常公开)
export HF_TOKEN=hf_新token          # 勿把 token 贴聊天
export HF_ENDPOINT=https://huggingface.co
export OMP_NUM_THREADS=1

# 1) clone + 依赖
bash setup_sefi_env.sh
# 若文件在 sefi_dit/ 子目录:
# bash sefi_dit/setup_sefi_env.sh

# 2) 数据: select_cloth -> parquet
python build_select_cloth_dataset.py
# 期望 records≈5023, 写出 data/sefi_select_cloth/data/train-*.parquet

# 3) smoke (2 step, 验证能跑)
CONFIG=/root/autodl-tmp/lora_5b_select_cloth_4x32g.yaml \
SMOKE=1 bash run_lora_5b_4x32g.sh

# 4) 正式训
CONFIG=/root/autodl-tmp/lora_5b_select_cloth_4x32g.yaml \
bash run_lora_5b_4x32g.sh
```

若 yaml 仍在子目录:

```bash
CONFIG=/root/autodl-tmp/sefi_dit/configs/lora_5b_select_cloth_4x32g.yaml \
SMOKE=1 bash /root/autodl-tmp/sefi_dit/run_lora_5b_4x32g.sh
```

## 配方摘要 (4x32GB)

| 项 | 值 |
|---|---|
| base | SeFi-Image-5B-Base |
| mode | LoRA (无 DeepSpeed / 无 EMA) |
| rank / alpha | 96 / 96 |
| resolution | **1024 固定** (官方硬约束) |
| batch / accum | 1 / 2 → 有效 batch 8 |
| steps | 10000, warmup 500 |
| caption | 全部 `outfit swap` |
| world size | 4 |

官方 smoke rank16 约 21 GiB/卡. rank96 更高; OOM 时 yaml 里改:

```yaml
tuning:
  lora:
    rank: 32
    alpha: 32
```

## 推理挂 adapter

训练结束后 adapter 在:

```text
/root/autodl-tmp/outputs/sefi_dit_lora_5b_select_cloth/adapter/
```

```bash
cd /root/autodl-tmp/SeFi-Image
python inference.py \
  --checkpoint SeFi-Image/SeFi-Image-5B-Base \
  --adapter-path /root/autodl-tmp/outputs/sefi_dit_lora_5b_select_cloth/adapter \
  --prompt "outfit swap" \
  --output-dir /root/autodl-tmp/outputs/sefi_infer
```

## 与旧 XDL 入口的关系

| 文件 | 状态 |
|---|---|
| `sefi_image_outfit_lora_finetune.py` | 不适用 SeFi (Flux2Pipeline 装不上) |
| `build_sefi_image_outfit_manifest.py` | 仅 XDL jsonl, 官方训练不用 |
| 本目录 | **当前推荐路径** |
