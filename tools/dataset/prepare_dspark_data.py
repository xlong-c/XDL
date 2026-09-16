"""DSpark / 投机解码 Drafter 训练数据准备脚本.

从高质量开源指令集(GPT-4 中文指令集与 Alpaca Cleaned 英文指令集)中下载并构建
平衡的中英通用对话/指令数据集, 供 DSpark Drafter 模型进行自监督后训练.

产物统一输出到 data/dspark_data/ 下:
- data/dspark_data/train.jsonl
- data/dspark_data/val.jsonl
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# 显式配置 (按 XDL 规范不使用 argparse, 采用代码内显式常量)
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = Path("data/dspark_data")
ZH_URL: str = "https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data_zh.json"
EN_URL: str = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"

TOTAL_TRAIN_SAMPLES: int = 10000  # 训练样本总数 (中英各 5000)
TOTAL_VAL_SAMPLES: int = 1000  # 验证样本总数 (中英各 500)
RANDOM_SEED: int = 42
REQUEST_TIMEOUT: int = 30


def _download_json(url: str, description: str) -> List[Dict[str, Any]]:
    """从指定 URL 下载 JSON 格式数据集."""
    print(f"[*] 开始下载 {description}: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
        content = response.read().decode("utf-8")
        data: List[Dict[str, Any]] = json.loads(content)
        print(f"[+] {description} 下载成功, 样本总数: {len(data)}")
        return data


def _format_sample(item: Dict[str, Any]) -> Dict[str, str]:
    """将单条样本标准化为指令对话格式文本."""
    instruction: str = str(item.get("instruction", "")).strip()
    inp: str = str(item.get("input", "")).strip()
    output: str = str(item.get("output", "")).strip()

    if inp:
        prompt = f"用户:\n{instruction}\n\n输入:\n{inp}\n\n助手:\n"
    else:
        prompt = f"用户:\n{instruction}\n\n助手:\n"

    full_text = prompt + output
    return {
        "instruction": instruction,
        "input": inp,
        "output": output,
        "text": full_text,
    }


def prepare_dspark_dataset() -> None:
    """下载, 格式化并划分 DSpark 训练与验证数据集."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    # 1. 下载中英文语料
    zh_raw = _download_json(ZH_URL, "中文 GPT-4 Alpaca 数据集")
    en_raw = _download_json(EN_URL, "英文 Alpaca Cleaned 数据集")

    # 2. 采样与均衡
    zh_needed = (TOTAL_TRAIN_SAMPLES + TOTAL_VAL_SAMPLES) // 2
    en_needed = (TOTAL_TRAIN_SAMPLES + TOTAL_VAL_SAMPLES) // 2

    random.shuffle(zh_raw)
    random.shuffle(en_raw)

    zh_selected = zh_raw[:zh_needed]
    en_selected = en_raw[:en_needed]

    # 3. 格式化样本
    all_samples: List[Dict[str, str]] = []
    for item in zh_selected:
        all_samples.append(_format_sample(item))
    for item in en_selected:
        all_samples.append(_format_sample(item))

    random.shuffle(all_samples)

    train_data = all_samples[:TOTAL_TRAIN_SAMPLES]
    val_data = all_samples[
        TOTAL_TRAIN_SAMPLES : TOTAL_TRAIN_SAMPLES + TOTAL_VAL_SAMPLES
    ]

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    print(f"[*] 写入训练集到: {train_path} ({len(train_data)} 条)")
    with open(train_path, "w", encoding="utf-8") as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[*] 写入验证集到: {val_path} ({len(val_data)} 条)")
    with open(val_path, "w", encoding="utf-8") as f:
        for sample in val_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("[✓] DSpark 数据集准备完毕!")


if __name__ == "__main__":
    prepare_dspark_dataset()
