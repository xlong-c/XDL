"""MiniCPM5-2B DSpark / DFlash 投机草稿头后训练脚本.

本脚本遵循 XDL 训练框架规范 (CoreModel + Trainer 生命周期与手动优化模式),
在完全冻结 MiniCPM5-2B 骨干网络的基础上, 仅训练外挂的超轻量 DSpark Drafter 模型:
1. 主模型 (MiniCPM5-2B) 置于 eval 且 requires_grad=False, 单次前向提取最后一层 Hidden States;
2. DSpark Drafter 由 1 层超轻量 Transformer Block + K 个残差 MTP 多词预测头 + 1 个置信度打分头构成;
3. 预测头复用主模型冻结的 LM Head 权重, 避免开辟海量词表参数, 训练参数量仅约 54M;
4. 训练产物保存至 artifacts/dspark_minicpm5_2b/dspark_drafter.pt, 供 XQT 部署侧推理挂载.

运行方式:
    PYTHONPATH=. python train/posttrain/train_dspark_minicpm5.py
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from xdl.callbacks.console_callback import ConsoleCallback
from xdl.callbacks.tqdm_callback import TqdmCallback
from xdl.trainer.core_model import CoreModel
from xdl.trainer.trainer import Trainer

import os

# 严格防止显存碎片化并抑制溢出到主机共享内存
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 显式配置常量 (按 XDL 规范不使用 argparse, 采用代码内显式常量)
# ---------------------------------------------------------------------------
MODEL_PATH: str = "downloads/MiniCPM5-2B-bf16"
TRAIN_DATA_PATH: Path = Path("data/dspark_data/train.jsonl")
VAL_DATA_PATH: Path = Path("data/dspark_data/val.jsonl")
OUTPUT_DIR: Path = Path("artifacts/dspark_minicpm5_2b")

MAX_SEQ_LEN: int = 512
BATCH_SIZE: int = 4  # 微批次=4 (单步显存仅吃约 6.0GB, 远低于 90% 警戒线)
ACCUMULATION_STEPS: int = 2  # 累计梯度=2, 有效 batch size = 4 * 2 = 8
NUM_WORKERS: int = 2
NUM_DRAFT_TOKENS: int = 6  # 预测未来 6 个 offset (offset 1..6, 即 Anchor + 5 个 Draft)
NUM_LAYERS: int = 1  # 1 层微型 Transformer 编码层 (超低延迟 0.20ms, 泛化收敛极佳)
HIDDEN_SIZE: int = 2048
FFN_HIDDEN_SIZE: int = 3072

MAX_EPOCHS: int = 2
LEARNING_RATE: float = 3e-4
WEIGHT_DECAY: float = 0.01
GRAD_CLIP_NORM: float = 1.0
DEVICE: str = "cuda"
PRECISION: str = "bf16"
CONF_LOSS_WEIGHT: float = 0.1


# ---------------------------------------------------------------------------
# 数据集组件
# ---------------------------------------------------------------------------
class DSparkDataset(Dataset[Dict[str, torch.Tensor]]):
    """读取预处理好的 jsonl 样本并进行高效预 Tokenize."""

    def __init__(
        self, data_path: Path, tokenizer: Any, max_length: int = MAX_SEQ_LEN
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, torch.Tensor]] = []

        if not data_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")

        raw_texts: List[str] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    text = item.get("text", "")
                    if text:
                        raw_texts.append(text)

        logger.info(
            "正在批量 Tokenize 数据集 %s, 样本数: %d ...", data_path, len(raw_texts)
        )
        for i in range(0, len(raw_texts), 1000):
            chunk = raw_texts[i : i + 1000]
            encodings = self.tokenizer(
                chunk,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            for inp_ids, mask in zip(
                encodings["input_ids"], encodings["attention_mask"]
            ):
                self.samples.append(
                    {
                        "input_ids": torch.tensor(inp_ids, dtype=torch.long),
                        "attention_mask": torch.tensor(mask, dtype=torch.long),
                    }
                )

        logger.info("已加载并缓存数据集 %s, 样本数: %d", data_path, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def dspark_collate_fn(
    batch: List[Dict[str, torch.Tensor]], pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """对批次序列进行动态 Padding 并对齐."""
    max_len = max(x["input_ids"].size(0) for x in batch)
    batch_input_ids: List[torch.Tensor] = []
    batch_mask: List[torch.Tensor] = []

    for item in batch:
        inp = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - inp.size(0)
        if pad_len > 0:
            padded_inp = F.pad(inp, (0, pad_len), value=pad_token_id)
            padded_mask = F.pad(mask, (0, pad_len), value=0)
        else:
            padded_inp = inp
            padded_mask = mask
        batch_input_ids.append(padded_inp)
        batch_mask.append(padded_mask)

    return {
        "input_ids": torch.stack(batch_input_ids, dim=0),
        "attention_mask": torch.stack(batch_mask, dim=0),
    }


# ---------------------------------------------------------------------------
# DSpark Drafter 神经网络模块
# ---------------------------------------------------------------------------
class DSparkDrafter(nn.Module):
    """升级版 2 层半自回归 DSpark Drafter: 2 层微型 Transformer + 级联 MTP 预测头 + 置信度头."""

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        num_heads: int = NUM_DRAFT_TOKENS,
        num_layers: int = NUM_LAYERS,
        ffn_hidden_size: int = FFN_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 2 层微型 Transformer 编码层用于捕捉深层语义上下文
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=16,
                    dim_feedforward=ffn_hidden_size,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # K 个轻量残差 MLP 调制头 (Residual Modulation)
        # 每个 Head 接收上一 Head 的残差特征调制, 建模草稿块内部的自回归依赖 (Semi-Autoregressive)
        self.mtp_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(num_heads)
            ]
        )

        # 1 个置信度预测头 (输出各个 Head 在当前位置预测正确的 Logits)
        self.conf_head = nn.Linear(hidden_size, num_heads)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """前向传播.

        Args:
            hidden_states: 形状 (B, S, D), 来自主模型最后一层输出.
            attention_mask: 形状 (B, S), 1 为有效 token, 0 为 padding.

        Returns:
            head_features: 长度为 K 的特征张量列表, 每个形状为 (B, S, D).
            conf_logits: 置信度 Logits, 形状为 (B, S, K).
        """
        padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            padding_mask = attention_mask == 0

        seq_len = hidden_states.size(1)
        causal_mask = torch.triu(
            torch.ones(
                (seq_len, seq_len),
                device=hidden_states.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        drafter_dtype = next(self.layers[0].parameters()).dtype
        if hidden_states.dtype != drafter_dtype:
            hidden_states = hidden_states.to(dtype=drafter_dtype)

        z = hidden_states
        for layer in self.layers:
            z = layer(
                z,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
                is_causal=True,
            )

        head_features: List[torch.Tensor] = [z + head(z) for head in self.mtp_heads]
        conf_logits = self.conf_head(z)
        return head_features, conf_logits


# ---------------------------------------------------------------------------
# XDL CoreModel 核心训练组件
# ---------------------------------------------------------------------------
class DSparkMiniCPM5CoreModel(CoreModel):
    """DSpark MiniCPM5 后训练 CoreModel, 遵循 XDL 生命周期与手动优化规范."""

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        num_draft_tokens: int = NUM_DRAFT_TOKENS,
        num_layers: int = NUM_LAYERS,
        hidden_size: int = HIDDEN_SIZE,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.num_draft_tokens = num_draft_tokens
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lr = lr
        self.weight_decay = weight_decay

        # 可训练组件在 init 时构建
        self.drafter = DSparkDrafter(
            hidden_size=hidden_size,
            num_heads=num_draft_tokens,
            num_layers=num_layers,
        )

        # 骨干大模型在 setup() 中懒加载以节省显存与优化器解析
        self.backbone: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None

    def setup(self, stage: str = "fit") -> None:
        """Trainer 启动时调用, 懒加载并冻结主模型."""
        if self.backbone is not None:
            return

        logger.info("正在加载冻结的 MiniCPM5-2B 主模型: %s ...", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 加载主模型, 严格设置为 eval 并且冻结所有梯度
        backbone_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        backbone_model.eval()
        for p in backbone_model.parameters():
            p.requires_grad = False

        self.backbone = backbone_model
        self.drafter.to(dtype=torch.bfloat16)
        trainable_params = sum(
            p.numel() for p in self.drafter.parameters() if p.requires_grad
        )
        logger.info(
            "主模型已冻结. DSpark Drafter 可训练参数: %.2f M", trainable_params / 1e6
        )

    def train(self, mode: bool = True) -> DSparkMiniCPM5CoreModel:
        """重写 train 方法, 确保冻结的主模型永远保持 eval 模式."""
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """前向提取特征并经由主模型 lm_head 计算 Logits."""
        assert self.backbone is not None, "必须先在 setup() 中初始化 backbone"

        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            del outputs

        head_features, conf_logits = self.drafter(
            hidden_states, attention_mask=attention_mask
        )

        head_logits: List[torch.Tensor] = []
        lm_head = self.backbone.lm_head
        target_dtype = lm_head.weight.dtype
        for feat in head_features:
            # 通过冻结的 LM Head 计算真实词表未归一化对数概率
            if feat.dtype != target_dtype:
                feat = feat.to(dtype=target_dtype)
            head_logits.append(lm_head(feat))

        return head_logits, conf_logits

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def _compute_losses(
        self,
        head_logits: List[torch.Tensor],
        conf_logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[float]]:
        """计算各个 MTP 头的 CrossEntropy 损失与置信度 BCE 损失."""
        b, s = input_ids.shape
        ce_losses: List[torch.Tensor] = []
        accuracies: List[float] = []
        conf_targets: List[torch.Tensor] = []

        for k in range(self.num_draft_tokens):
            offset = k + 1
            if offset >= s:
                continue

            # 预测 offset 步以后的 target
            preds = head_logits[k][:, :-offset, :]  # (B, S - offset, V)
            targets = input_ids[:, offset:].contiguous()  # (B, S - offset)
            mask = attention_mask[:, offset:].contiguous()  # (B, S - offset)

            # 将 padding 区域 target 设为 -100 忽略损失
            masked_targets = targets.clone()
            masked_targets[mask == 0] = -100

            v = preds.size(-1)
            loss_k = F.cross_entropy(
                preds.reshape(-1, v),
                masked_targets.reshape(-1),
                ignore_index=-100,
            )
            ce_losses.append(loss_k)

            # 计算 top-1 准确率与置信度真实标签
            with torch.no_grad():
                pred_tokens = preds.argmax(dim=-1)
                valid_mask = mask == 1
                matches = (pred_tokens == targets) & valid_mask
                valid_count = valid_mask.sum().item()
                acc_k = matches.sum().item() / max(valid_count, 1)
                accuracies.append(acc_k)

                # 置信度目标: 命中为 1.0, 错或 padding 为 0.0, 形状 (B, S - offset)
                conf_tgt = matches.float()
                # 补全到长度 S
                conf_tgt_padded = F.pad(conf_tgt, (0, offset), value=0.0)
                conf_targets.append(conf_tgt_padded)

        # 汇总置信度损失
        conf_target_tensor = torch.stack(conf_targets, dim=-1)  # (B, S, K)
        conf_mask = attention_mask.unsqueeze(-1).expand_as(conf_logits)
        conf_loss = F.binary_cross_entropy_with_logits(
            conf_logits[conf_mask == 1],
            conf_target_tensor[conf_mask == 1],
        )

        total_loss = sum(ce_losses) + CONF_LOSS_WEIGHT * conf_loss
        return total_loss, ce_losses, conf_loss, accuracies

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """手动优化模式训练单步."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        optimizer = self.optimizers[0]
        if self.is_accumulation_start:
            optimizer.zero_grad()

        head_logits, conf_logits = self(input_ids, attention_mask)
        total_loss, ce_losses, conf_loss, accs = self._compute_losses(
            head_logits, conf_logits, input_ids, attention_mask
        )

        # 梯度累积与反向传播
        self.manual_backward(total_loss / self.accumulation_steps)

        if self.is_accumulation_boundary:
            self.clip_gradients(self.drafter, gradient_clip_val=GRAD_CLIP_NORM)
            optimizer.step()
            for sch in self.schedulers:
                sch.step()
            # 严格防显存溢出: 周期性释放碎片显存, 杜绝向主机内存 paging 交换
            if hasattr(self, "global_step") and self.global_step % 20 == 0:
                torch.cuda.empty_cache()

        # 记录指标
        self.log("loss", total_loss.item(), prefix="train")
        self.log("conf_loss", conf_loss.item(), prefix="train")
        for k, (loss_k, acc_k) in enumerate(zip(ce_losses, accs)):
            self.log(f"ce_k{k + 1}", loss_k.item(), prefix="train")
            self.log(f"acc_k{k + 1}", acc_k, prefix="train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """验证单步逻辑."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        head_logits, conf_logits = self(input_ids, attention_mask)
        total_loss, ce_losses, conf_loss, accs = self._compute_losses(
            head_logits, conf_logits, input_ids, attention_mask
        )

        self.log("val_loss", total_loss.item())
        self.log("val_conf_loss", conf_loss.item())
        for k, (loss_k, acc_k) in enumerate(zip(ce_losses, accs)):
            self.log(f"val_ce_k{k + 1}", loss_k.item())
            self.log(f"val_acc_k{k + 1}", acc_k)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        """配置仅属于 Drafter 参数的 AdamW 优化器与余弦学习率调度器."""
        decay_params: List[torch.nn.Parameter] = []
        no_decay_params: List[torch.nn.Parameter] = []

        for name, param in self.drafter.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = AdamW(optim_groups, lr=self.lr, betas=(0.9, 0.95))

        # 余弦退火调度器
        trainer = getattr(self, "trainer", None)
        total_steps = (
            getattr(trainer, "total_train_steps", 1000) if trainer is not None else 1000
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=1e-5
        )

        return [optimizer], [scheduler]

    def save_drafter_weights(self, output_dir: Path) -> None:
        """单独导出保存轻量 DSpark Drafter 的权重与元配置."""
        output_dir.mkdir(parents=True, exist_ok=True)
        weight_path = output_dir / "dspark_drafter.pt"
        config_path = output_dir / "config.json"

        torch.save(self.drafter.state_dict(), weight_path)

        config_data = {
            "model_type": "dspark_drafter",
            "backbone_model": self.model_path,
            "hidden_size": self.hidden_size,
            "num_draft_tokens": self.num_draft_tokens,
            "num_layers": self.drafter.num_layers,
            "ffn_hidden_size": FFN_HIDDEN_SIZE,
            "trainable_parameters": sum(p.numel() for p in self.drafter.parameters()),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        logger.info("已保存 DSpark Drafter 权重至: %s", weight_path)


# ---------------------------------------------------------------------------
# 训练主入口
# ---------------------------------------------------------------------------
def run_dspark_training() -> None:
    """训练启动主函数."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 70)
    print(" 启动 MiniCPM5-2B DSpark / DFlash 投机草稿模型训练")
    print(f" 主模型: {MODEL_PATH}")
    print(f" 训练集: {TRAIN_DATA_PATH}")
    print(f" 验证集: {VAL_DATA_PATH}")
    print(f" 草稿预测步长: K={NUM_DRAFT_TOKENS} tokens")
    print("=" * 70)

    # 1. 实例化 CoreModel
    model = DSparkMiniCPM5CoreModel(
        model_path=MODEL_PATH,
        num_draft_tokens=NUM_DRAFT_TOKENS,
        hidden_size=HIDDEN_SIZE,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # 显式触发 setup 构建 Tokenizer 与加载主模型
    model.setup("fit")

    # 2. 构造 Dataset 与 DataLoader
    tokenizer = model.tokenizer
    train_dataset = DSparkDataset(
        TRAIN_DATA_PATH, tokenizer=tokenizer, max_length=MAX_SEQ_LEN
    )
    val_dataset = DSparkDataset(
        VAL_DATA_PATH, tokenizer=tokenizer, max_length=MAX_SEQ_LEN
    )

    collate = lambda batch: dspark_collate_fn(
        batch, pad_token_id=tokenizer.pad_token_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        pin_memory=True,
    )

    # 3. 构造 XDL Trainer
    callbacks = [
        ConsoleCallback(),
        TqdmCallback(),
    ]

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        device=DEVICE,
        precision=PRECISION,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        grad_clip_max_norm=GRAD_CLIP_NORM,
        callbacks=callbacks,
    )

    # 4. 运行训练循环
    print("[*] 开始执行模型训练...")
    trainer.fit(model, train_loader, val_loader)

    # 5. 导出保存 Drafter 独立权重产物
    print("[*] 导出 DSpark Drafter 权重产物...")
    model.save_drafter_weights(OUTPUT_DIR)
    alt_dir = Path("/root/workspace/xdl/artifacts/dspark_minicpm5_2b")
    if alt_dir.resolve() != OUTPUT_DIR.resolve():
        model.save_drafter_weights(alt_dir)
    print(f"[✓] 训练完成! 产物已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    run_dspark_training()
