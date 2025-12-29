"""
Layer Monitor Callback
监控特定网络层的权重和梯度分布

用于深度分析模型训练过程中特定层的状态变化。
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch

from .base import Callback


class LayerMonitor(Callback):
    """
    网络层监控回调

    监控指定网络层的权重和梯度统计信息，包括：
    - 权重分布（均值、标准差、最小值、最大值）
    - 梯度分布（均值、标准差、最小值、最大值）
    - 权重范数（L1, L2）
    - 梯度范数（L1, L2）
    - 权重直方图数据
    - 梯度直方图数据

    Args:
        layer_names: 要监控的层名称列表，支持通配符
        log_weights: 是否记录权重统计
        log_gradients: 是否记录梯度统计
        log_histograms: 是否记录直方图数据（用于可视化）
        logging_interval: 日志记录间隔 ('epoch', 'step')
        log_frequency: 记录频率（每N个epoch/step）
        verbose: 是否输出详细信息

    Example:
        >>> # 监控单个层
        >>> monitor = LayerMonitor(layer_names=['model.layer1.weight'])
        >>>
        >>> # 监控多个层
        >>> monitor = LayerMonitor(
        ...     layer_names=[
        ...         'model.conv1.weight',
        ...         'model.layer4.1.weight'
        ...     ],
        ...     logging_interval='epoch'
        ... )
        >>>
        >>> # 添加到训练器
        >>> trainer = Trainer(callbacks=[monitor])
    """

    def __init__(
        self,
        layer_names: Union[List[str], str],
        log_weights: bool = True,
        log_gradients: bool = True,
        log_histograms: bool = False,
        logging_interval: str = "epoch",
        log_frequency: int = 1,
        verbose: bool = False,
    ):
        super().__init__(priority=100)

        # 处理 layer_names 参数
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        self.layer_names = layer_names
        self.log_weights = log_weights
        self.log_gradients = log_gradients
        self.log_histograms = log_histograms
        self.logging_interval = logging_interval.lower()
        self.log_frequency = log_frequency
        self.verbose = verbose

        # 验证参数
        if self.logging_interval not in ["epoch", "step"]:
            raise ValueError(f"logging_interval must be 'epoch' or 'step', got {logging_interval}")

        # 状态管理
        self._state.update({"layer_stats_history": [], "matched_layers": []})

        self._logger = logging.getLogger(__name__)

    def on_train_start(self, trainer, core_module):
        """训练开始时，查找匹配的层"""
        self._find_matched_layers(core_module)

        if not self._state["matched_layers"]:
            self._logger.warning("No matching layers found for monitoring")
        elif self.verbose:
            self._logger.info(
                f"Monitoring {len(self._state['matched_layers'])} layers: "
                f"{self._state['matched_layers']}"
            )

    def on_train_epoch_end(self, trainer, core_module):
        """每个 epoch 结束时记录统计（如果间隔为 epoch）"""
        if self.logging_interval == "epoch":
            epoch = getattr(core_module, "current_epoch", 0)
            if epoch % self.log_frequency == 0:
                self._record_layer_stats(trainer, core_module, "epoch")

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """每个 batch 结束时记录统计（如果间隔为 step）"""
        if self.logging_interval == "step" and batch_idx % (self.log_frequency * 10) == 0:
            # 降低记录频率避免日志过多
            self._record_layer_stats(trainer, core_module, "step")

    def _find_matched_layers(self, core_module):
        """查找匹配的层"""
        matched = []
        model = core_module

        # 如果 core_module 包装了 model，获取实际的 model
        if hasattr(core_module, "model") and hasattr(core_module.model, "named_parameters"):
            model = core_module.model

        # 获取所有参数名称
        all_param_names = [name for name, _ in model.named_parameters()]

        # 匹配用户指定的层名称
        for pattern in self.layer_names:
            # 支持简单的通配符匹配
            for param_name in all_param_names:
                if self._match_pattern(pattern, param_name) and param_name not in matched:
                    matched.append(param_name)

        self._state["matched_layers"] = sorted(matched)

    def _match_pattern(self, pattern: str, param_name: str) -> bool:
        """
        简单的通配符匹配

        支持两种模式：
        1. 精确匹配: "model.layer1.weight"
        2. 后缀通配符: "*.weight" 匹配所有 weight 参数
        """
        if pattern == param_name:
            return True

        if pattern.startswith("*"):
            suffix = pattern[1:]
            if param_name.endswith(suffix):
                return True

        return False

    def _record_layer_stats(self, trainer, core_module, step_type: str):
        """记录层统计信息"""
        try:
            model = core_module

            # 如果 core_module 包装了 model，获取实际的 model
            if hasattr(core_module, "model") and hasattr(core_module.model, "named_parameters"):
                model = core_module.model

            stats = {
                "step_type": step_type,
                "epoch": getattr(core_module, "current_epoch", 0),
                "step": getattr(trainer, "global_step", 0),
                "timestamp": __import__("time").time(),
                "layers": {},
            }

            # 收集每个匹配层的统计信息
            for layer_name in self._state["matched_layers"]:
                layer_stats = self._compute_layer_stats(model, layer_name)
                if layer_stats:
                    stats["layers"][layer_name] = layer_stats

            # 记录到历史
            self._state["layer_stats_history"].append(stats)

            # 输出日志
            if self.verbose:
                self._log_stats_summary(stats)

        except Exception as e:
            self._logger.error(f"Error recording layer stats: {e}")

    def _compute_layer_stats(self, model, layer_name: str) -> Optional[Dict[str, Any]]:
        """
        计算单个层的统计信息

        Args:
            model: 模型
            layer_name: 层名称

        Returns:
            包含统计信息的字典
        """
        try:
            # 获取参数
            param_dict = dict(model.named_parameters())
            if layer_name not in param_dict:
                return None

            param = param_dict[layer_name]
            stats = {}

            # 权重统计
            if self.log_weights and param.data is not None:
                stats["weight"] = {
                    "mean": float(param.data.mean()),
                    "std": float(param.data.std()),
                    "min": float(param.data.min()),
                    "max": float(param.data.max()),
                    "l1_norm": float(param.data.abs().sum()),
                    "l2_norm": float(param.data.norm()),
                    "numel": int(param.data.numel()),
                }

                # 直方图数据
                if self.log_histograms:
                    hist = torch.histc(param.data.flatten(), bins=50)
                    stats["weight"]["histogram"] = hist.cpu().tolist()

            # 梯度统计
            if self.log_gradients:
                if param.grad is not None:
                    grad_norm = float(param.grad.norm())
                    grad_mean = float(param.grad.mean())

                    stats["gradient"] = {
                        "mean": grad_mean,
                        "std": float(param.grad.std()),
                        "min": float(param.grad.min()),
                        "max": float(param.grad.max()),
                        "l1_norm": float(param.grad.abs().sum()),
                        "l2_norm": grad_norm,
                        "numel": int(param.grad.numel()),
                    }

                    # 梯度直方图
                    if self.log_histograms:
                        hist = torch.histc(param.grad.flatten(), bins=50)
                        stats["gradient"]["histogram"] = hist.cpu().tolist()

                    # 检测异常梯度并给出诊断建议
                    if grad_norm < 1e-10:
                        self._diagnose_zero_gradient(param, layer_name)
                    elif grad_norm > 1000:
                        self._logger.warning(
                            f"⚠️ [{layer_name}] 梯度爆炸！norm={grad_norm:.2e}，"
                            f"建议：降低学习率或使用梯度裁剪"
                        )
                else:
                    # grad 为 None 的情况
                    if param.requires_grad:
                        self._diagnose_none_gradient(layer_name)
                    else:
                        if self.verbose:
                            self._logger.info(
                                f"ℹ️ [{layer_name}] 参数 requires_grad=False，跳过梯度检查"
                            )

            return stats

        except Exception as e:
            self._logger.warning(f"Error computing stats for {layer_name}: {e}")
            return None

    def _log_stats_summary(self, stats: Dict[str, Any]):
        """输出统计摘要"""
        epoch = stats["epoch"]
        step = stats["step"]

        for layer_name, layer_stats in stats["layers"].items():
            # 权重信息
            if "weight" in layer_stats:
                w = layer_stats["weight"]
                self._logger.info(
                    f"[Epoch {epoch}, Step {step}] {layer_name} - "
                    f"Weight: mean={w['mean']:.6e}, std={w['std']:.6e}, "
                    f"norm={w['l2_norm']:.6e}"
                )

            # 梯度信息
            if "gradient" in layer_stats:
                g = layer_stats["gradient"]
                self._logger.info(
                    f"[Epoch {epoch}, Step {step}] {layer_name} - "
                    f"Gradient: mean={g['mean']:.6e}, std={g['std']:.6e}, "
                    f"norm={g['l2_norm']:.6e}"
                )

    def get_layer_history(self, layer_name: str) -> List[Dict[str, Any]]:
        """
        获取特定层的历史记录

        Args:
            layer_name: 层名称

        Returns:
            该层的历史统计记录列表
        """
        history = []
        for record in self._state["layer_stats_history"]:
            if layer_name in record["layers"]:
                history.append(
                    {
                        "step_type": record["step_type"],
                        "epoch": record["epoch"],
                        "step": record["step"],
                        "stats": record["layers"][layer_name],
                    }
                )
        return history

    def get_current_stats(self) -> Dict[str, Any]:
        """获取最新的统计信息"""
        if self._state["layer_stats_history"]:
            return self._state["layer_stats_history"][-1]
        return {}

    def print_layer_summary(self, layer_name: Optional[str] = None):
        """
        打印层统计摘要

        Args:
            layer_name: 层名称，如果为 None 则打印所有层
        """
        history = self._state["layer_stats_history"]
        if not history:
            print("No layer statistics available.")
            return

        print("\n" + "=" * 80)
        print("LAYER MONITORING SUMMARY")
        print("=" * 80)

        current = history[-1]

        if layer_name:
            # 打印单个层的摘要
            layers_to_print = [layer_name] if layer_name in current["layers"] else []
        else:
            # 打印所有层的摘要
            layers_to_print = list(current["layers"].keys())

        for name in layers_to_print:
            stats = current["layers"][name]
            print(f"\nLayer: {name}")
            print("-" * 80)

            if "weight" in stats:
                w = stats["weight"]
                print("  Weight Statistics:")
                print(f"    Mean:   {w['mean']:.6e}")
                print(f"    Std:    {w['std']:.6e}")
                print(f"    Min:    {w['min']:.6e}")
                print(f"    Max:    {w['max']:.6e}")
                print(f"    L1 Norm: {w['l1_norm']:.6e}")
                print(f"    L2 Norm: {w['l2_norm']:.6e}")

            if "gradient" in stats:
                g = stats["gradient"]
                print("  Gradient Statistics:")
                print(f"    Mean:   {g['mean']:.6e}")
                print(f"    Std:    {g['std']:.6e}")
                print(f"    Min:    {g['min']:.6e}")
                print(f"    Max:    {g['max']:.6e}")
                print(f"    L1 Norm: {g['l1_norm']:.6e}")
                print(f"    L2 Norm: {g['l2_norm']:.6e}")

        print(f"\nTotal logged samples: {len(history)}")
        print("=" * 80)

    def export_to_csv(self, layer_name: str, save_path: str):
        """
        导出层的统计历史到 CSV 文件

        Args:
            layer_name: 层名称
            save_path: 保存路径
        """
        import csv

        history = self.get_layer_history(layer_name)
        if not history:
            self._logger.warning(f"No history found for layer {layer_name}")
            return

        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 写入表头
            writer.writerow(
                [
                    "epoch",
                    "step",
                    "step_type",
                    "weight_mean",
                    "weight_std",
                    "weight_l2_norm",
                    "grad_mean",
                    "grad_std",
                    "grad_l2_norm",
                ]
            )

            # 写入数据
            for record in history:
                stats = record["stats"]
                row = [record["epoch"], record["step"], record["step_type"]]

                if "weight" in stats:
                    row.extend(
                        [
                            stats["weight"]["mean"],
                            stats["weight"]["std"],
                            stats["weight"]["l2_norm"],
                        ]
                    )
                else:
                    row.extend([None, None, None])

                if "gradient" in stats:
                    row.extend(
                        [
                            stats["gradient"]["mean"],
                            stats["gradient"]["std"],
                            stats["gradient"]["l2_norm"],
                        ]
                    )
                else:
                    row.extend([None, None, None])

                writer.writerow(row)

        self._logger.info(f"Layer {layer_name} statistics exported to {save_path}")

    def _diagnose_zero_gradient(self, param: torch.Tensor, layer_name: str):
        """
        诊断零梯度的可能原因并提供建议

        Args:
            param: 参数张量
            layer_name: 层名称
        """
        self._logger.warning(f"⚠️ [{layer_name}] 检测到零梯度！开始诊断...")

        # 检查 1: 参数是否冻结
        if not param.requires_grad:
            self._logger.error(
                "  ❌ 原因：参数被冻结（requires_grad=False）\n"
                "  💡 解决：如果需要训练此层，设置 param.requires_grad=True"
            )
            return

        # 检查 2: 检查梯度是否真的为零（考虑数值精度）
        if param.grad is not None:
            grad_abs_sum = param.grad.abs().sum()
            if grad_abs_sum < 1e-15:
                # 梯度确实为零或接近零
                self._logger.error(
                    f"""  ❌ 原因：梯度为零或接近零（sum={grad_abs_sum:.2e}）
  💡 可能的原因和解决方案：
     1. 检查时机：确保在 backward() 之后检查梯度
     2. 数据流问题：检查是否使用了 .detach() 或 .data 切断了计算图
     3. 激活函数饱和：ReLU 可能导致死神经元，尝试 LeakyReLU
     4. 学习率过小：尝试增大学习率
     5. 权重初始化：检查权重初始化是否合适
     6. 损失函数：确认损失函数与该层有连接"""
                )
            else:
                self._logger.warning(
                    f"""  ⚠️ 梯度很小但不是完全零（sum={grad_abs_sum:.2e}）
  💡 可能是梯度消失，考虑：
     - 使用残差连接
     - 使用 BatchNorm/LayerNorm
     - 使用梯度裁剪
     - 更换激活函数（如用 ReLU 替代 Sigmoid）"""
                )

        # 检查 3: 训练模式
        self._logger.info(
            """  🔍 检查建议：
     1. 确认 model.training=True（不是 eval 模式）
     2. 确认在 training_step 中调用：
        ```python
        def training_step(self, batch, batch_idx):
            self.optimizer.zero_grad()  # ① 清空梯度
            output = self.model(batch)   # ② 前向传播
            loss = self.criterion(output, target)
            self.manual_backward(loss)  # ③ 反向传播
            # 此时检查 param.grad 才有意义
            self.optimizer.step()       # ④ 更新参数
        ```
     3. 如果使用梯度累积，确保 backward 调用次数正确
     4. 检查是否在 torch.no_grad() 上下文中"""
        )

    def _diagnose_none_gradient(self, layer_name: str):
        """
        诊断梯度为 None 的可能原因

        Args:
            layer_name: 层名称
        """
        self._logger.warning(f"⚠️ [{layer_name}] 梯度为 None! 开始诊断...")
        diagnosis_msg = """
        ❌ 原因：参数 requires_grad=True 但 grad=None
    💡 可能的原因和解决方案：
        1. 检查时机：在 backward() 之前 grad=None 是正常的
        应该在 on_train_batch_end 中检查
        2. 计算图断开：检查是否有 .detach() 或 .data 使用
        3. 损失函数问题：确认该层参与了损失计算
        4. 优化器问题：确认该层参数在优化器中
        5. torch.no_grad(): 确保前向传播不在 no_grad 上下文中

    🔍 调试步骤：
        在 training_step 中添加调试代码：
        ```python
        def training_step(self, batch, batch_idx):
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.criterion(output, target)

            # 检查计算图
            print(f'Loss requires_grad: {loss.requires_grad}')

            self.manual_backward(loss)

            # 检查梯度
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    print(f'{name}: grad_norm={param.grad.norm():.6f}')
                else:
                    print(f'{name}: grad=None')

            self.optimizer.step()
        ```
     """
        self._logger.error(diagnosis_msg)
