"""
Timer Callback
记录训练过程中各个阶段的时间消耗

参考 PyTorch Lightning 的 Timer 实现。
"""

import time
from typing import Dict, Any, List
from .base import Callback


class Timer(Callback):
    """
    训练时间统计回调

    参考 Lightning 的 Timer, 提供：
    - 训练、验证、测试各阶段的时间统计
    - 批次处理速度统计
    - ETA(预计完成时间)估算
    - 详细的性能分析

    Args:
        verbose: 是否输出详细信息
        track_eta: 是否跟踪ETA
        track_batch_speed: 是否跟踪批次处理速度
    """

    def __init__(self, verbose: bool = True, track_eta: bool = True, track_batch_speed: bool = True):
        super().__init__()

        self.verbose = verbose
        self.track_eta = track_eta
        self.track_batch_speed = track_batch_speed

        # 状态管理
        self._state.update({
            'timers': {},
            'epoch_times': [],
            'batch_times': [],
            'stage_start_times': {},
            'total_samples': 0,
            'processed_samples': 0
        })

        # 内部计时器
        self._timers = {}
        self._stage_timers = {}

    # ========================================
    # 时间管理辅助方法
    # ========================================

    def _start_timer(self, name: str) -> float:
        """开始计时"""
        start_time = time.time()
        self._timers[name] = start_time
        return start_time

    def _end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name not in self._timers:
            return 0.0

        elapsed = time.time() - self._timers[name]
        del self._timers[name]

        # 保存到状态
        if 'timers' not in self._state:
            self._state['timers'] = {}
        self._state['timers'][name] = elapsed

        return elapsed

    def _get_elapsed_time(self, name: str) -> float:
        """获取已运行时间"""
        if name in self._timers:
            return time.time() - self._timers[name]
        elif name in self._state.get('timers', {}):
            return self._state['timers'][name]
        return 0.0

    # ========================================
    # 训练生命周期钩子
    # ========================================

    def on_train_start(self, trainer, core_module):
        """训练开始时初始化计时器"""
        self._start_timer('total_training')
        self._state['epoch_times'] = []
        self._state['batch_times'] = []

        if self.verbose:
            print("Timer started - training initialized")

    def on_train_end(self, trainer, core_module):
        """训练结束时记录总时间"""
        total_time = self._end_timer('total_training')
        self._state['total_training_time'] = total_time

        if self.verbose:
            print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

    def on_train_epoch_start(self, trainer, core_module):
        """epoch开始时计时"""
        epoch_num = getattr(core_module, 'current_epoch', 0)
        self._start_timer(f'epoch_{epoch_num}')
        self._state['epoch_start_time'] = time.time()
        self._state['epoch_batch_times'] = []

    def on_train_epoch_end(self, trainer, core_module):
        """epoch结束时记录时间"""
        epoch_num = getattr(core_module, 'current_epoch', 0)
        epoch_time = self._end_timer(f'epoch_{epoch_num}')

        epoch_info = {
            'epoch': epoch_num,
            'duration': epoch_time,
            'batch_times': self._state.get('epoch_batch_times', []),
            'timestamp': time.time()
        }

        self._state['epoch_times'].append(epoch_info)

        if self.verbose:
            avg_batch_time = sum(epoch_info['batch_times']) / len(epoch_info['batch_times']) if epoch_info['batch_times'] else 0
            print(f"Epoch {epoch_num} completed in {epoch_time:.2f}s (avg batch: {avg_batch_time:.4f}s)")

    def on_train_batch_start(self, trainer, core_module, batch, batch_idx):
        """批次开始时计时"""
        self._start_timer(f'batch_{batch_idx}')

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """批次结束时记录时间"""
        batch_time = self._end_timer(f'batch_{batch_idx}')

        # 记录批次时间
        self._state['batch_times'].append(batch_time)

        if 'epoch_batch_times' in self._state:
            self._state['epoch_batch_times'].append(batch_time)

        # 更新样本计数
        if hasattr(batch, 'size') and hasattr(batch, '__len__'):
            try:
                batch_size = len(batch)
                self._state['processed_samples'] += batch_size
            except Exception:
                pass

        # 计算批次速度
        if self.track_batch_speed and batch_time > 0:
            if hasattr(batch, '__len__'):
                try:
                    batch_size = len(batch)
                    samples_per_sec = batch_size / batch_time
                    self._state['current_samples_per_sec'] = samples_per_sec
                except Exception:
                    pass

    # ========================================
    # 验证生命周期钩子
    # ========================================

    def on_validation_start(self, trainer, core_module):
        """验证开始时计时"""
        self._start_timer('validation')

    def on_validation_end(self, trainer, core_module):
        """验证结束时记录时间"""
        val_time = self._end_timer('validation')
        self._state['last_validation_time'] = val_time

        if self.verbose:
            print(f"Validation completed in {val_time:.2f}s")

    # ========================================
    # 统计和分析方法
    # ========================================

    def get_training_speed(self) -> Dict[str, float]:
        """获取训练速度统计"""
        stats = {}

        # 平均批次时间
        batch_times = self._state.get('batch_times', [])
        if batch_times:
            stats['avg_batch_time'] = sum(batch_times) / len(batch_times)
            stats['min_batch_time'] = min(batch_times)
            stats['max_batch_time'] = max(batch_times)

        # 平均epoch时间
        epoch_times = self._state.get('epoch_times', [])
        if epoch_times:
            durations = [e['duration'] for e in epoch_times]
            stats['avg_epoch_time'] = sum(durations) / len(durations)
            stats['min_epoch_time'] = min(durations)
            stats['max_epoch_time'] = max(durations)

        # 样本处理速度
        if 'current_samples_per_sec' in self._state:
            stats['current_samples_per_sec'] = self._state['current_samples_per_sec']

        if self._state.get('processed_samples', 0) > 0 and self._state.get('total_training_time', 0) > 0:
            stats['avg_samples_per_sec'] = self._state['processed_samples'] / self._state['total_training_time']

        return stats

    def estimate_eta(self, current_epoch: int, total_epochs: int) -> float:
        """估算剩余训练时间"""
        epoch_times = self._state.get('epoch_times', [])
        if not epoch_times or current_epoch >= total_epochs:
            return 0.0

        # 使用最近的epoch平均时间
        recent_epochs = epoch_times[-min(5, len(epoch_times)):]  # 最近5个epoch
        avg_epoch_time = sum(e['duration'] for e in recent_epochs) / len(recent_epochs)

        remaining_epochs = total_epochs - current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        return eta_seconds

    def format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.0f}s"
        else:
            hours = int(seconds // 3600)
            remaining = seconds % 3600
            minutes = int(remaining // 60)
            secs = remaining % 60
            return f"{hours}h {minutes}m {secs:.0f}s"

    def print_time_summary(self):
        """打印时间统计摘要"""
        if not self._state.get('epoch_times') and not self._state.get('batch_times'):
            print("No timing data available.")
            return

        print("\n" + "="*60)
        print("TRAINING TIME SUMMARY")
        print("="*60)

        # 总训练时间
        total_time = self._state.get('total_training_time', 0)
        if total_time > 0:
            print(f"\nTotal training time: {self.format_time(total_time)}")

        # Epoch统计
        epoch_times = self._state.get('epoch_times', [])
        if epoch_times:
            durations = [e['duration'] for e in epoch_times]
            avg_epoch = sum(durations) / len(durations)
            print(f"Epochs completed: {len(epoch_times)}")
            print(f"Average epoch time: {self.format_time(avg_epoch)}")
            print(f"Fastest epoch: {self.format_time(min(durations))}")
            print(f"Slowest epoch: {self.format_time(max(durations))}")

        # Batch统计
        batch_times = self._state.get('batch_times', [])
        if batch_times:
            total_batches = len(batch_times)
            avg_batch = sum(batch_times) / total_batches
            print(f"\nTotal batches: {total_batches}")
            print(f"Average batch time: {avg_batch:.4f}s")
            print(f"Fastest batch: {min(batch_times):.4f}s")
            print(f"Slowest batch: {max(batch_times):.4f}s")

            # 速度统计
            current_epoch = len(epoch_times) if epoch_times else 0
            if current_epoch > 0 and self._state.get('processed_samples', 0) > 0:
                avg_samples_per_sec = self._state['processed_samples'] / total_time
                print(f"Average processing speed: {avg_samples_per_sec:.1f} samples/sec")

        # 验证时间
        val_time = self._state.get('last_validation_time', 0)
        if val_time > 0:
            print(f"Last validation time: {self.format_time(val_time)}")

        print("="*60)

    def get_eta_report(self, current_epoch: int, total_epochs: int) -> str:
        """获取ETA报告"""
        eta_seconds = self.estimate_eta(current_epoch, total_epochs)
        if eta_seconds > 0:
            return f"ETA: {self.format_time(eta_seconds)}"
        return "ETA: Unknown"