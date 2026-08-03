"""
Device Stats Monitor Callback
监控设备使用情况(CPU、GPU等)

参考 PyTorch Lightning 的 DeviceStatsMonitor 实现。
"""

import logging
import time
from typing import Any, Dict, Optional

# 尝试导入psutil
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from .base import Callback

# 尝试导入torch和CUDA
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import pynvml

    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, Exception):
    pynvml = None
    NVML_AVAILABLE = False


class DeviceStatsMonitor(Callback):
    """
    设备统计监控回调

    参考 Lightning 的 DeviceStatsMonitor, 提供：
    - CPU使用率监控
    - 内存使用监控
    - GPU使用率监控(如果可用)
    - 磁盘I/O监控

    Args:
        cpu_stats: 是否监控CPU统计
        memory_stats: 是否监控内存统计
        gpu_stats: 是否监控GPU统计
        disk_stats: 是否监控磁盘统计
        log_frequency: 日志记录频率(每N个epoch/step)
    """

    def __init__(
        self,
        cpu_stats: bool = True,
        memory_stats: bool = True,
        gpu_stats: bool = True,
        disk_stats: bool = False,
        log_frequency: int = 1,
    ):
        super().__init__()

        self.cpu_stats = cpu_stats and PSUTIL_AVAILABLE
        self.memory_stats = memory_stats and PSUTIL_AVAILABLE
        self.gpu_stats = gpu_stats
        self.disk_stats = disk_stats and PSUTIL_AVAILABLE
        self.log_frequency = log_frequency

        if not PSUTIL_AVAILABLE:
            logging.getLogger(__name__).warning(
                "psutil not installed, CPU/Memory/Disk stats will be disabled"
            )

        # 状态管理
        self._state.update({"device_stats_history": [], "baseline_stats": {}})

        self._logger = logging.getLogger(__name__)
        self._gpu_count = 0
        self._gpu_handles = []

        # 初始化GPU监控
        if self.gpu_stats and NVML_AVAILABLE and pynvml is not None:
            try:
                self._gpu_count = pynvml.nvmlDeviceGetCount()  # type: ignore[attr-defined]
                for i in range(self._gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # type: ignore[attr-defined]
                    self._gpu_handles.append(handle)
                self._logger.info(f"GPU monitoring enabled for {self._gpu_count} GPUs")
            except Exception as e:
                self._logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_stats = False

    def on_train_start(self, trainer, core_module):
        """训练开始时记录基线统计"""
        self._record_baseline_stats()

    def on_train_epoch_end(self, trainer, core_module):
        """每个epoch结束时记录设备统计"""
        if getattr(core_module, "current_epoch", 0) % self.log_frequency == 0:
            self._record_device_stats(trainer, core_module, "epoch")

    def on_train_batch_end(self, trainer, core_module, outputs, batch, batch_idx, dataloader_idx=0):
        """批次结束时记录设备统计(降低频率)"""
        if batch_idx % (self.log_frequency * 100) == 0:  # 降低批次级别记录频率
            self._record_device_stats(trainer, core_module, "step")

    def _record_baseline_stats(self):
        """记录基线统计信息"""
        baseline = {}

        # CPU基线
        if self.cpu_stats and psutil is not None:
            baseline["cpu_percent"] = psutil.cpu_percent(interval=1)
            baseline["cpu_count"] = psutil.cpu_count()

        # 内存基线
        if self.memory_stats and psutil is not None:
            memory = psutil.virtual_memory()
            baseline["memory_total"] = memory.total
            baseline["memory_available"] = memory.available
            baseline["memory_percent"] = memory.percent

        # GPU基线
        if self.gpu_stats:
            gpu_info = self._get_gpu_stats()
            baseline.update(gpu_info)

        self._state["baseline_stats"] = baseline

        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info("Device baseline stats recorded")

    def _record_device_stats(self, trainer, core_module, step_type: str):
        """记录设备统计信息"""
        try:
            stats = {
                "step_type": step_type,
                "epoch": getattr(core_module, "current_epoch", 0),
                "step": getattr(trainer, "global_step", 0),
                "timestamp": time.time(),
            }

            # CPU统计
            if self.cpu_stats and psutil is not None:
                stats["cpu_percent"] = psutil.cpu_percent()
                stats["load_avg"] = psutil.getloadavg() if hasattr(psutil, "getloadavg") else None

            # 内存统计
            if self.memory_stats and psutil is not None:
                memory = psutil.virtual_memory()
                stats["memory_used"] = memory.used
                stats["memory_available"] = memory.available
                stats["memory_percent"] = memory.percent

                # 进程内存使用
                process = psutil.Process()
                process_memory = process.memory_info()
                stats["process_memory_rss"] = process_memory.rss
                stats["process_memory_vms"] = process_memory.vms

            # GPU统计
            if self.gpu_stats:
                gpu_stats = self._get_gpu_stats()
                stats.update(gpu_stats)

            # 磁盘统计
            if self.disk_stats and psutil is not None:
                disk = psutil.disk_usage("/")
                stats["disk_used"] = disk.used
                stats["disk_free"] = disk.free
                stats["disk_percent"] = (disk.used / disk.total) * 100

            # 记录到历史
            self._state["device_stats_history"].append(stats)

            # 检查资源警告
            self._check_resource_warnings(stats)

        except Exception as e:
            self._logger.error(f"Error recording device stats: {e}")

    def _get_gpu_stats(self) -> Dict[str, Any]:
        """获取GPU统计信息"""
        gpu_stats = {}

        if not self.gpu_stats or self._gpu_count == 0:
            return gpu_stats

        for i, handle in enumerate(self._gpu_handles):
            try:
                # 获取GPU使用率
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]

                prefix = f"gpu_{i}"
                gpu_stats[f"{prefix}_utilization"] = util.gpu
                gpu_stats[f"{prefix}_memory_used"] = int(memory_info.used)
                gpu_stats[f"{prefix}_memory_total"] = int(memory_info.total)
                gpu_stats[f"{prefix}_memory_percent"] = (
                    int(memory_info.used) / int(memory_info.total)
                ) * 100

                # 获取温度(如果可用)
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)  # type: ignore[attr-defined]
                    gpu_stats[f"{prefix}_temperature"] = temp
                except Exception:
                    # 温度获取失败, 忽略但记录到日志
                    self._logger.debug(f"Could not get temperature for GPU {i}")

            except Exception as e:
                self._logger.warning(f"Error getting stats for GPU {i}: {e}")
                # 继续处理下一个GPU, 不中断整个流程
                continue

        return gpu_stats

    def _check_resource_warnings(self, stats: Dict[str, Any]):
        """检查资源使用警告"""
        # CPU警告
        if self.cpu_stats and "cpu_percent" in stats and stats["cpu_percent"] > 90:
            self._logger.warning(f"High CPU usage: {stats['cpu_percent']:.1f}%")

        # 内存警告
        if self.memory_stats and "memory_percent" in stats and stats["memory_percent"] > 90:
            self._logger.warning(f"High memory usage: {stats['memory_percent']:.1f}%")

        # GPU警告
        if self.gpu_stats:
            for key, value in stats.items():
                if key.startswith("gpu_") and key.endswith("_utilization"):
                    if value > 95:
                        self._logger.warning(f"High GPU utilization: {key} = {value}%")
                elif key.startswith("gpu_") and key.endswith("_temperature"):
                    if value > 85:  # 通常GPU安全温度阈值
                        self._logger.warning(f"High GPU temperature: {key} = {value}°C")

    def get_device_stats_history(self) -> list:
        """获取设备统计历史"""
        return self._state.get("device_stats_history", [])

    def get_current_stats(self) -> Optional[Dict[str, Any]]:
        """获取最新的设备统计"""
        history = self._state.get("device_stats_history", [])
        return history[-1] if history else None

    def print_device_summary(self):
        """打印设备使用摘要"""
        history = self._state.get("device_stats_history", [])
        if not history:
            print("No device statistics available.")
            return

        print("\n" + "=" * 70)
        print("DEVICE STATISTICS SUMMARY")
        print("=" * 70)

        current = history[-1]
        baseline = self._state.get("baseline_stats", {})

        # CPU信息
        if self.cpu_stats and "cpu_percent" in current:
            cpu_current = current["cpu_percent"]
            cpu_baseline = baseline.get("cpu_percent", 0)
            print("\nCPU Usage:")
            print(f"  Current: {cpu_current:.1f}%")
            print(f"  Baseline: {cpu_baseline:.1f}%")
            if "cpu_count" in baseline:
                print(f"  Cores: {baseline['cpu_count']}")

        # 内存信息
        if self.memory_stats and "memory_percent" in current:
            mem_current = current["memory_percent"]
            mem_baseline = baseline.get("memory_percent", 0)
            if "memory_total" in baseline:
                total_gb = baseline["memory_total"] / (1024**3)
                print("\nMemory Usage:")
                print(f"  Current: {mem_current:.1f}%")
                print(f"  Baseline: {mem_baseline:.1f}%")
                print(f"  Total: {total_gb:.1f} GB")

            if "process_memory_rss" in current:
                process_gb = current["process_memory_rss"] / (1024**3)
                print(f"  Process RSS: {process_gb:.2f} GB")

        # GPU信息
        if self.gpu_stats:
            gpu_keys = [k for k in current if k.startswith("gpu_")]
            if gpu_keys:
                print("\nGPU Usage:")
                gpu_id = {k.split("_")[1] for k in gpu_keys if k.startswith("gpu_")}
                for gid in sorted(gpu_id):
                    util = current.get(f"gpu_{gid}_utilization", 0)
                    mem_percent = current.get(f"gpu_{gid}_memory_percent", 0)
                    temp = current.get(f"gpu_{gid}_temperature", "N/A")
                    print(
                        f"  GPU {gid}: {util}% utilization, {mem_percent:.1f}% memory, {temp}°C temperature"
                    )

        print(f"\nTotal logged samples: {len(history)}")
        print("=" * 70)

    def teardown(self, trainer, core_module, stage: str):
        """清理时关闭NVML"""
        if NVML_AVAILABLE and self._gpu_handles:
            try:
                for _handle in self._gpu_handles:
                    # NVML不需要显式关闭句柄
                    pass
            except Exception as e:
                self._logger.warning(f"Error during GPU cleanup: {e}")
