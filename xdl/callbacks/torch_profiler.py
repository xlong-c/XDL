"""Torch profiler callback for training-step performance tracing."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from .base import Callback

if TYPE_CHECKING:
    from xdl.trainer.coreModel import CoreModel
    from xdl.trainer.trainer import Trainer


class TorchProfilerCallback(Callback):
    """Collect PyTorch profiler traces from the XDL training loop.

    The callback wraps ``torch.profiler.profile`` and calls ``profiler.step()``
    after each training batch. By default it records a short scheduled window
    and writes TensorBoard-compatible traces under ``log_dir``.
    """

    def __init__(
        self,
        log_dir: str = "logs/profiler",
        *,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        skip_first: int = 0,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        use_cuda: bool = True,
        worker_name: Optional[str] = None,
        enabled: bool = True,
        priority: int = 999,
        on_trace_ready: Optional[Callable[[Any], None]] = None,
    ) -> None:
        super().__init__(priority=priority)
        self.log_dir = Path(log_dir)
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.skip_first = skip_first
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        self.use_cuda = use_cuda
        self.worker_name = worker_name
        self.enabled = enabled
        self.on_trace_ready = on_trace_ready

        self._profiler: Optional[Any] = None
        self._is_running = False
        self._state.update(
            {
                "enabled": enabled,
                "log_dir": str(self.log_dir),
                "completed_steps": 0,
                "started": False,
                "stopped": False,
            }
        )
        self._validate_schedule()

    def on_train_start(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        del trainer, core_module
        if not self.enabled:
            return
        if self._is_running:
            return

        torch = self._import_torch()
        activities = self._build_activities(torch)
        schedule = torch.profiler.schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
            skip_first=self.skip_first,
        )
        trace_handler = self.on_trace_ready or torch.profiler.tensorboard_trace_handler(
            str(self.log_dir),
            worker_name=self.worker_name,
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
        )
        self._profiler.__enter__()
        self._is_running = True
        self._state["started"] = True
        self._state["stopped"] = False

    def on_train_batch_end(
        self,
        trainer: "Trainer",
        core_module: "CoreModel",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        del trainer, core_module, outputs, batch, batch_idx, dataloader_idx
        if not self._is_running or self._profiler is None:
            return
        self._profiler.step()
        self._state["completed_steps"] = int(self._state.get("completed_steps", 0)) + 1

    def on_train_end(self, trainer: "Trainer", core_module: "CoreModel") -> None:
        del trainer, core_module
        self._stop_profiler()

    def teardown(self, trainer: "Trainer", core_module: "CoreModel", stage: str) -> None:
        del trainer, core_module, stage
        self._stop_profiler()

    def on_exception(
        self, trainer: "Trainer", core_module: "CoreModel", exception: Exception
    ) -> None:
        del trainer, core_module, exception
        self._stop_profiler()

    def _validate_schedule(self) -> None:
        if self.wait < 0:
            raise ValueError("wait must be >= 0")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.active < 1:
            raise ValueError("active must be >= 1")
        if self.repeat < 0:
            raise ValueError("repeat must be >= 0")
        if self.skip_first < 0:
            raise ValueError("skip_first must be >= 0")

    def _import_torch(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TorchProfilerCallback requires PyTorch.") from exc
        if not hasattr(torch, "profiler"):
            raise RuntimeError("TorchProfilerCallback requires torch.profiler.")
        return torch

    def _build_activities(self, torch: Any) -> List[Any]:
        activities = [torch.profiler.ProfilerActivity.CPU]
        cuda_available = bool(
            self.use_cuda
            and hasattr(torch, "cuda")
            and callable(getattr(torch.cuda, "is_available", None))
            and torch.cuda.is_available()
        )
        if cuda_available:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        return activities

    def _stop_profiler(self) -> None:
        if not self._is_running or self._profiler is None:
            return
        profiler = self._profiler
        self._profiler = None
        self._is_running = False
        profiler.__exit__(None, None, None)
        self._state["stopped"] = True
