"""Background NVML GPU sampler.

Samples utilization, memory, power, and temperature from a daemon thread so the
training loop never blocks on NVML. All NVML symbols are resolved through the
optional ``pynvml`` dependency; when it is missing or any call fails the sampler
degrades to ``available() == False`` and ``summary()`` returns an unavailable
``GpuSummary`` without raising. The sampling thread never touches
``torch.cuda``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from statistics import mean
from typing import Deque, List, Optional

from .records import GpuSample, GpuSummary

logger = logging.getLogger(__name__)

try:  # Optional dependency: never break import.
    import pynvml
except ImportError:  # pragma: no cover - exercised only without pynvml
    pynvml = None  # type: ignore[assignment]

# Hard cap on retained samples to bound memory (~100k * few ints).
_MAX_SAMPLES: int = 100_000


def _percentile(values: List[float], pct: float) -> float:
    """Linear-interpolated percentile; returns 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


class GpuSampler:
    """Polls one NVIDIA device from a background daemon thread."""

    def __init__(self, interval_s: float = 0.5, device_index: int = 0) -> None:
        self.interval_s = float(interval_s)
        self.device_index = int(device_index)
        self._samples: Deque[GpuSample] = deque(maxlen=_MAX_SAMPLES)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handle: Optional[object] = None
        self._available = False
        if self.available():
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(  # type: ignore[union-attr]
                    self.device_index
                )
                self._available = True
            except Exception as exc:  # pragma: no cover - hardware dependent
                logger.warning("GPU %s unavailable: %s", self.device_index, exc)
                self._handle = None
                self._available = False

    @staticmethod
    def available() -> bool:
        """True only if pynvml imports, ``nvmlInit`` succeeds, and a device exists."""
        if pynvml is None:
            return False
        try:
            pynvml.nvmlInit()
        except Exception:
            return False
        try:
            return int(pynvml.nvmlDeviceGetCount()) > 0
        except Exception:
            return False

    def start(self) -> None:
        """Start the background sampling thread (idempotent)."""
        if not self._available:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="xdl-gpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the sampling thread to stop and join it."""
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self.interval_s * 4.0))

    def samples(self) -> List[GpuSample]:
        """Return a thread-safe copy of the collected samples."""
        with self._lock:
            return list(self._samples)

    def summary(self) -> GpuSummary:
        """Aggregate util/power/memory/temperature over the collected samples."""
        if not self._available:
            return GpuSummary(available=False)
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return GpuSummary(samples=0, available=True)

        utils = [s.utilization for s in samples if s.utilization is not None]
        powers = [s.power_w for s in samples if s.power_w is not None]
        mem_fracs = [
            s.memory_used_frac for s in samples if s.memory_used_frac is not None
        ]
        temps = [s.temperature for s in samples if s.temperature is not None]
        limits = [s.power_limit_w for s in samples if s.power_limit_w is not None]

        return GpuSummary(
            samples=len(samples),
            util_mean=mean(utils) if utils else None,
            util_p50=_percentile(utils, 50.0) if utils else None,
            util_p95=_percentile(utils, 95.0) if utils else None,
            power_mean=mean(powers) if powers else None,
            power_max=max(powers) if powers else None,
            power_limit=limits[-1] if limits else None,
            mem_used_frac_mean=mean(mem_fracs) if mem_fracs else None,
            temp_max=max(temps) if temps else None,
            available=True,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self._collect()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
            self._stop_event.wait(self.interval_s)

    def _collect(self) -> Optional[GpuSample]:
        if not self._available or self._handle is None:
            return None
        sample = GpuSample(timestamp=time.time())
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)  # type: ignore[union-attr]
            sample.utilization = float(util.gpu)
        except Exception:
            pass
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)  # type: ignore[union-attr]
            sample.memory_used = int(mem.used)
            sample.memory_total = int(mem.total)
        except Exception:
            pass
        try:
            milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)  # type: ignore[union-attr]
            sample.power_w = float(milliwatts) / 1000.0
        except Exception:
            pass
        try:
            limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle)  # type: ignore[union-attr]
            sample.power_limit_w = float(limit) / 1000.0
        except Exception:
            pass
        try:
            temp_sensor = getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)
            sample.temperature = float(
                pynvml.nvmlDeviceGetTemperature(  # type: ignore[union-attr]
                    self._handle, temp_sensor
                )
            )
        except Exception:
            pass
        return sample
