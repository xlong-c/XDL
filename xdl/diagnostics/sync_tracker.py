"""Reserved-memory sawtooth detection for empty_cache thrash.

Sampling ``torch.cuda.memory_reserved()`` per step is host-side and does not
synchronize the device. Repeated sharp drops at a periodic cadence indicate a
periodic ``torch.cuda.empty_cache()`` defeating the caching allocator.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from .records import MemorySummary


class ReservedMemoryTracker:
    """Tracks reserved-memory drops to flag empty_cache thrash.

    A sawtooth event is recorded when reserved memory drops by at least
    ``drop_ratio`` relative to the recent peak and the drop exceeds
    ``min_drop_bytes``.
    """

    def __init__(
        self,
        drop_ratio: float = 0.15,
        min_drop_bytes: int = 64 * 1024 * 1024,
        window: int = 200,
    ) -> None:
        if not 0.0 < drop_ratio < 1.0:
            raise ValueError("drop_ratio must be in (0, 1)")
        self.drop_ratio = drop_ratio
        self.min_drop_bytes = min_drop_bytes
        self._window = window
        self._series: Deque[int] = deque(maxlen=window)
        self._peak: int = 0
        self._events: List[int] = []
        self._peak_before_last_drop: int = 0
        self._min_after_last_drop: int = 0

    def observe(self, reserved_bytes: int, step: Optional[int] = None) -> bool:
        """Record one reserved-memory sample; return True on a detected drop."""
        reserved = int(reserved_bytes)
        detected = False
        if self._peak > 0:
            drop = self._peak - reserved
            if drop >= self.min_drop_bytes and drop >= self.drop_ratio * self._peak:
                detected = True
                self._events.append(step if step is not None else len(self._series))
                self._peak_before_last_drop = self._peak
                self._min_after_last_drop = reserved
                # Reset peak so subsequent cycles are measured independently.
                self._peak = reserved
        if reserved > self._peak:
            self._peak = reserved
        self._series.append(reserved)
        return detected

    def summary(self) -> MemorySummary:
        peak = max(self._series) if self._series else 0
        return MemorySummary(
            sawtooth_count=len(self._events),
            reserved_peak=int(peak),
            reserved_min_after_peak=int(self._min_after_last_drop),
            sawtooth_events=list(self._events),
        )

    def reset(self) -> None:
        self._series.clear()
        self._events.clear()
        self._peak = 0
        self._peak_before_last_drop = 0
        self._min_after_last_drop = 0
