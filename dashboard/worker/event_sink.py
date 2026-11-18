"""Bounded, non-blocking event sink for dashboard training workers."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, TextIO


@dataclass(frozen=True)
class DashboardEvent:
    """Serializable event sent from a worker to the Go supervisor."""

    event_type: str
    payload: Dict[str, Any]
    reliable: bool
    timestamp: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.event_type,
            "payload": self.payload,
            "reliable": self.reliable,
            "ts": self.timestamp,
        }


class EventSink:
    """Write structured events from a background thread without blocking emitters.

    Reliable events are kept in a small unbounded control deque because they are
    infrequent. Metric events are coalesced to the newest value so a slow pipe
    cannot stall the training thread or grow memory without a bound.
    """

    def __init__(self, *, max_reliable_events: int = 1024) -> None:
        if max_reliable_events < 1:
            raise ValueError("max_reliable_events must be >= 1")
        self.max_reliable_events = int(max_reliable_events)
        self._condition = threading.Condition()
        self._reliable: Deque[DashboardEvent] = deque()
        self._latest_metric: DashboardEvent | None = None
        self._closed = False
        self.dropped_metric_events = 0
        self.dropped_reliable_events = 0

    def emit(
        self, event_type: str, payload: Dict[str, Any], reliable: bool = False
    ) -> None:
        """Enqueue an event without waiting for the pipe writer."""
        event = DashboardEvent(
            event_type=event_type,
            payload=dict(payload),
            reliable=bool(reliable),
            timestamp=time.time(),
        )
        with self._condition:
            if self._closed:
                return
            if event.reliable:
                self._reliable.append(event)
            elif event.event_type == "metrics":
                if self._latest_metric is not None:
                    self.dropped_metric_events += 1
                self._latest_metric = event
            else:
                if len(self._reliable) >= self.max_reliable_events:
                    self.dropped_reliable_events += 1
                else:
                    self._reliable.append(event)
            self._condition.notify()

    def close(self) -> None:
        """Stop accepting events after the writer drains pending data."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def run(self, stream: TextIO = sys.stdout) -> None:
        """Drain events to a line-oriented stdout protocol."""
        while True:
            event = self._next_event()
            if event is None:
                return
            line = "XDL_EVENT\t" + json.dumps(
                event.as_dict(), ensure_ascii=True, separators=(",", ":")
            )
            stream.write(line + "\n")
            stream.flush()

    def _next_event(self) -> DashboardEvent | None:
        with self._condition:
            while not self._reliable and self._latest_metric is None:
                if self._closed:
                    return None
                self._condition.wait(timeout=0.5)
            if self._reliable:
                return self._reliable.popleft()
            event = self._latest_metric
            self._latest_metric = None
            return event


__all__ = ["DashboardEvent", "EventSink"]
