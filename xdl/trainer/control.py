"""Thread-safe control signals for long-running training jobs."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Literal, Optional

ControlState = Literal["running", "paused", "stopping", "finished"]
StateObserver = Callable[[str, int], None]


class TrainerControl:
    """Coordinate pause, resume, and safe-stop requests at batch boundaries.

    An observer may publish control-state events. It must be non-blocking because
    state transitions are reported while the condition lock is held.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._state: ControlState = "running"
        self._stop_requested = False
        self._save_checkpoint_requested = False
        self._stop_mode: Optional[str] = None
        self._sequence = 0
        self._pause_announced = False
        self._observer: Optional[StateObserver] = None

    @property
    def state(self) -> ControlState:
        """Return the current control state."""
        with self._condition:
            return self._state

    @property
    def stop_requested(self) -> bool:
        """Return whether a safe stop has been requested."""
        with self._condition:
            return self._stop_requested

    @property
    def save_checkpoint_requested(self) -> bool:
        """Return whether the safe stop should persist a final checkpoint."""
        with self._condition:
            return self._save_checkpoint_requested

    @property
    def stop_mode(self) -> Optional[str]:
        """Return the requested terminal action, if any."""
        with self._condition:
            return self._stop_mode

    def set_observer(self, observer: Optional[StateObserver]) -> None:
        """Set a non-blocking callback for control state changes."""
        with self._condition:
            self._observer = observer

    def request_pause(self) -> None:
        """Pause at the next safe point while retaining the worker process."""
        with self._condition:
            if not self._stop_requested and self._state == "running":
                self._sequence += 1
                self._state = "paused"
                self._pause_announced = False
                self._notify_locked("pause_requested")

    def resume(self) -> None:
        """Resume a worker waiting at a pause boundary."""
        with self._condition:
            if not self._stop_requested and self._state == "paused":
                self._sequence += 1
                self._state = "running"
                self._pause_announced = False
                self._notify_locked("running")
                self._condition.notify_all()

    def request_stop(self, *, save_checkpoint: bool = True, mode: str = "stop") -> None:
        """Request a safe stop after the current batch."""
        if mode not in {"stop", "checkpoint_pause"}:
            raise ValueError(f"Unsupported stop mode: {mode!r}")
        with self._condition:
            if not self._stop_requested:
                self._sequence += 1
            self._stop_requested = True
            if self._stop_mode is None or mode == "checkpoint_pause":
                self._stop_mode = mode
            self._save_checkpoint_requested = self._save_checkpoint_requested or bool(
                save_checkpoint
            )
            self._state = "stopping"
            self._notify_locked("stopping")
            self._condition.notify_all()

    def wait_if_paused(self, timeout: Optional[float] = None) -> bool:
        """Wait while paused and return whether training may continue.

        A finite timeout is useful for worker health checks. ``None`` waits
        indefinitely until resume or stop.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            if self._state == "paused" and not self._pause_announced:
                self._pause_announced = True
                self._notify_locked("paused")
            while self._state == "paused" and not self._stop_requested:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return not self._stop_requested

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the control state."""
        with self._condition:
            return {
                "state": self._state,
                "stop_requested": self._stop_requested,
                "save_checkpoint_requested": self._save_checkpoint_requested,
                "stop_mode": self._stop_mode,
                "sequence": self._sequence,
            }

    def apply_command(self, command: str, *, save_checkpoint: bool = True) -> None:
        """Apply a trusted command received from an external worker channel."""
        if command == "pause":
            self.request_pause()
        elif command == "resume":
            self.resume()
        elif command in {"stop", "checkpoint_pause"}:
            self.request_stop(
                save_checkpoint=save_checkpoint or command == "checkpoint_pause",
                mode=command,
            )
        else:
            raise ValueError(f"Unsupported trainer control command: {command!r}")

    def apply_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Apply a rank-zero control snapshot on all distributed workers."""
        state = snapshot.get("state")
        if state not in {"running", "paused", "stopping", "finished"}:
            raise ValueError(f"Invalid trainer control state: {state!r}")
        with self._condition:
            sequence = int(snapshot.get("sequence", 0))
            if sequence < self._sequence:
                return
            changed = state != self._state
            self._sequence = sequence
            self._state = state
            self._stop_requested = bool(snapshot.get("stop_requested", False))
            self._save_checkpoint_requested = bool(
                snapshot.get("save_checkpoint_requested", False)
            )
            stop_mode = snapshot.get("stop_mode")
            if stop_mode in {"stop", "checkpoint_pause"}:
                self._stop_mode = stop_mode
            if changed and state in {"running", "stopping"}:
                self._notify_locked(state)
            self._condition.notify_all()

    def mark_finished(self) -> None:
        """Mark the control lifecycle as finished and wake waiters."""
        with self._condition:
            self._state = "finished"
            self._condition.notify_all()

    def _notify_locked(self, state: str) -> None:
        if self._observer is not None:
            self._observer(state, self._sequence)


__all__ = ["ControlState", "StateObserver", "TrainerControl"]
