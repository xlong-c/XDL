"""Cross-process control file watcher for dashboard workers."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from xdl.trainer import TrainerControl

logger = logging.getLogger(__name__)


class ControlFileWatcher:
    """Poll an atomically replaced JSON command file in a daemon thread."""

    def __init__(
        self,
        path: str | Path,
        control: TrainerControl,
        *,
        interval_s: float = 0.25,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        self.path = Path(path)
        self.control = control
        self.interval_s = max(0.05, float(interval_s))
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_sequence = -1

    def start(self) -> None:
        """Start the watcher once."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="xdl-dashboard-control",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the watcher and join briefly."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self.interval_s * 4.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                command = self._read_command()
                if command is not None:
                    self._apply(command)
            except Exception as exc:  # pragma: no cover - defensive thread guard
                logger.warning("Dashboard control command failed: %s", exc)
                if self.on_error is not None:
                    self.on_error(exc)
            self._stop.wait(self.interval_s)

    def _read_command(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        raw = self.path.read_text(encoding="utf-8")
        if not raw.strip():
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("control command must be a JSON object")
        return value

    def _apply(self, command: Dict[str, Any]) -> None:
        sequence = int(command.get("seq", -1))
        if sequence <= self._last_sequence:
            return
        name = command.get("command")
        if not isinstance(name, str):
            raise ValueError("control command requires a string command")
        self.control.apply_command(
            name,
            save_checkpoint=bool(command.get("save_checkpoint", True)),
        )
        self._last_sequence = sequence


__all__ = ["ControlFileWatcher"]
