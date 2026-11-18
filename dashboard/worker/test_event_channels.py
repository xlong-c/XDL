"""Tests for bounded dashboard event and control channels."""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any

from dashboard.worker.control_file import ControlFileWatcher
from dashboard.worker.event_sink import EventSink
from xdl.trainer import TrainerControl


def test_event_sink_coalesces_metrics_and_preserves_reliable_events() -> None:
    sink = EventSink(max_reliable_events=4)
    sink.emit("run_state", {"state": "running"}, reliable=True)
    sink.emit("metrics", {"global_step": 1}, reliable=False)
    sink.emit("metrics", {"global_step": 2}, reliable=False)
    sink.close()
    stream = io.StringIO()

    sink.run(stream)
    records = [
        json.loads(line.split("\t", 1)[1]) for line in stream.getvalue().splitlines()
    ]

    assert [record["type"] for record in records] == ["run_state", "metrics"]
    assert records[1]["payload"]["global_step"] == 2
    assert sink.dropped_metric_events == 1


def test_control_file_watcher_applies_only_new_commands(tmp_path: Path) -> None:
    control_path = tmp_path / "control.json"
    control = TrainerControl()
    watcher = ControlFileWatcher(control_path, control, interval_s=0.05)
    watcher.start()
    try:
        control_path.write_text(
            json.dumps({"seq": 1, "command": "pause"}), encoding="utf-8"
        )
        deadline = time.monotonic() + 1
        while control.state != "paused" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert control.state == "paused"

        control_path.write_text(
            json.dumps({"seq": 2, "command": "resume"}), encoding="utf-8"
        )
        deadline = time.monotonic() + 1
        while control.state != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert control.state == "running"
    finally:
        watcher.stop()


def test_event_sink_emit_does_not_wait_for_slow_stream() -> None:
    class SlowStream:
        def write(self, value: str) -> int:
            del value
            time.sleep(0.2)
            return 1

        def flush(self) -> None:
            return None

    sink = EventSink()
    writer = threading.Thread(target=sink.run, kwargs={"stream": SlowStream()})
    writer.start()
    start = time.monotonic()
    for step in range(100):
        sink.emit("metrics", {"global_step": step}, reliable=False)
    elapsed = time.monotonic() - start
    sink.close()
    writer.join(timeout=2)

    assert elapsed < 0.1
    assert sink.dropped_metric_events > 0
