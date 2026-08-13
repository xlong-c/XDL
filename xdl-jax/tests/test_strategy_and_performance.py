"""device strategy 和 benchmark 报告测试."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from xdl_jax import (
    BenchmarkReport,
    DataParallelStrategy,
    MeshConfig,
    SingleDeviceStrategy,
    benchmark_callable,
)
from xdl_jax.errors import TrainingError


def test_single_device_strategy_reports_and_places_batch() -> None:
    strategy = SingleDeviceStrategy()
    strategy.setup()
    batch = strategy.place_batch({"x": jnp.ones((2, 3))})
    assert strategy.report()["strategy"] == "single_device"
    assert batch["x"].devices()


def test_data_parallel_requires_real_multiple_devices() -> None:
    if len(jax.devices()) >= 2:
        pytest.skip("single-device failure contract is not applicable")
    with pytest.raises(TrainingError, match="at least two devices"):
        DataParallelStrategy().setup()


def test_single_device_data_mesh_placement_is_explicit() -> None:
    strategy = DataParallelStrategy(MeshConfig(require_multiple_devices=False))
    strategy.setup()
    placed = strategy.place_batch({"x": jnp.ones((2, 3))})
    assert placed["x"].sharding is not None
    assert strategy.report()["axis_names"] == ["data"]


def test_benchmark_separates_first_execution() -> None:
    compiled = jax.jit(lambda value: value + 1)
    report = benchmark_callable(
        compiled,
        jnp.ones((4,)),
        warmup_steps=1,
        measured_steps=3,
    )
    assert isinstance(report, BenchmarkReport)
    assert report.first_compile_time_s > 0
    assert report.measured_steps == 3
    assert report.steady_state_p90_s >= 0


def test_two_device_data_parallel_matches_single_device() -> None:
    script = """
import json
import jax
from xdl_jax import JaxTrainer, TrainerConfig
from xdl_jax.distributed import DataParallelStrategy, MeshConfig
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData

data = SyntheticRegressionData(
    n_samples=32,
    input_dim=2,
    batch_size=8,
    seed=73,
)
single_result = JaxTrainer(
    LinearRegressionTask(input_dim=2, learning_rate=0.04),
    config=TrainerConfig(max_epochs=2),
).fit(
    SyntheticRegressionData(
        n_samples=32,
        input_dim=2,
        batch_size=8,
        seed=73,
    )
)
parallel_result = JaxTrainer(
    LinearRegressionTask(input_dim=2, learning_rate=0.04),
    config=TrainerConfig(max_epochs=2),
    strategy=DataParallelStrategy(MeshConfig(require_multiple_devices=True)),
).fit(data)
single_weight = single_result.state.model_state.params["weight"]
parallel_weight = parallel_result.state.model_state.params["weight"]
max_weight_diff = max(
    abs(left - right)
    for left, right in zip(
        single_weight.reshape(-1).tolist(),
        parallel_weight.reshape(-1).tolist(),
    )
)
print(json.dumps({
    "device_count": len(jax.devices()),
    "first_loss": parallel_result.history[0].loss,
    "last_loss": parallel_result.history[-1].loss,
    "optimizer_step": int(parallel_result.state.optimizer_step),
    "max_weight_diff": max_weight_diff,
}))
"""
    package_root = str(Path(__file__).parents[1])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = package_root
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    process = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    report = json.loads(process.stdout.strip().splitlines()[-1])
    assert report["device_count"] == 2
    assert report["optimizer_step"] == 8
    assert report["last_loss"] < report["first_loss"]
    assert report["max_weight_diff"] < 1e-5
