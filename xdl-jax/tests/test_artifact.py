"""模型侧产物导出测试."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xdl_jax import (
    JaxTrainer,
    TrainerConfig,
    export_model_artifact,
    materialize_model_artifact,
    read_model_artifact_metadata,
)
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


def test_model_artifact_contains_only_model_side_arrays(tmp_path: Path) -> None:
    data = SyntheticRegressionData(
        n_samples=16,
        input_dim=2,
        batch_size=4,
        seed=51,
    )
    trainer = JaxTrainer(
        LinearRegressionTask(input_dim=2),
        config=TrainerConfig(max_epochs=1),
    )
    result = trainer.fit(data)
    artifact = materialize_model_artifact(
        result.state.model_state,
        adapter=trainer.adapter,
        metadata={"consumer": "model-side-tool"},
    )
    assert artifact.parameters
    assert artifact.mutable == {}
    assert "restore_scope" in artifact.metadata
    assert "sharding" in artifact.metadata
    assert "optimizer_state" not in artifact.metadata
    assert "rng_key" not in artifact.metadata

    report = export_model_artifact(
        tmp_path / "artifact",
        result.state.model_state,
        adapter=trainer.adapter,
    )
    assert report.parameter_count == len(artifact.parameters)
    assert Path(report.weights_path).exists()
    metadata = read_model_artifact_metadata(tmp_path / "artifact")
    assert metadata["restore_scope"] == "model_state_only"
    loaded = np.load(report.weights_path)
    assert loaded.files
    assert json.loads(Path(report.metadata_path).read_text())["format_version"] == 1
