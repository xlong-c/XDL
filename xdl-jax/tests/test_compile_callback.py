"""compile hook 和 report callback 测试."""

from __future__ import annotations

from xdl_jax import (
    CompileReportCallback,
    JaxTrainer,
    TrainerConfig,
)
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


def test_compile_report_callback_receives_first_compile_time() -> None:
    callback = CompileReportCallback()
    result = JaxTrainer(
        LinearRegressionTask(input_dim=2),
        config=TrainerConfig(max_epochs=1),
        callbacks=[callback],
    ).fit(
        SyntheticRegressionData(
            n_samples=8,
            input_dim=2,
            batch_size=4,
            seed=81,
        )
    )
    assert callback.report is not None
    assert callback.report["first_compile_time_s"] == result.first_compile_time_s
    assert callback.report["strategy"]["strategy"] == "single_device"
