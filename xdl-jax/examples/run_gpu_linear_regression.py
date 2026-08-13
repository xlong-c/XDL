"""运行 xdl-jax GPU YAML-free smoke."""

from __future__ import annotations

import jax

from xdl_jax import JaxTrainer, TrainerConfig
from xdl_jax.examples import LinearRegressionTask, SyntheticRegressionData


def main() -> None:
    """在第一张 GPU 上运行一个可验证的训练闭环."""

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(
            "GPU is unavailable; install xdl-jax[gpu-cuda12] or "
            "xdl-jax[gpu-cuda13] and verify the NVIDIA driver"
        )
    data = SyntheticRegressionData(
        n_samples=64,
        input_dim=3,
        batch_size=8,
        seed=4,
    )
    trainer = JaxTrainer(
        LinearRegressionTask(input_dim=3, learning_rate=0.05),
        config=TrainerConfig(max_epochs=3, platform="gpu"),
    )
    result = trainer.fit(data)
    print(
        {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices("gpu")],
            "optimizer_step": int(result.state.optimizer_step),
            "first_loss": result.history[0].loss,
            "final_loss": result.history[-1].loss,
            "first_compile_time_s": result.first_compile_time_s,
        }
    )


if __name__ == "__main__":
    main()
