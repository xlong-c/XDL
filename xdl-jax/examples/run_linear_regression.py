"""运行 xdl-jax CPU YAML 示例."""

from pathlib import Path

from xdl_jax import JaxTrainer
from xdl_jax.config import setup_from_yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    setup = setup_from_yaml(root / "examples" / "linear_regression.yaml")
    trainer = JaxTrainer.from_setup(setup)
    result = trainer.fit_from_setup(setup)
    print(
        {
            "optimizer_step": result.state.optimizer_step,
            "final_loss": result.history[-1].loss,
            "first_compile_time_s": result.first_compile_time_s,
        }
    )


if __name__ == "__main__":
    main()
