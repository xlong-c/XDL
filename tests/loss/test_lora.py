import pytest

from xdl.post_training.lora import LoRAParameters, normalize_lora_parameters


def test_normalize_lora_parameters_accepts_string_and_coerces_values() -> None:
    params = normalize_lora_parameters("to_q, to_v", "8", "16", "0.05")

    assert params == LoRAParameters(
        target_modules=("to_q", "to_v"),
        rank=8,
        alpha=16,
        dropout=0.05,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_modules": [], "rank": 8, "alpha": 16, "dropout": 0.0}, "target_modules"),
        ({"target_modules": ["to_q"], "rank": 0, "alpha": 16, "dropout": 0.0}, "rank"),
        ({"target_modules": ["to_q"], "rank": 8, "alpha": 0, "dropout": 0.0}, "alpha"),
        ({"target_modules": ["to_q"], "rank": 8, "alpha": 16, "dropout": 1.0}, "dropout"),
    ],
)
def test_normalize_lora_parameters_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        normalize_lora_parameters(**kwargs)
