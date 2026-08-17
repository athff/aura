"""Tests for the configuration validation improvements in core/settings.py.

The real `settings` singleton uses the values in `.env`. We also verify the
validators reject clearly-misconfigured values, so a broken .env fails FAST at
startup instead of costing a live LLM request.
"""

import pytest

from aura.config.settings import Settings, settings


def test_defaults_are_valid() -> None:
    # The singleton already validated at import; just confirm sane defaults.
    # Confirm sane defaults. Temperature is kept low (0.2) so simple factual
    # questions get direct, consistent answers instead of creative drift.
    assert settings.temperature == 0.2
    assert settings.model and settings.model.strip()


def test_temperature_validator_accepts_range_bounds() -> None:
    assert Settings._check_temperature(0.0) == 0.0
    assert Settings._check_temperature(1.0) == 1.0
    assert Settings._check_temperature(2.0) == 2.0


@pytest.mark.parametrize("bad", [-0.1, 2.1, 7.0])
def test_temperature_validator_rejects_out_of_range(bad: float) -> None:
    with pytest.raises(ValueError):
        Settings._check_temperature(bad)


def test_model_validator_accepts_normal_model() -> None:
    assert Settings._check_model("nvidia/nemotron-3-ultra") == "nvidia/nemotron-3-ultra"


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_model_validator_rejects_blank(bad) -> None:
    with pytest.raises(ValueError):
        Settings._check_model(bad)