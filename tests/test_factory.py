"""
Tests for the provider factory: given AURA's configured provider name, the
factory must return the correct concrete Brain -- or raise a clear error for
an unknown provider or a missing key for the selected provider.

No real LLM is contacted: creating an OpenAI/Anthropic client is lazy, and the
tests only assert on the class returned, so the suite stays fully offline.
"""

import pytest

import aura.brain.factory as factory
from aura.brain.cloud import CloudBrain
from aura.brain.nemotron import NemotronBrain
from aura.config.settings import settings


def _set_provider(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    monkeypatch.setattr(settings, "llm_provider", provider)


def test_nemotron_provider_returns_nemotron_brain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "nvidia_api_key", "test-nvidia-key")
    _set_provider(monkeypatch, "nemotron")
    assert isinstance(factory.create_brain(), NemotronBrain)


def test_anthropic_provider_returns_cloud_brain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "test-anthropic-key")
    _set_provider(monkeypatch, "anthropic")
    assert isinstance(factory.create_brain(), CloudBrain)


def test_unknown_provider_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_provider(monkeypatch, "bogus")
    with pytest.raises(ValueError):
        factory.create_brain()


def test_missing_nvidia_key_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "nvidia_api_key", None)
    _set_provider(monkeypatch, "nemotron")
    with pytest.raises(RuntimeError, match="AURA_NVIDIA_API_KEY"):
        factory.create_brain()