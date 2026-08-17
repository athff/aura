"""
Shared fixtures for the AURA test suite.

`StubBrain` lets us test the engine and factory WITHOUT calling a real LLM,
so the whole unit-test suite runs offline with no API keys and no network.
"""

import pytest

from aura.brain.base import Brain


class StubBrain(Brain):
    """A brain that returns a canned reply and records every call it receives."""

    REPLY = "AURA: stub reply"

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def think(self, messages: list[dict]) -> str:
        # Record a copy so later assertions can't be confused by mutation.
        self.calls.append([dict(m) for m in messages])
        return self.REPLY


@pytest.fixture
def stub_brain() -> StubBrain:
    return StubBrain()


@pytest.fixture(autouse=True)
def _isolate_kokoro_pipeline_cache():
    """Reset AURA's warm Kokoro pipeline cache before every test.

    The production voice path shares ONE process-wide Kokoro pipeline so the model
    is never reloaded per response. Tests, however, must stay independent -- e.g.
    several Kokoro tests intentionally re-use the same model/voices paths while
    swapping in different fake engines. Clearing the cache first restores the
    prior "each case loads its own engine" behaviour so no test sees a pipeline
    left behind by an earlier one.
    """
    from aura.interaction.voice import providers as _voice_providers

    _voice_providers._reset_kokoro_pipeline_cache()
    yield
    _voice_providers._reset_kokoro_pipeline_cache()