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