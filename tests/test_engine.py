"""
Tests for AuraEngine (the orchestrator), using StubBrain so no network or API
keys are needed. These verify the engine's conversation-history responsibility.
"""

from conftest import StubBrain  # tests/ is on sys.path when pytest runs

from aura.core.engine import AuraEngine


def test_send_returns_the_brain_reply(stub_brain: StubBrain) -> None:
    engine = AuraEngine(brain=stub_brain)
    assert engine.send("hello") == StubBrain.REPLY


def test_brain_receives_user_message_first(stub_brain: StubBrain) -> None:
    engine = AuraEngine(brain=stub_brain)
    engine.send("hello")
    assert stub_brain.calls[0][0] == {"role": "user", "content": "hello"}


def test_history_accumulates_across_turns(stub_brain: StubBrain) -> None:
    engine = AuraEngine(brain=stub_brain)
    engine.send("first")
    engine.send("second")

    assert len(stub_brain.calls) == 2
    # Second call should already include: user("first"), assistant(reply), user("second").
    messages = stub_brain.calls[1]
    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": "first"}
    assert messages[1] == {"role": "assistant", "content": StubBrain.REPLY}
    assert messages[2] == {"role": "user", "content": "second"}