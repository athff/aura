"""
Tests for AuraEngine (the orchestrator), using StubBrain so no network or API
keys are needed. These verify the engine's conversation-history responsibility
and its robustness rules (rejecting empty replies as BrainError, normalizing).
"""

from conftest import StubBrain  # tests/ is on sys.path when pytest runs

import pytest

from aura.core.engine import AuraEngine
from aura.core.errors import BrainError


class EmptyBrain(StubBrain):
    """A brain that returns a blank reply (what a misbehaving provider might)."""

    def think(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return "   "


class NoneBrain(StubBrain):
    """A brain that returns None instead of text."""

    def think(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return None  # type: ignore[return-value]


class WhitespacePadBrain(StubBrain):
    """A brain that returns padded text; the engine must strip it."""

    REPLY = "  hello world  "

    def think(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.REPLY


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
def test_send_strips_and_normalizes_provider_text() -> None:
    engine = AuraEngine(brain=WhitespacePadBrain())
    reply = engine.send("hello")
    assert reply == "hello world"
    # The assistant turn stored in history is the normalized text.
    assistant_turn = [m for m in engine._memory.messages() if m["role"] == "assistant"][-1]
    assert assistant_turn["content"] == "hello world"


def test_engine_rejects_blank_brain_reply_with_brain_error() -> None:
    engine = AuraEngine(brain=EmptyBrain())
    with pytest.raises(BrainError):
        engine.send("hello")


def test_engine_rejects_none_brain_reply_with_brain_error() -> None:
    engine = AuraEngine(brain=NoneBrain())
    with pytest.raises(BrainError):
        engine.send("hello")


def test_engine_does_not_record_history_for_empty_reply() -> None:
    engine = AuraEngine(brain=EmptyBrain())
    with pytest.raises(BrainError):
        engine.send("hello")
    # The rejected empty reply must not be appended as an assistant turn.
    assert len([m for m in engine._memory.messages() if m["role"] == "assistant"]) == 0


class TokenStreamingBrain(StubBrain):
    """A brain that streams its reply as multiple incremental fragments."""

    TOKENS = ["Hello", " there", "!", " Nice", " to", " meet", " you."]

    def think_stream(self, messages):
        self.calls.append([dict(m) for m in messages])
        for t in self.TOKENS:
            yield t


def test_send_stream_yields_incremental_tokens_and_records_history(
    stub_brain: StubBrain,
) -> None:
    engine = AuraEngine(brain=stub_brain)
    # StubBrain falls back to the default non-streaming think_stream (one chunk).
    received = list(engine.send_stream("hello"))
    assert received == [StubBrain.REPLY]
    # History is preserved: user + assistant recorded in order.
    msgs = engine._memory.messages()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0] == {"role": "user", "content": "hello"}
    assert msgs[1] == {"role": "assistant", "content": StubBrain.REPLY}


def test_send_stream_yields_each_fragment_from_a_streaming_brain() -> None:
    brain = TokenStreamingBrain()
    engine = AuraEngine(brain=brain)
    received = list(engine.send_stream("hi"))
    # The full reply equals the concatenation of the streamed fragments.
    assert received == TokenStreamingBrain.TOKENS
    assert "".join(received) == "Hello there! Nice to meet you."
    # The final complete reply (joined) is stored as the assistant turn.
    msgs = engine._memory.messages()
    assert msgs[-1] == {
        "role": "assistant",
        "content": "Hello there! Nice to meet you.",
    }


def test_send_stream_rejects_empty_reply_with_brain_error() -> None:
    engine = AuraEngine(brain=EmptyBrain())
    with pytest.raises(BrainError):
        # Exhaust the generator to trigger the post-stream empty-reply check.
        list(engine.send_stream("hi"))