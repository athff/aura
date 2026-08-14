"""
Tests for the memory layer: the abstract ``Memory`` contract and the concrete
in-memory ``ConversationMemory``, plus the engine's use of injected memory.

These run fully offline -- no API keys, no network, no database.
"""

import pytest

from conftest import StubBrain

from aura.core.engine import AuraEngine
from aura.memory.base import Memory
from aura.memory.conversation import ConversationMemory


def test_conversation_memory_implements_memory_contract() -> None:
    assert issubclass(ConversationMemory, Memory)
    # Concrete memory must be instantiable (unlike the abstract base).
    assert isinstance(ConversationMemory(), Memory)
    with pytest.raises(TypeError):
        Memory()  # type: ignore[abstract]


def test_add_retrieve_returns_messages_in_order() -> None:
    memory = ConversationMemory()
    assert memory.messages() == []

    memory.add("user", "hello")
    memory.add("assistant", "hi there")
    memory.add("user", "again")

    assert memory.messages() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "again"},
    ]


def test_add_appends_to_end_regardless_of_role() -> None:
    memory = ConversationMemory()
    memory.add("assistant", "a")
    memory.add("user", "b")
    memory.add("assistant", "c")
    assert [m["content"] for m in memory.messages()] == ["a", "b", "c"]


def test_clear_empties_history() -> None:
    memory = ConversationMemory()
    memory.add("user", "hello")
    memory.add("assistant", "hi")
    assert len(memory.messages()) == 2

    memory.clear()
    assert memory.messages() == []

    # Still usable after a clear.
    memory.add("user", "fresh start")
    assert memory.messages() == [{"role": "user", "content": "fresh start"}]


def test_returned_history_cannot_mutate_internal_state() -> None:
    memory = ConversationMemory()
    memory.add("user", "hello")

    # Mutating the returned list and its dicts must not touch stored state.
    returned = memory.messages()
    returned.append({"role": "assistant", "content": "hacked"})
    returned[0]["content"] = "tampered"

    assert memory.messages() == [{"role": "user", "content": "hello"}]


def test_messages_returns_fresh_copies_each_call() -> None:
    memory = ConversationMemory()
    memory.add("user", "hello")

    first = memory.messages()
    second = memory.messages()
    # Distinct list objects AND distinct message dicts each call.
    assert first is not second
    assert first[0] is not second[0]


class RecordingMemory(Memory):
    """A memory double that records the messages it returns and what was added."""

    def __init__(self) -> None:
        self.messages_returned: list[list[dict]] = []
        self.adds: list[tuple[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.adds.append((role, content))

    def messages(self) -> list[dict]:
        # Reflect every add the engine performed into the list we hand back.
        current = [{"role": r, "content": c} for r, c in self.adds]
        self.messages_returned.append(list(current))
        return current

    def clear(self) -> None:
        self.adds.clear()


def test_engine_uses_injected_memory() -> None:
    brain = StubBrain()
    memory = RecordingMemory()
    engine = AuraEngine(brain=brain, memory=memory)

    engine.send("hello")

    # The engine stored the user turn and returned an assistant turn via the
    # injected memory (not an internal list), in the right order.
    assert memory.adds == [
        ("user", "hello"),
        ("assistant", StubBrain.REPLY),
    ]
    # The brain received the exact conversation the memory handed back.
    assert memory.messages_returned and memory.messages_returned[-1] == brain.calls[-1]


def test_engine_creates_default_conversation_memory_when_omitted() -> None:
    engine = AuraEngine(brain=StubBrain())
    assert isinstance(engine._memory, ConversationMemory)

    engine.send("first")
    engine.send("second")
    # Default memory accumulates history across turns like the original `_history`.
    assert len(engine._memory.messages()) == 4  # user, assistant, user, assistant
    assert engine._memory.messages()[-1] == {
        "role": "assistant",
        "content": StubBrain.REPLY,
    }