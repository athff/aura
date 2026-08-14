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


# ---------------------------------------------------------------------------
# Context-limited memory: max message count, oldest-first eviction, ordering.
# ---------------------------------------------------------------------------


def test_max_messages_caps_history_length() -> None:
    memory = ConversationMemory(max_messages=3)
    for content in ["a", "b", "c", "d", "e"]:
        memory.add("user", content)
    msgs = memory.messages()
    assert len(msgs) == 3
    assert [m["content"] for m in msgs] == ["c", "d", "e"]


def test_oldest_messages_evicted_first() -> None:
    memory = ConversationMemory(max_messages=2)
    memory.add("user", "first")
    memory.add("assistant", "second")
    memory.add("user", "third")
    assert [m["content"] for m in memory.messages()] == ["second", "third"]


def test_eviction_preserves_ordering_and_roles() -> None:
    memory = ConversationMemory(max_messages=4)
    roles = ["user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]
    for i, role in enumerate(roles):
        memory.add(role, f"m{i}")

    msgs = memory.messages()
    assert len(msgs) == 4
    assert [m["content"] for m in msgs] == ["m6", "m7", "m8", "m9"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


def test_unlimited_when_max_messages_is_none() -> None:
    memory = ConversationMemory()  # default is unlimited
    assert memory.max_messages is None
    for i in range(100):
        memory.add("user", str(i))
    assert len(memory.messages()) == 100
    assert memory.messages()[-1] == {"role": "user", "content": "99"}


def test_clear_resets_bounded_memory_and_lifts_eviction() -> None:
    memory = ConversationMemory(max_messages=2)
    memory.add("user", "one")
    memory.add("assistant", "two")
    memory.add("user", "three")
    assert len(memory.messages()) == 2

    memory.clear()
    assert memory.messages() == []

    # Still bounded after a clear, and newly added messages obey the cap.
    memory.add("user", "fresh")
    memory.add("assistant", "again")
    memory.add("user", "third turn")
    assert [m["content"] for m in memory.messages()] == ["again", "third turn"]


def test_evicted_memory_still_returns_defensive_copies() -> None:
    memory = ConversationMemory(max_messages=2)
    memory.add("user", "one")
    memory.add("assistant", "two")
    memory.add("user", "three")

    returned = memory.messages()
    returned.append({"role": "assistant", "content": "hacked"})
    returned[0]["content"] = "tampered"

    assert memory.messages() == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


@pytest.mark.parametrize("bad", [0, -1])
def test_max_messages_rejects_non_positive_values(bad: int) -> None:
    with pytest.raises(ValueError):
        ConversationMemory(max_messages=bad)


@pytest.mark.parametrize("bad", [True, "10", 2.5])
def test_max_messages_rejects_wrong_types(bad: object) -> None:
    with pytest.raises(TypeError):
        ConversationMemory(max_messages=bad)  # type: ignore[arg-type]


def test_max_messages_exposed_on_contract() -> None:
    # Every implementation exposes the configured cap via the Memory contract.
    assert ConversationMemory(max_messages=5).max_messages == 5
    assert ConversationMemory().max_messages is None


def test_engine_honors_bounded_injected_memory() -> None:
    brain = StubBrain()
    memory = ConversationMemory(max_messages=2)
    engine = AuraEngine(brain=brain, memory=memory)

    engine.send("one")
    engine.send("two")
    assert [m["content"] for m in memory.messages()] == ["two", StubBrain.REPLY]

    engine.send("three")
    # Oldest pair evicted; only the newest two turns survive, in order.
    msgs = memory.messages()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert [m["content"] for m in msgs] == ["three", StubBrain.REPLY]

    # The brain no longer receives evicted turns in its last call.
    seen_contents = [m["content"] for m in brain.calls[-1]]
    assert "one" not in seen_contents
