from conftest import StubBrain

from aura.core.engine import AuraEngine
from aura.memory.long_term import LongTermMemory


class RecordingLongTermMemory(LongTermMemory):
    """Offline test double for AURA long-term semantic memory."""

    def __init__(self, results: list[dict] | None = None) -> None:
        self.results = results or []
        self.remembered: list[tuple[str, dict | None]] = []
        self.searches: list[tuple[str, int]] = []
        self.cleared = False

    def remember(self, text: str, metadata: dict | None = None) -> None:
        self.remembered.append((text, metadata))

    def search(self, query: str, limit: int = 5) -> list[dict]:
        self.searches.append((query, limit))
        return list(self.results[:limit])

    def clear(self) -> None:
        self.cleared = True
        self.remembered.clear()


def test_engine_works_without_long_term_memory() -> None:
    brain = StubBrain()
    engine = AuraEngine(brain=brain)

    assert engine.send("hello") == StubBrain.REPLY


def test_engine_searches_long_term_memory_before_brain() -> None:
    brain = StubBrain()
    long_term = RecordingLongTermMemory(
        results=[
            {
                "text": "The user's favorite language is Python.",
                "metadata": {},
                "score": 0.91,
            }
        ]
    )

    engine = AuraEngine(brain=brain, long_term_memory=long_term)
    engine.send("What programming language do I like?")

    assert long_term.searches == [
        ("What programming language do I like?", 5)
    ]

    messages = brain.calls[-1]
    assert any(
        "The user's favorite language is Python." in m["content"]
        for m in messages
    )


def test_engine_preserves_recent_conversation_with_long_term_memory() -> None:
    brain = StubBrain()
    long_term = RecordingLongTermMemory(
        results=[
            {
                "text": "AURA is the user's personal AI assistant.",
                "metadata": {},
                "score": 0.88,
            }
        ]
    )

    engine = AuraEngine(brain=brain, long_term_memory=long_term)

    engine.send("first message")
    engine.send("second message")

    messages = brain.calls[-1]

    # Recent conversation remains present.
    assert any(
        m["role"] == "user" and m["content"] == "second message"
        for m in messages
    )

    # Retrieved long-term memory is also present.
    assert any(
        "AURA is the user's personal AI assistant." in m["content"]
        for m in messages
    )


def test_engine_remembers_successful_turn_long_term() -> None:
    brain = StubBrain()
    long_term = RecordingLongTermMemory()

    engine = AuraEngine(brain=brain, long_term_memory=long_term)
    engine.send("I am building AURA")

    assert long_term.remembered == [
        ("I am building AURA", {"role": "user"}),
        (StubBrain.REPLY, {"role": "assistant"}),
    ]


def test_engine_does_not_store_failed_reply_long_term() -> None:
    class EmptyBrain(StubBrain):
        def think(self, messages: list[dict]) -> str:
            self.calls.append([dict(m) for m in messages])
            return "   "

    brain = EmptyBrain()
    long_term = RecordingLongTermMemory()

    engine = AuraEngine(brain=brain, long_term_memory=long_term)

    try:
        engine.send("hello")
    except Exception:
        pass

    assert long_term.remembered == []


def test_long_term_memory_is_optional() -> None:
    brain = StubBrain()
    engine = AuraEngine(brain=brain, long_term_memory=None)

    assert engine.send("hello") == StubBrain.REPLY
