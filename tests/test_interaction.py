"""
Tests for the interaction foundation: the abstract ``Interaction`` contract,
the concrete ``TextInteraction``, and the reusable ``Session`` driver.

These run fully offline -- no API keys, no network, no audio, no FastAPI. They
deliberately leave out the whole multimodal world (voice, streaming, TTS, etc.)
because that is the point of this layer: to be provider- and modality-agnostic.
"""

import pytest

from conftest import StubBrain

from aura.core.engine import AuraEngine
from aura.interaction.base import Interaction, InteractionEnd
from aura.interaction.session import Session
from aura.interaction.text import TextInteraction


def test_interaction_is_abstract_and_not_instantiable() -> None:
    with pytest.raises(TypeError):
        Interaction()  # type: ignore[abstract]


def test_text_interaction_implements_the_contract() -> None:
    assert isinstance(TextInteraction(), Interaction)
    assert TextInteraction().kind == "text"


def test_read_returns_stripped_input() -> None:
    interaction = TextInteraction(read_line=lambda prompt: "  hello  ")
    assert interaction.read() == "hello"


def test_read_uses_the_configured_prompt() -> None:
    prompts: list[str] = []
    interaction = TextInteraction(
        prompt="You: ", read_line=lambda prompt: prompts.append(prompt) or "hi"
    )
    interaction.read()
    assert prompts == ["You: "]


def test_read_raises_interaction_end_on_eof() -> None:
    def eof(_prompt: str) -> str:
        raise EOFError

    interaction = TextInteraction(read_line=eof)
    with pytest.raises(InteractionEnd):
        interaction.read()


def test_read_raises_interaction_end_on_interrupt() -> None:
    def interrupt(_prompt: str) -> str:
        raise KeyboardInterrupt

    interaction = TextInteraction(read_line=interrupt)
    with pytest.raises(InteractionEnd):
        interaction.read()


def test_write_emits_text_through_the_writer() -> None:
    written: list[str] = []
    interaction = TextInteraction(write_line=written.append)
    interaction.write("AURA: hello world")
    assert written == ["AURA: hello world"]


class ScriptedInteraction(Interaction):
    """A test double that plays back a script and ends when it runs out."""

    kind = "fake"

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)
        self.written: list[str] = []

    def read(self) -> str:
        if not self._lines:
            raise InteractionEnd
        return self._lines.pop(0)

    def write(self, text: str) -> None:
        self.written.append(text)


class RecordingResponder:
    """A minimal ``Responder`` (anything with ``send``) that records input."""

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def send(self, user_message: str) -> str:
        self.inputs.append(user_message)
        return f"reply-to-{user_message}"


def test_session_runs_turns_and_replies_until_exit() -> None:
    responder = RecordingResponder()
    session = Session(
        interaction=ScriptedInteraction(["hello", "world", "exit"]),
        engine=responder,
    )
    session.run()

    assert responder.inputs == ["hello", "world"]
    assert session.engine is responder
    assert session.interaction.kind == "fake"
    assert session.interaction.written == [
        "AURA: reply-to-hello",
        "AURA: reply-to-world",
        "AURA: Goodbye.",
    ]


def test_session_skips_blank_input_without_calling_the_responder() -> None:
    responder = RecordingResponder()
    Session(
        interaction=ScriptedInteraction(["   ", "hello", "quit"]),
        engine=responder,
    ).run()
    # The blank line must not trigger a turn.
    assert responder.inputs == ["hello"]


def test_session_farewells_when_input_ends() -> None:
    responder = RecordingResponder()
    interaction = ScriptedInteraction(["hi"])
    Session(interaction=interaction, engine=responder).run()

    assert interaction.written == ["AURA: reply-to-hi", "\nAURA: Goodbye."]


def test_session_exit_words_are_case_insensitive() -> None:
    responder = RecordingResponder()
    interaction = ScriptedInteraction(["EXIT"])
    Session(interaction=interaction, engine=responder).run()
    assert responder.inputs == []
    assert interaction.written == ["AURA: Goodbye."]

def test_session_recognizes_quit_case_insensitively() -> None:
    responder = RecordingResponder()
    for word in ("quit", "Quit", "QUIT"):
        interaction = ScriptedInteraction([word])
        Session(interaction=interaction, engine=responder).run()
        assert responder.inputs == []  # never reached the responder
        assert interaction.written == ["AURA: Goodbye."]


def test_session_recognizes_exit_with_surrounding_whitespace() -> None:
    responder = RecordingResponder()
    interaction = ScriptedInteraction(["   exit   "])
    Session(interaction=interaction, engine=responder).run()
    assert responder.inputs == []
    assert interaction.written == ["AURA: Goodbye."]


def test_session_does_not_terminate_when_exit_appears_inside_a_sentence() -> None:
    responder = RecordingResponder()
    question = "Why is the word exit used in programs?"
    interaction = ScriptedInteraction([question])
    Session(interaction=interaction, engine=responder).run()
    # The word "exit" is part of a real question, so it must be answered, not
    # treated as a termination command.
    assert responder.inputs == [question]
    assert interaction.written[0] == f"AURA: reply-to-{question}"


def test_session_does_not_terminate_when_quit_appears_inside_a_sentence() -> None:
    responder = RecordingResponder()
    question = "Why does quitting a program matter?"
    Session(interaction=ScriptedInteraction([question]), engine=responder).run()
    assert responder.inputs == [question]



def test_session_drives_the_real_engine_via_stub_brain() -> None:
    # Proves the interaction layer plugs into AuraEngine unchanged (DI).
    brain = StubBrain()
    engine = AuraEngine(brain=brain)
    interaction = ScriptedInteraction(["hello", "quit"])

    Session(interaction=interaction, engine=engine).run()

    # The engine produced a real reply for the user turn.
    assert interaction.written == [
        f"AURA: {StubBrain.REPLY}",
        "AURA: Goodbye.",
    ]
    assert len(brain.calls) == 1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    