"""
Tests for ``VoiceInteraction``: the composition that turns the existing speech
providers into an ``Interaction`` any ``Session``/``AuraEngine`` can drive.

These run fully offline using the dependency-free stubs and small fake audio
source/sink callables -- no microphone, speaker, audio library, or OS code is
ever touched.
"""

import pytest

from conftest import StubBrain

from aura.core.engine import AuraEngine
from aura.interaction import (
    Audio,
    InteractionEnd,
    Session,
    SpeechError,
    SpeechInput,                                                                                                                                                                                                                                                                                                                                                                
    VoiceInteraction,
)
from aura.interaction.base import Interaction
from aura.interaction.voice.stubs import StubSpeechInput, StubSpeechOutput


def _voice(capture, playback, stt=None, tts=None) -> VoiceInteraction:
    return VoiceInteraction(
        speech_input=stt or StubSpeechInput(transcripts=["hello"]),
        speech_output=tts or StubSpeechOutput(),
        capture=capture,
        playback=playback,
    )


def _noop_playback(audio: Audio) -> None:
    return None


def test_voice_interaction_implements_the_contract() -> None:
    interaction = _voice(
        capture=lambda: Audio(data=b"mic"), playback=_noop_playback
    )
    assert isinstance(interaction, Interaction)
    assert interaction.kind == "voice"
    assert isinstance(interaction.speech_input, SpeechInput)
    assert isinstance(interaction.speech_output, StubSpeechOutput)


def test_read_captures_then_transcribes() -> None:
    captured = Audio(data=b"micbytes", format="pcm16")
    stt = StubSpeechInput(transcripts=["hello AURA"])
    interaction = _voice(
        capture=lambda: captured, playback=_noop_playback, stt=stt
    )

    assert interaction.read() == "hello AURA"
    # The exact captured payload was handed to the speech engine.
    assert stt.calls == [captured]


def test_read_captures_once_per_turn() -> None:
    audio = Audio(data=b"x")
    stt = StubSpeechInput(transcripts=["one", "two"])
    captures: list[Audio] = []

    def capture() -> Audio:
        captures.append(audio)
        return audio

    interaction = _voice(capture=capture, playback=_noop_playback, stt=stt)

    assert interaction.read() == "one"
    assert interaction.read() == "two"
    assert len(captures) == 2


def test_read_returns_blank_when_stt_reports_speech_error() -> None:
    class BrokenSTT(StubSpeechInput):
        def transcribe(self, audio: Audio) -> str:
            raise SpeechError("nothing intelligible heard")

    interaction = _voice(
        capture=lambda: Audio(data=b"mic"),
        playback=_noop_playback,
        stt=BrokenSTT(),
    )
    # A failed transcription must skip the turn, not crash the session.
    assert interaction.read() == ""


def test_read_propagates_interaction_end_from_capture() -> None:
    def capture() -> Audio:
        raise InteractionEnd

    interaction = _voice(capture=capture, playback=_noop_playback)
    with pytest.raises(InteractionEnd):
        interaction.read()


def test_write_synthesizes_then_plays_back() -> None:
    tts = StubSpeechOutput()
    played: list[Audio] = []
    interaction = _voice(
        capture=lambda: Audio(data=b"mic"),
        playback=played.append,
        tts=tts,
    )

    interaction.write("hello world")

    assert tts.calls == ["hello world"]
    assert played == [Audio(data=b"hello world", format="wav")]
    assert interaction.speech_output is tts


def test_write_propagates_synthesis_speech_error() -> None:
    class BrokenTTS(StubSpeechOutput):
        def synthesize(self, text: str) -> Audio:
            raise SpeechError("tts engine down")

    interaction = _voice(
        capture=lambda: Audio(data=b"mic"),
        playback=_noop_playback,
        tts=BrokenTTS(),
    )
    with pytest.raises(SpeechError):
        interaction.write("hello")


def test_write_propagates_playback_speech_error() -> None:
    def playback(_audio: Audio) -> None:
        raise SpeechError("speaker unavailable")

    interaction = _voice(capture=lambda: Audio(data=b"mic"), playback=playback)
    with pytest.raises(SpeechError):
        interaction.write("hello")


def test_session_drives_voice_end_to_end() -> None:
    # One utterance, then the input ends: Speech->Engine->Speech->speaker + goodbye.
    brain = StubBrain()
    engine = AuraEngine(brain=brain)
    remaining = [Audio(data=b"hello", format="pcm16")]
    played: list[Audio] = []

    def capture() -> Audio:
        if not remaining:
            raise InteractionEnd
        return remaining.pop(0)

    interaction = _voice(capture=capture, playback=played.append, stt=StubSpeechInput())
    Session(interaction=interaction, engine=engine, output_prefix="").run()

    # The user's audio became text, the engine replied, and that reply was
    # spoken; then the input ended and Session said goodbye (its EOF farewell
    # carries the leading newline Session always writes there).
    assert len(brain.calls) == 1
    assert played and played[0].data == StubBrain.REPLY.encode()
    assert played[-1].data == b"\nGoodbye."


def test_session_stops_on_voice_exit_word() -> None:
    stt = StubSpeechInput(transcripts=["exit"])
    brain = StubBrain()
    played: list[Audio] = []
    interaction = _voice(
        capture=lambda: Audio(data=b"mic"),
        playback=played.append,
        stt=stt,
    )

    Session(interaction=interaction, engine=AuraEngine(brain=brain), output_prefix="").run()

    # The exit word ended the conversation without calling the engine, and the
    # farewell was spoken.
    assert brain.calls == []
    assert played == [Audio(data=b"Goodbye.", format="wav")]


def test_voice_continues_to_next_turn_after_response() -> None:
    # Two utterances are handled one after another: after AURA speaks each
    # reply, the loop returns immediately to listening (the next capture).
    brain = StubBrain()
    engine = AuraEngine(brain=brain)
    stt = StubSpeechInput(transcripts=["what is python", "how are you"])
    remaining = [Audio(data=b"a1"), Audio(data=b"a2")]
    captures: list[int] = []
    played: list[Audio] = []

    def capture() -> Audio:
        captures.append(1)
        if not remaining:
            raise InteractionEnd
        return remaining.pop(0)

    interaction = VoiceInteraction(
        speech_input=stt,
        speech_output=StubSpeechOutput(),
        capture=capture,
        playback=played.append,
    )
    Session(interaction=interaction, engine=engine, output_prefix="").run()

    # Two turns reached the engine (multi-turn works). The engine accumulates
    # history, so the final call carries the complete ordered set of utterances.
    assert len(brain.calls) == 2
    user_msgs = [
        m["content"]
        for m in brain.calls[-1]
        if m.get("role") == "user"
    ]
    assert user_msgs == ["what is python", "how are you"]

    # Listening resumed after each spoken reply: 2 real captures, then a 3rd
    # that ended the loop.
    assert len(captures) == 3
    # Two spoken replies plus the spoken farewell.
    assert [a.data for a in played] == [
        StubBrain.REPLY.encode(),
        StubBrain.REPLY.encode(),
        b"\nGoodbye.",
    ]
