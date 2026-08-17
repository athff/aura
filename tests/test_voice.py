"""
Tests for the voice I/O abstraction: the two provider-independent speech
contracts (``SpeechInput`` audio->text, ``SpeechOutput`` text->audio), the
lightweight ``Audio`` value object, and the deterministic offline stubs.

These run fully offline -- no microphone, speakers, STT/TTS engines, audio
libraries, or OS-specific code. They only exercise the contracts so future
voice engines can plug in without redesign.
"""

import dataclasses

import pytest

from aura.interaction import (
    Audio,
    SpeechError,
    SpeechInput,
    SpeechOutput,
    StubSpeechInput,
    StubSpeechOutput,
)


def test_speech_input_and_output_are_distinct_abstractions() -> None:
    # Two separate interfaces, not a single blob: each is its own ABC.
    assert SpeechInput is not SpeechOutput
    assert issubclass(SpeechInput, SpeechInput)
    assert issubclass(SpeechOutput, SpeechOutput)
    with pytest.raises(TypeError):
        SpeechInput()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        SpeechOutput()  # type: ignore[abstract]


def test_audio_is_a_frozen_value_object_with_default_format() -> None:
    audio = Audio(data=b"\x00\x01")
    assert audio.data == b"\x00\x01"
    assert audio.format == "wav"  # default hint
    assert Audio(data=b"\x00", format="mp3").format == "mp3"
    # Immutable: it must never be mutated after creation.
    with pytest.raises(dataclasses.FrozenInstanceError):
        audio.format = "ogg"  # type: ignore[misc]


def test_stub_speech_input_implements_the_contract() -> None:
    assert isinstance(StubSpeechInput(), SpeechInput)


def test_stub_speech_input_transcribes_scripted_transcripts() -> None:
    stub = StubSpeechInput(transcripts=["hello AURA", "second"])
    assert stub.transcribe(Audio(data=b"\x00", format="wav")) == "hello AURA"
    assert stub.transcribe(Audio(data=b"\x00", format="wav")) == "second"


def test_stub_speech_input_echoes_payload_when_no_script() -> None:
    stub = StubSpeechInput()
    # Deterministic fallback: the payload bytes are treated as the transcript.
    assert stub.transcribe(Audio(data=b"what time is it?")) == "what time is it?"
    assert stub.transcribe(Audio(data=b"  padded  ")) == "padded"


def test_stub_speech_input_records_audio_calls() -> None:
    stub = StubSpeechInput()
    first = Audio(data=b"a")
    second = Audio(data=b"b", format="pcm16")
    stub.transcribe(first)
    stub.transcribe(second)
    assert stub.calls == [first, second]


def test_stub_speech_output_implements_the_contract() -> None:
    assert isinstance(StubSpeechOutput(), SpeechOutput)


def test_stub_speech_output_synthesizes_deterministic_audio() -> None:
    stub = StubSpeechOutput()
    result = stub.synthesize("hello")
    assert isinstance(result, Audio)
    # Offline determinism: payload is the UTF-8 bytes of the text.
    assert result.data == "hello".encode("utf-8")
    assert result.format == "wav"


def test_stub_speech_output_uses_configured_format() -> None:
    stub = StubSpeechOutput(format="mp3")
    assert stub.synthesize("x").format == "mp3"


def test_stub_speech_output_records_text_calls() -> None:
    stub = StubSpeechOutput()
    stub.synthesize("one")
    stub.synthesize("two")
    assert stub.calls == ["one", "two"]


def test_stub_speech_round_trips_text_through_both_contracts() -> None:
    # Speaking then listening with the stubs preserves the text -- proving the
    # audio<->text contracts line up without any real engine.
    tts = StubSpeechOutput()
    stt = StubSpeechInput()
    spoken = tts.synthesize("hello world")
    assert stt.transcribe(spoken) == "hello world"


def test_voice_symbols_exported_from_the_interaction_package() -> None:
    import aura.interaction as interaction

    for name in (
        "Audio",
        "SpeechError",
        "SpeechInput",
        "SpeechOutput",
        "StubSpeechInput",
        "StubSpeechOutput",
    ):
        assert name in interaction.__all__