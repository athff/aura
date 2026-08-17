"""
Offline tests for the voice runner entry point (``python -m aura.voice``).

These run FULLY OFFLINE. Vosk/Kokoro models are simulated with temporary files,
the speech providers are swapped for the dependency-free stubs, and the
microphone/speaker are represented by tiny fake callables -- so no real model,
microphone, speaker, or audio library is ever touched.
"""

import os
import sys
import types

import pytest

from conftest import StubBrain

from aura import voice as voice_pkg
from aura.core.engine import AuraEngine
from aura.interaction import (
    Audio,
    KokoroSpeechOutput,
    SoundDeviceHardware,
    StubSpeechInput,
    StubSpeechOutput,
    VoiceInteraction,
    VoskSpeechInput,
)
from aura.voice import __main__ as runner


def _fspath(p) -> str:
    return os.fspath(p)


# ---------------------------------------------------------------------------
# build_hardware() — config
# ---------------------------------------------------------------------------


def test_build_hardware_uses_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_VOICE_SAMPLE_RATE", "48000")
    monkeypatch.setenv("AURA_VOICE_CHANNELS", "2")
    monkeypatch.setenv("AURA_VOICE_RECORDING_LIMIT", "3.5")

    hw = runner.build_hardware()

    assert isinstance(hw, SoundDeviceHardware)
    assert hw.sample_rate == 48000
    assert hw.channels == 2
    assert hw.recording_limit == 3.5


def test_build_hardware_defaults_match_vosk(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AURA_VOICE_SAMPLE_RATE",
        "AURA_VOICE_CHANNELS",
        "AURA_VOICE_RECORDING_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)

    hw = runner.build_hardware()

    assert hw.sample_rate == 16000
    assert hw.channels == 1
    assert hw.recording_limit == 5.0


def test_build_hardware_ignores_invalid_numeric_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_VOICE_RECORDING_LIMIT", "not-a-number")
    assert runner.build_hardware().recording_limit == 5.0


def test_build_hardware_end_of_speech_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_VOICE_SILENCE_DURATION", "1.2")
    monkeypatch.setenv("AURA_VOICE_SILENCE_THRESHOLD", "0.01")

    hw = runner.build_hardware()

    assert hw._silence_duration == 1.2
    assert hw._silence_threshold == 0.01


def test_build_hardware_end_of_speech_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("AURA_VOICE_SILENCE_DURATION", "AURA_VOICE_SILENCE_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)

    hw = runner.build_hardware()

    assert hw._silence_duration == 0.6
    assert hw._silence_threshold == 0.005


def test_build_hardware_barge_in_defaults_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("AURA_VOICE_BARGE_IN", "AURA_VOICE_BARGE_SUSTAIN"):
        monkeypatch.delenv(name, raising=False)

    hw = runner.build_hardware()

    assert hw.barge_in is False
    assert hw._barge_sustain == 0.2


def test_build_hardware_barge_in_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_VOICE_BARGE_IN", "1")
    monkeypatch.setenv("AURA_VOICE_BARGE_SUSTAIN", "0.5")

    hw = runner.build_hardware()

    assert hw.barge_in is True
    assert hw._barge_sustain == 0.5


# ---------------------------------------------------------------------------
# build_speech_input() / build_speech_output() — model path handling
# ---------------------------------------------------------------------------


def test_build_speech_input_raises_clear_error_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_VOSK_MODEL", "C:/no/such/vosk-model")
    with pytest.raises(FileNotFoundError) as exc:
        runner.build_speech_input()
    msg = str(exc.value)
    assert "Vosk model not found" in msg
    assert "AURA_VOSK_MODEL" in msg
    assert "alphacephei.com/vosk" in msg


def test_build_speech_output_raises_clear_error_when_voice_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_KOKORO_MODEL", "C:/no/such/model.onnx")
    monkeypatch.setenv("AURA_KOKORO_VOICES", "C:/no/such/voices.bin")
    with pytest.raises(FileNotFoundError) as exc:
        runner.build_speech_output()
    msg = str(exc.value)
    assert "Kokoro model not found" in msg
    assert "AURA_KOKORO_MODEL" in msg
    assert "kokoro-onnx" in msg


def test_build_speech_input_uses_existing_model_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model_dir = tmp_path / "vosk-model-small-en-us"
    model_dir.mkdir()
    monkeypatch.setenv("AURA_VOSK_MODEL", str(model_dir))

    stt = runner.build_speech_input()

    assert isinstance(stt, VoskSpeechInput)
    assert stt._model_path == _fspath(model_dir)


def test_build_speech_output_uses_existing_voice_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"fake-onnx")
    voices.write_bytes(b"fake-voices")
    monkeypatch.setenv("AURA_KOKORO_MODEL", str(model))
    monkeypatch.setenv("AURA_KOKORO_VOICES", str(voices))
    monkeypatch.setenv("AURA_KOKORO_VOICE", "af_sarah")
    monkeypatch.setenv("AURA_KOKORO_LANG", "en-us")
    monkeypatch.setenv("AURA_KOKORO_SPEED", "1.0")

    load_log: list[tuple[str, str]] = []

    class FakeKokoro:
        def __init__(self, model_path: str, voices_path: str) -> None:
            load_log.append((model_path, voices_path))

        def create(self, text: str, voice: str, speed: float, lang: str):
            assert text == ""
            assert voice == "af_sarah"
            assert speed == 1.0
            assert lang == "en-us"
            return [0.0, 0.0], 24000

    mod = types.ModuleType("kokoro_onnx")
    mod.Kokoro = FakeKokoro  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mod)

    tts = runner.build_speech_output()

    assert isinstance(tts, KokoroSpeechOutput)
    assert tts._model_path == _fspath(model)
    assert tts._voices_path == _fspath(voices)
    assert load_log == [(_fspath(model), _fspath(voices))]



# ---------------------------------------------------------------------------
# build_voice_interaction() — pure wiring (stubs + fake hardware)
# ---------------------------------------------------------------------------


def test_build_voice_interaction_wires_providers_and_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner, "build_speech_input", lambda: StubSpeechInput(["hello AURA"])
    )
    monkeypatch.setattr(runner, "build_speech_output", lambda: StubSpeechOutput())
    played: list[Audio] = []

    class FakeHardware:
        def capture(self) -> Audio:
            return Audio(data=b"mic-bytes", format="pcm16")

        def playback(self, audio: Audio) -> None:
            played.append(audio)

    interaction = runner.build_voice_interaction(FakeHardware())

    assert isinstance(interaction, VoiceInteraction)
    assert interaction.read() == "hello AURA"
    interaction.write("bye")
    assert played == [Audio(data=b"bye", format="wav")]


def test_build_voice_interaction_builds_default_hardware_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "build_speech_input", lambda: StubSpeechInput())
    monkeypatch.setattr(runner, "build_speech_output", lambda: StubSpeechOutput())
    # build_hardware must be called by default; verify via a spy.
    calls: list[bool] = []
    real = runner.build_hardware

    def spy():
        calls.append(True)
        return real()

    monkeypatch.setattr(runner, "build_hardware", spy)
    interaction = runner.build_voice_interaction()
    assert isinstance(interaction, VoiceInteraction)
    assert calls == [True]


# ---------------------------------------------------------------------------
# build_engine()
# ---------------------------------------------------------------------------


def test_build_engine_uses_configured_brain(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = StubBrain()
    monkeypatch.setattr(runner, "create_brain", lambda: brain)

    engine = runner.build_engine()

    assert isinstance(engine, AuraEngine)
    assert engine.send("hi") == StubBrain.REPLY


# ---------------------------------------------------------------------------
# main() — the runnable entry point (never touches hardware)
# ---------------------------------------------------------------------------


def test_main_runs_voice_session_and_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict = {}

    interaction = VoiceInteraction(
        speech_input=StubSpeechInput(),
        speech_output=StubSpeechOutput(),
        capture=lambda: Audio(data=b"mic"),
        playback=lambda audio: None,
    )
    monkeypatch.setattr(runner, "build_engine", lambda: AuraEngine(brain=StubBrain()))
    monkeypatch.setattr(runner, "build_hardware", lambda: object())
    monkeypatch.setattr(runner, "build_voice_interaction", lambda hw: interaction)

    class FakeSession:
        def __init__(self, interaction, engine, output_prefix):
            recorded["interaction"] = interaction
            recorded["engine"] = engine
            recorded["output_prefix"] = output_prefix

        def run(self) -> None:
            recorded["ran"] = True

    monkeypatch.setattr(runner, "Session", FakeSession)

    runner.main()

    assert recorded["interaction"] is interaction
    assert isinstance(recorded["engine"], AuraEngine)
    assert recorded["output_prefix"] == ""  # voice speaks replies, no text prefix
    assert recorded["ran"] is True


def test_voice_package_has_docstring() -> None:
    # The -m entry resolves to aura.voice.__main__, which runs off this package.
    assert voice_pkg.__doc__
