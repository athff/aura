"""
Tests for the real voice providers: ``VoskSpeechInput`` (speech-to-text) and
``KokoroSpeechOutput`` (text-to-speech).

These run FULLY offline. The heavy engines (``vosk`` / ``kokoro-onnx``) are
never imported or invoked -- tests inject fake recognizer/pipeline objects
(dependency injection), and simulate missing/broken engines to confirm the
providers raise ``SpeechError`` instead of crashing.
"""

import io
import json
import struct
import sys
import types
import wave

import pytest

from aura.interaction import (
    Audio,
    KokoroSpeechOutput,
    SpeechError,
    SpeechInput,
    SpeechOutput,
    VoskSpeechInput,
)

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def make_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Build a minimal, valid RIFF/WAVE file containing ``pcm`` bytes."""
    fmt_chunk = struct.pack(
        "<4sIHHIIHH", b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16
    )
    data_chunk = struct.pack("<4sI", b"data", len(pcm)) + pcm
    riff_size = 4 + (8 + len(fmt_chunk)) + (8 + len(data_chunk))
    return b"RIFF" + struct.pack("<I", riff_size) + b"WAVE" + fmt_chunk + data_chunk


def decode_wav(data: bytes) -> tuple[bytes, int, int]:
    """Decode a WAV container into ``(pcm, sample_rate, channels)``."""
    with wave.open(io.BytesIO(data), "rb") as w:
        return (
            w.readframes(w.getnframes()),
            w.getframerate(),
            w.getnchannels(),
        )


class FakeRecognizer:
    """A fake Vosk ``KaldiRecognizer`` that records chunks and returns text."""

    def __init__(self, text: str = "hello world") -> None:
        self._text = text
        self.accepted: list[bytes] = []
        self.final_called = False

    def AcceptWaveform(self, chunk: bytes) -> None:
        self.accepted.append(bytes(chunk))

    def FinalResult(self) -> str:
        self.final_called = True
        return json.dumps({"text": self._text})


class FakeKokoroPipeline:
    """A fake Kokoro pipeline that records synth calls and returns samples."""

    def __init__(self, samples: list[float] | None = None, sample_rate: int = 24000) -> None:
        self.samples = samples if samples is not None else [0.0, 0.25, -0.25]
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, str, float, str]] = []

    def create(self, text: str, voice: str, speed: float, lang: str):
        self.calls.append((text, voice, speed, lang))
        return self.samples, self.sample_rate


# ---------------------------------------------------------------------------
# VoskSpeechInput
# ---------------------------------------------------------------------------


def test_vosk_input_implements_speech_contract() -> None:
    assert isinstance(VoskSpeechInput("model"), SpeechInput)


def test_vosk_input_transcribes_pcm16_payload() -> None:
    recognizer = FakeRecognizer(text="what is the weather?")
    provider = VoskSpeechInput(model_path="unused-model", recognizer=recognizer)
    text = provider.transcribe(Audio(data=b"\x00\x01\x02", format="pcm16"))

    assert text == "what is the weather?"
    assert recognizer.final_called
    assert recognizer.accepted == [b"\x00\x01\x02"]


def test_vosk_input_strips_wav_header_before_recognition() -> None:
    recognizer = FakeRecognizer(text="ok")
    provider = VoskSpeechInput(model_path="unused-model", recognizer=recognizer)
    pcm = (b"\x00\x01" * 16) + b"\xaa\xbb"
    provider.transcribe(Audio(data=make_wav(pcm), format="wav"))

    # The recognizer must receive only the PCM samples, not the RIFF header.
    assert recognizer.accepted == [pcm]


def test_vosk_input_feeds_large_pcm_in_chunks() -> None:
    recognizer = FakeRecognizer(text="long")
    provider = VoskSpeechInput(model_path="unused-model", recognizer=recognizer)
    pcm = b"\x00\x01" * 4096  # longer than the chunk size => split feed
    provider.transcribe(Audio(data=pcm, format="pcm16"))

    assert len(recognizer.accepted) > 1
    assert b"".join(recognizer.accepted) == pcm


def test_vosk_input_rejects_empty_payload() -> None:
    provider = VoskSpeechInput("unused-model", recognizer=FakeRecognizer())
    with pytest.raises(SpeechError):
        provider.transcribe(Audio(data=b"", format="pcm16"))


def test_vosk_input_rejects_invalid_wav_payload() -> None:
    provider = VoskSpeechInput("unused-model", recognizer=FakeRecognizer())
    with pytest.raises(SpeechError):
        provider.transcribe(Audio(data=b"NOT A WAV FILE", format="wav"))


def test_vosk_input_rejects_unsupported_format() -> None:
    provider = VoskSpeechInput("unused-model", recognizer=FakeRecognizer())
    with pytest.raises(SpeechError):
        provider.transcribe(Audio(data=b"\x00\x01", format="flac"))


def test_vosk_input_raises_speech_error_when_vosk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate `vosk` being unimportable: no injected recognizer means the
    # provider must try to import it and fall back to SpeechError.
    monkeypatch.setitem(sys.modules, "vosk", None)
    provider = VoskSpeechInput("some-model")
    with pytest.raises(SpeechError):
        provider.transcribe(Audio(data=b"\x00\x01", format="pcm16"))


def test_vosk_input_raises_speech_error_on_model_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = types.ModuleType("vosk")

    class BadModel:
        def __init__(self, _path: str) -> None:
            raise RuntimeError("model dir missing")

    mod.Model = BadModel
    mod.KaldiRecognizer = lambda model, rate: None  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "vosk", mod)

    provider = VoskSpeechInput("no/such/model")
    with pytest.raises(SpeechError):
        provider.transcribe(Audio(data=b"\x00\x01", format="pcm16"))


def test_vosk_model_loaded_exactly_once_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {"model": 0, "recognizer": 0}
    loaded_models: list[object] = []

    class FakeModel:
        def __init__(self, _path: str) -> None:
            counts["model"] += 1
            loaded_models.append(self)

    class FakeKaldiRecognizer:
        def __init__(self, model, sample_rate: int) -> None:
            counts["recognizer"] += 1
            self._model = model
            self._sample_rate = sample_rate

        def AcceptWaveform(self, _chunk: bytes) -> None:
            return None

        def FinalResult(self) -> str:
            return json.dumps({"text": "hello once"})

    mod = types.ModuleType("vosk")
    mod.Model = FakeModel  # type: ignore[attr-defined]
    mod.KaldiRecognizer = FakeKaldiRecognizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vosk", mod)

    provider = VoskSpeechInput("some/model/path")
    pcm = b"\x00\x01" * 8
    first = provider.transcribe(Audio(data=pcm, format="pcm16"))
    second = provider.transcribe(Audio(data=pcm, format="pcm16"))

    # Transcription is unchanged across multiple utterances...
    assert first == "hello once"
    assert second == "hello once"
    # ...but the heavy Vosk model is loaded exactly once and reused...
    assert counts["model"] == 1
    assert len(loaded_models) == 1
    # ...while a fresh recognizer is created per utterance against that model.
    assert counts["recognizer"] == 2



def test_vosk_input_raises_speech_error_on_broken_recognizer() -> None:
    class BrokenRecognizer:
        def AcceptWaveform(self, _chunk: bytes) -> None:
            raise RuntimeError("engine crashed")

        def FinalResult(self) -> str:
            return "{}"  # pragma: no cover

    provider = VoskSpeechInput("unused-model", recognizer=BrokenRecognizer())
    with pytest.raises(SpeechError):
        provider.transcribe(Audio(data=b"\x00\x01", format="pcm16"))
# ---------------------------------------------------------------------------
# KokoroSpeechOutput
# ---------------------------------------------------------------------------


def test_kokoro_output_implements_speech_contract() -> None:
    assert isinstance(KokoroSpeechOutput(), SpeechOutput)


def test_kokoro_output_synthesizes_wav_audio() -> None:
    pipeline = FakeKokoroPipeline()
    provider = KokoroSpeechOutput(
        model_path="kokoro-v1.0.onnx",
        voices_path="voices-v1.0.bin",
        pipeline=pipeline,
    )

    result = provider.synthesize("hello AURA")

    assert isinstance(result, Audio)
    assert result.format == "wav"
    assert result.data  # non-empty WAV payload is produced
    assert pipeline.calls == [("hello AURA", "af_sarah", 1.0, "en-us")]
    # The resulting in-memory WAV round-trips the frames Kokoro returned.
    pcm, rate, channels = decode_wav(result.data)
    assert rate == 24000
    assert channels == 1
    assert pcm


def test_kokoro_output_coerces_none_to_empty_then_rejects() -> None:
    provider = KokoroSpeechOutput(
        model_path="kokoro-v1.0.onnx",
        voices_path="voices-v1.0.bin",
        pipeline=FakeKokoroPipeline(),
    )
    with pytest.raises(SpeechError):
        provider.synthesize(None)  # type: ignore[arg-type]


def test_kokoro_output_rejects_empty_text() -> None:
    provider = KokoroSpeechOutput(
        model_path="kokoro-v1.0.onnx",
        voices_path="voices-v1.0.bin",
        pipeline=FakeKokoroPipeline(),
    )
    for blank in ("", "   "):
        with pytest.raises(SpeechError):
            provider.synthesize(blank)


def test_kokoro_output_raises_when_no_model_or_voice_pack() -> None:
    provider = KokoroSpeechOutput()  # neither model path nor injected pipeline
    with pytest.raises(SpeechError):
        provider.synthesize("hello")


def test_kokoro_output_raises_when_kokoro_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kokoro_onnx", None)
    provider = KokoroSpeechOutput(model_path="voice.onnx", voices_path="voices.bin")
    with pytest.raises(SpeechError):
        provider.synthesize("hello")


def test_kokoro_output_raises_on_model_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = types.ModuleType("kokoro_onnx")

    class BadKokoro:
        def __init__(self, _model_path: str, _voices_path: str) -> None:
            raise RuntimeError("no onnx model")

    mod.Kokoro = BadKokoro  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", mod)

    provider = KokoroSpeechOutput(model_path="voice.onnx", voices_path="voices.bin")
    with pytest.raises(SpeechError):
        provider.synthesize("hello")


def test_kokoro_output_raises_on_engine_failure() -> None:
    class BrokenPipeline:
        def create(self, _text: str, _voice: str, _speed: float, _lang: str):
            raise RuntimeError("synthesis crashed")

    provider = KokoroSpeechOutput(
        model_path="voice.onnx",
        voices_path="voices.bin",
        pipeline=BrokenPipeline(),
    )
    with pytest.raises(SpeechError):
        provider.synthesize("hello")

def _fake_kokoro_module(load_log: list[tuple[str, str]]) -> types.ModuleType:
    """Build a fake ``kokoro_onnx`` module that records constructor calls."""

    class CountingKokoro:
        def __init__(self, model_path: str, voices_path: str) -> None:
            load_log.append((model_path, voices_path))
            self.calls: list[tuple[str, str, float, str]] = []

        def create(self, text: str, voice: str, speed: float, lang: str):
            self.calls.append((text, voice, speed, lang))
            return [0.0, 0.25, -0.25], 24000

    mod = types.ModuleType("kokoro_onnx")
    mod.Kokoro = CountingKokoro  # type: ignore[attr-defined]
    return mod


def test_kokoro_pipeline_loaded_only_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    load_log: list[tuple[str, str]] = []
    monkeypatch.setitem(sys.modules, "kokoro_onnx", _fake_kokoro_module(load_log))

    provider = KokoroSpeechOutput(model_path="voice.onnx", voices_path="voices.bin")
    first = provider.synthesize("hello AURA")
    second = provider.synthesize("second utterance")

    assert len(load_log) == 1
    assert load_log[0] == ("voice.onnx", "voices.bin")
    assert first.data and second.data
    assert first.format == "wav" and second.format == "wav"


def test_kokoro_warm_up_preloads_pipeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    load_log: list[tuple[str, str]] = []
    monkeypatch.setitem(sys.modules, "kokoro_onnx", _fake_kokoro_module(load_log))

    provider = KokoroSpeechOutput(model_path="voice.onnx", voices_path="voices.bin")
    provider.warm_up()
    provider.warm_up()
    provider.synthesize("hello")

    assert len(load_log) == 1


def test_kokoro_uses_injected_pipeline_without_loading_another() -> None:
    pipeline = FakeKokoroPipeline()
    provider = KokoroSpeechOutput(
        model_path="voice.onnx",
        voices_path="voices.bin",
        pipeline=pipeline,
    )

    provider.warm_up()
    result = provider.synthesize("hello")

    assert provider._pipeline is pipeline
    assert pipeline.calls == [("hello", "af_sarah", 1.0, "en-us")]
    assert result.format == "wav"
    assert result.data


def test_audio_metrics_reports_duration_and_levels() -> None:
    from aura.interaction.voice.providers import _audio_metrics

    # 1 second of 16-bit mono PCM @ 16 kHz = 32 000 bytes. Use a signal period
    # that does not divide the 8-sample metric stride so sampled readings are
    # heterogeneous (mean < peak).
    samples = struct.pack("<5h", 4000, 4000, 4000, 0, 0) * 3200
    duration_s, mean_abs, peak = _audio_metrics(samples, 16000)
    assert duration_s == 1.0
    assert 0.0 < mean_abs < peak <= 1.0
    assert peak > 0.05  # well above silence

    # Pure silence -> zero levels.
    assert _audio_metrics(b"\x00\x00" * 16000, 16000)[1:] == (0.0, 0.0)
    assert _audio_metrics(b"", 16000) == (0.0, 0.0, 0.0)
