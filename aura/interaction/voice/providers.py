"""
interaction/voice/providers.py
------------------------------
The FIRST real, LOCAL/offline voice providers behind AURA's speech contracts.

  * ``VoskSpeechInput``   -- speech-to-text via Vosk (audio -> text)
  * ``KokoroSpeechOutput`` -- text-to-speech via Kokoro-82M ONNX (text -> audio)

Both are Windows-compatible and fully offline. They are INTENTIONALLY optional:
the heavy engines are imported lazily, so importing AURA never requires them.
When the engine or its model is unavailable, the provider raises ``SpeechError``
instead of crashing -- letting AURA surface a clear startup error.

Neither class touches a microphone or a speaker: ``transcribe``/``synthesize``
only transform bytes in memory. Capturing/playback is left to a future layer.
"""

import io
import json
import os
import struct
import wave
from time import perf_counter
from typing import Any

from aura.core.logging import get_logger
from aura.interaction.voice.base import Audio, SpeechError, SpeechInput, SpeechOutput

logger = get_logger(__name__)

# Vosk's recognizer accepts raw 16kHz mono 16-bit PCM. We feed it in small
# chunks so memory use stays bounded for long utterances.
_PCM_CHUNK_BYTES = 4000

# Process-wide cache of loaded Kokoro pipelines, keyed by (model_path,
# voices_path). The heavy `Kokoro(...)` constructor (ONNX session + espeak/vocab
# tokenizer) takes seconds to run, while `voice`/`lang`/`speed` are cheap
# per-call arguments passed to `create()`. Sharing ONE warm pipeline across every
# `KokoroSpeechOutput` instance guarantees the model is loaded exactly once per
# process and is never reloaded per response -- so repeated `build_speech_output()`
# calls and multiple TTS instances all reuse the same loaded model.
_KOKORO_PIPELINE_CACHE: dict[tuple[str, str], Any] = {}


def _reset_kokoro_pipeline_cache() -> None:
    """Clear the warm-pipeline cache.

    Used by the offline test-suite to isolate each test case so tests that share
    model/voices paths each observe their own load (production never calls this;
    a long-lived server keeps the model warm for its whole lifetime).
    """
    _KOKORO_PIPELINE_CACHE.clear()


def _wav_to_pcm(data: bytes) -> bytes | None:
    """
    Extract the raw PCM sample bytes from a minimal RIFF/WAVE container.

    Returns ``None`` when ``data`` is not a header-shaped WAVE payload. This is
    a deliberately tiny reader (no audio library) that only strips the chunks it
    needs for Vosk.
    """
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None

    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if chunk_id == b"data":
            return data[offset + 8 : offset + 8 + size]
        # RIFF chunks are 2-byte aligned.
        offset += 8 + size + (size % 2)
    return None


def _audio_metrics(pcm: bytes, sample_rate: int) -> tuple[float, float, float]:
    """Cheap signal stats over 16-bit PCM: ``(duration_s, mean_abs, peak)``.

    ``mean_abs`` and ``peak`` are normalised to [0.0, 1.0] (1.0 == full-scale
    int16). Sampling a strided subset keeps this O(n/8) and near-zero cost.
    Used only for diagnostics; the audio bytes are never modified.
    """
    if not pcm or len(pcm) < 2:
        return len(pcm) / (2 * sample_rate), 0.0, 0.0
    stride = 8  # every 8th sample is plenty for level diagnostics
    # Only whole int16 samples; ignore any trailing odd byte.
    whole = len(pcm) - (len(pcm) % 2)
    samples = struct.unpack(f"<{whole // 2}h", pcm[:whole])[::stride]
    if not samples:
        return len(pcm) / (2 * sample_rate), 0.0, 0.0
    total_abs = 0.0
    peak_abs = 0.0
    for value in samples:
        v = value if value >= 0 else -value
        total_abs += v
        if v > peak_abs:
            peak_abs = v
    mean_abs = total_abs / len(samples) / 32768.0
    peak = peak_abs / 32768.0
    return len(pcm) / (2 * sample_rate), mean_abs, peak


class VoskSpeechInput(SpeechInput):
    """Local/offline speech-to-text using Vosk.

    Parameters
    ----------
    model_path:
        Path to a downloaded Vosk model directory (contains an ``am`` /
        ``conf`` / ``graph`` sub-structure).
    sample_rate:
        Expected sample rate of the raw PCM, in Hz. Vosk models are trained at
        16 kHz, which is the default.
    recognizer:
        Optional pre-built Vosk ``KaldiRecognizer`` for dependency injection
        (e.g. a fake in tests). When omitted, one is built from ``model_path``.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        sample_rate: int = 16000,
        recognizer: Any | None = None,
    ) -> None:
        self._model_path = os.fspath(model_path)
        self._sample_rate = sample_rate
        self._recognizer = recognizer
        # The heavy Vosk ``Model`` is loaded once and cached for reuse across
        # utterances (loading it again per turn is wasteful). Recognizers are
        # inexpensive and are created fresh each time from this model.
        self._model: Any | None = None
        # Diagnostic: confirm WHICH model is actually being used (accuracy of
        # proper nouns is strongly model-dependent; the small en-us model is
        # known to miss rare words/names like "Altaf").
        logger.info(
            "VoskSpeechInput using model %s at %d Hz",
            self._model_path,
            self._sample_rate,
        )

    def transcribe(self, audio: Audio) -> str:
        """
        Transcribe ``audio`` into plain text.

        Accepts raw ``pcm16`` payloads or ``wav`` containers (the header is
        stripped automatically). Any engine or payload problem surfaces as a
        ``SpeechError``.
        """
        pcm = self._to_pcm(audio)
        if not pcm:
            raise SpeechError("Speech input received no usable audio payload")

        # Signal diagnostics at the STT boundary (duration + asynchronous RMS /
        # peak level). Cheap (strided sample) and helps distinguish a quiet/clipped
        # mic feed (bad amplitude) from a model-vocabulary limitation (fine
        # amplitude, misspelled name). No audio is modified.
        duration_s, mean_abs, peak = _audio_metrics(pcm, self._sample_rate)
        logger.info(
            "Vosk input audio: %.2f s, mean|amp|=%.4f, peak=%.3f",
            duration_s,
            mean_abs,
            peak,
        )

        recognizer = self._recognizer or self._build_recognizer()
        started = perf_counter()
        try:
            for start in range(0, len(pcm), _PCM_CHUNK_BYTES):
                recognizer.AcceptWaveform(pcm[start : start + _PCM_CHUNK_BYTES])
            final = json.loads(recognizer.FinalResult())
        except Exception as exc:  # any engine/parse failure -> typed SpeechError
            raise SpeechError(f"Speech-to-text failed: {exc}") from exc
        text = str(final.get("text", "")).strip()
        # Diagnostic: the EXACT Vosk final transcript (pre-normalization).
        logger.info(
            "Vosk final transcript (%d ms): %r",
            (perf_counter() - started) * 1000,
            text,
        )
        return text

    def _to_pcm(self, audio: Audio) -> bytes:
        """Normalize the payload to raw PCM bytes, rejecting unsupported formats."""
        data = bytes(audio.data)
        fmt = audio.format.lower()
        if fmt in {"pcm16", "raw", "pcm"}:
            return data
        if fmt == "wav":
            pcm = _wav_to_pcm(data)
            if pcm is None:
                raise SpeechError("Speech input received an invalid WAV payload")
            return pcm
        raise SpeechError(f"Unsupported audio format for speech input: {audio.format!r}")

    def _build_recognizer(self) -> Any:
        try:
            from vosk import KaldiRecognizer, Model  # type: ignore
        except ImportError as exc:
            raise SpeechError(
                "Vosk is not installed: run 'pip install -r requirements-voice.txt' "
                "to enable local speech-to-text"
            ) from exc
        try:
            # Load the model exactly once per instance, then create a fresh
            # recognizer against the SAME cached model for this utterance.
            if self._model is None:
                self._model = Model(self._model_path)
        except Exception as exc:
            raise SpeechError(f"Vosk model could not be loaded: {exc}") from exc
        return KaldiRecognizer(self._model, self._sample_rate)


def _samples_to_pcm16(samples: Any) -> bytes:
    """Convert Kokoro samples to signed 16-bit PCM bytes."""
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - kokoro deps are optional
        raise SpeechError(
            "numpy is required to convert Kokoro samples to WAV; install "
            "'pip install -r requirements-voice.txt'"
        ) from exc

    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise SpeechError("Kokoro synthesis produced no audio samples")
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def _samples_to_wav(samples: Any, sample_rate: int) -> bytes:
    """Encode Kokoro samples as a browser-playable mono WAV payload."""
    if sample_rate <= 0:
        raise SpeechError(f"Kokoro returned an invalid sample rate: {sample_rate!r}")

    pcm = _samples_to_pcm16(samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()


class KokoroSpeechOutput(SpeechOutput):
    """Local/offline text-to-speech using Kokoro-82M via ONNX Runtime."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        voices_path: str | os.PathLike[str] | None = None,
        voice: str = "af_sarah",
        lang: str = "en-us",
        speed: float = 1.0,
        pipeline: Any | None = None,
    ) -> None:
        self._model_path = os.fspath(model_path) if model_path is not None else None
        self._voices_path = os.fspath(voices_path) if voices_path is not None else None
        self._voice = str(voice).strip()
        self._lang = str(lang).strip()
        self._speed = float(speed)
        self._pipeline = pipeline
        logger.info(
            "KokoroSpeechOutput configured model=%s voices=%s voice=%s lang=%s speed=%.2f",
            self._model_path,
            self._voices_path,
            self._voice,
            self._lang,
            self._speed,
        )

    def synthesize(self, text: str) -> Audio:
        """Synthesize ``text`` into a WAV ``Audio`` payload."""
        text = str(text).strip() if text is not None else ""
        if not text:
            raise SpeechError("Cannot synthesize empty text to speech")

        pipeline = self._get_pipeline()
        started = perf_counter()
        try:
            samples, sample_rate = pipeline.create(
                text,
                voice=self._voice,
                speed=self._speed,
                lang=self._lang,
            )
        except Exception as exc:
            raise SpeechError(f"Text-to-speech failed: {exc}") from exc

        logger.info(
            "Kokoro synthesized %d chars in %.0f ms",
            len(text),
            (perf_counter() - started) * 1000,
        )
        return Audio(data=_samples_to_wav(samples, sample_rate), format="wav")

    def warm_up(self) -> None:
        """Eagerly initialize and cache the Kokoro pipeline."""
        self._get_pipeline()

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> Any:
        if self._model_path is None or self._voices_path is None:
            raise SpeechError("Kokoro model and voices paths are required to build the voice")

        # Reuse the single warm, process-wide pipeline for this model so the
        # expensive ONNX session / espeak / vocab are never loaded more than once.
        key = (os.fspath(self._model_path), os.fspath(self._voices_path))
        cached = _KOKORO_PIPELINE_CACHE.get(key)
        if cached is not None:
            logger.info(
                "Reusing warm Kokoro pipeline for %s (load skipped, %d cached)",
                self._model_path,
                len(_KOKORO_PIPELINE_CACHE),
            )
            return cached

        try:
            from kokoro_onnx import Kokoro  # type: ignore
        except ImportError as exc:
            raise SpeechError(
                "kokoro-onnx is not installed: run 'pip install -r requirements-voice.txt' "
                "to enable local text-to-speech"
            ) from exc
        logger.info(
            "Loading Kokoro model from %s with voices %s ...",
            self._model_path,
            self._voices_path,
        )
        try:
            pipeline = Kokoro(self._model_path, self._voices_path)
        except Exception as exc:
            raise SpeechError(f"Kokoro model could not be loaded: {exc}") from exc
        # Only cache on success; a failed load must be retried next time.
        _KOKORO_PIPELINE_CACHE[key] = pipeline
        logger.info(
            "Kokoro model loaded and cached (id=%d cached=%d)",
            id(pipeline),
            len(_KOKORO_PIPELINE_CACHE),
        )
        return pipeline