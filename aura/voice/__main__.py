"""
aura/voice/__main__.py
---------------------
The voice entry point. Run with::

    python -m aura.voice

Environment variables (all optional, the module works with defaults). These can
be set in the shell OR in the same ``.env`` used by AURA's settings:

    AURA_VOSK_MODEL           Path to the Vosk model directory
                              (default: ``models/vosk-model-small-en-us-0.15``)

    AURA_KOKORO_MODEL         Path to the Kokoro ONNX model file
                              (default: ``models/kokoro-v1.0.onnx``)
    AURA_KOKORO_VOICES        Path to the Kokoro voices binary
                              (default: ``models/voices-v1.0.bin``)
    AURA_KOKORO_VOICE         Kokoro voice name (default: ``af_sarah``)
    AURA_KOKORO_LANG          Kokoro language code (default: ``en-us``)
    AURA_KOKORO_SPEED         Kokoro speaking speed (default: ``1.0``)

    AURA_VOICE_SAMPLE_RATE    Microphone sample rate  (default: 16000)

    AURA_VOICE_CHANNELS       Microphone channels       (default: 1)

    AURA_VOICE_RECORDING_LIMIT  Hard capture timeout    (default: 5.0)

    AURA_VOICE_SILENCE_DURATION    Trailing-silence end-of-speech cutoff  (default: 0.6)
    AURA_VOICE_SILENCE_THRESHOLD   RMS below which a chunk is "silence"     (default: 0.005)

    AURA_VOICE_BARGE_IN        Listen while speaking; user speech interrupts TTS (default: 0)
    AURA_VOICE_BARGE_SUSTAIN   Seconds of voiced audio required to barge     (default: 0.2)

    AURA_PLAYBACK_SAMPLE_RATE  Force playback sample rate (0=auto-detect device
                               rate and resample TTS audio to it)  (default: 0)

    AURA_VOSK_MODEL            Path to the Vosk model directory              (default: models/vosk-model-small-en-us-0.15)
                               For better accuracy with names/rare words use a larger
                               model (e.g. vosk-model-en-us-0.22) and set this var.

The LLM brain is configured through the same ``.env`` / ``AURA_LLM_PROVIDER``
system that ``aura.main`` uses — no separate voice-brain configuration.
"""

import os
from pathlib import Path

try:  # python-dotenv ships with pydantic-settings; guard in case it's absent
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is an expected dependency
    load_dotenv = None  # type: ignore

from aura.brain.factory import create_brain
from aura.core.engine import AuraEngine
from aura.core.logging import get_logger
from aura.interaction import (
    KokoroSpeechOutput,
    Session,
    SoundDeviceHardware,
    VoiceInteraction,
    VoskSpeechInput,
)
from aura.config.settings import settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default model paths (relative to the project root, i.e. two levels up from
# aura/voice/__main__.py)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent.parent  # aura/ -> project root

# Make the same `.env` used by AURA's settings available to the runner, so
# AURA_VOSK_MODEL / AURA_KOKORO_* set in `.env` work here too. Existing
# process env vars take precedence (load_dotenv never overrides them).
if load_dotenv is not None:
    load_dotenv(_PROJECT / ".env")

_DEFAULT_VOSK = _PROJECT / "models" / "vosk-model-small-en-us-0.15"
_DEFAULT_KOKORO_MODEL = _PROJECT / "models" / "kokoro-v1.0.onnx"
_DEFAULT_KOKORO_VOICES = _PROJECT / "models" / "voices-v1.0.bin"


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _get_env_str(name: str, default: str) -> str:
    """Read an optional env var, falling back to *default*."""
    return os.environ.get(name, default).strip()


def _get_env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is not None:
        try:
            return float(val.strip())
        except ValueError:
            pass
    return default


def _get_env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is not None:
        try:
            return int(val.strip())
        except ValueError:
            pass
    return default


def _get_env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    low = val.strip().lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    return default


# ---------------------------------------------------------------------------
# Builders (individually testable)
# ---------------------------------------------------------------------------


def build_hardware() -> SoundDeviceHardware:
    """Create the microphone/speaker adapter with optional env overrides."""
    # silence_duration / silence_threshold tune END-OF-SPEECH detection (the
    # period of trailing quiet after speech begins that ends a capture). They
    # can be tuned per-machine without touching code.
    return SoundDeviceHardware(
        sample_rate=_get_env_int("AURA_VOICE_SAMPLE_RATE", 16000),
        channels=_get_env_int("AURA_VOICE_CHANNELS", 1),
        recording_limit=_get_env_float("AURA_VOICE_RECORDING_LIMIT", 5.0),
        silence_duration=_get_env_float("AURA_VOICE_SILENCE_DURATION", 0.6),
        silence_threshold=_get_env_float("AURA_VOICE_SILENCE_THRESHOLD", 0.005),
        barge_in=_get_env_bool("AURA_VOICE_BARGE_IN", False),
        barge_sustain=_get_env_float("AURA_VOICE_BARGE_SUSTAIN", 0.2),
        # Explicit playback sample rate (0 / unset = auto-detect the device's
        # native rate and resample TTS audio to it).
        output_sample_rate=_get_env_int("AURA_PLAYBACK_SAMPLE_RATE", 0) or None,
    )


def build_speech_input() -> VoskSpeechInput:
    """Create the Vosk STT engine, raising a clear error if the model is missing."""
    path = Path(_get_env_str("AURA_VOSK_MODEL", str(_DEFAULT_VOSK)))
    if not path.is_dir():
        raise FileNotFoundError(
            f"Vosk model not found at {path}. "
            f"Download a model from https://alphacephei.com/vosk/models "
            f"and point AURA_VOSK_MODEL at it, or place it at:\n"
            f"  {_DEFAULT_VOSK}\n"
            f"Example: vosk-model-small-en-us-0.15 (≈40 MB)"
        )
    sample_rate = _get_env_int("AURA_VOICE_SAMPLE_RATE", 16000)
    return VoskSpeechInput(model_path=os.fspath(path), sample_rate=sample_rate)


def build_speech_output() -> KokoroSpeechOutput:
    """Create the Kokoro TTS engine, raising a clear error if files are missing."""
    model_path = Path(_get_env_str("AURA_KOKORO_MODEL", settings.kokoro_model or str(_DEFAULT_KOKORO_MODEL)))
    voices_path = Path(_get_env_str("AURA_KOKORO_VOICES", settings.kokoro_voices or str(_DEFAULT_KOKORO_VOICES)))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Kokoro model not found at {model_path}. "
            f"Download kokoro-v1.0.onnx from https://github.com/thewh1teagle/kokoro-onnx/releases "
            f"and point AURA_KOKORO_MODEL at it, or place it at:\n"
            f"  {_DEFAULT_KOKORO_MODEL}\n"
            f"Example: kokoro-v1.0.onnx"
        )
    if not voices_path.is_file():
        raise FileNotFoundError(
            f"Kokoro voices file not found at {voices_path}. "
            f"Download voices-v1.0.bin from https://github.com/thewh1teagle/kokoro-onnx/releases "
            f"and point AURA_KOKORO_VOICES at it, or place it at:\n"
            f"  {_DEFAULT_KOKORO_VOICES}\n"
            f"Example: voices-v1.0.bin"
        )
    tts = KokoroSpeechOutput(
        model_path=os.fspath(model_path),
        voices_path=os.fspath(voices_path),
        voice=_get_env_str("AURA_KOKORO_VOICE", settings.kokoro_voice),
        lang=_get_env_str("AURA_KOKORO_LANG", settings.kokoro_lang),
        speed=_get_env_float("AURA_KOKORO_SPEED", settings.kokoro_speed),
    )
    tts.warm_up()
    return tts


def build_engine() -> AuraEngine:
    """Create AURA's LLM engine from the configured brain provider."""
    brain = create_brain()
    return AuraEngine(brain=brain)


def build_voice_interaction(
    hardware: SoundDeviceHardware | None = None,
) -> VoiceInteraction:
    """Wire the speech providers and hardware into a ``VoiceInteraction``."""
    if hardware is None:
        hardware = build_hardware()
    stt = build_speech_input()
    tts = build_speech_output()
    return VoiceInteraction(
        speech_input=stt,
        speech_output=tts,
        capture=hardware.capture,
        playback=hardware.playback,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run a single bounded voice conversation then exit."""
    print("AURA voice is starting...")
    print("Speak now. Say 'exit' or 'quit' to stop.")
    print()

    engine = build_engine()
    hardware = build_hardware()
    interaction = build_voice_interaction(hardware)

    # output_prefix="" prevents Session from printing a text prefix before the
    # spoken reply; the voice already speaks AURA's reply directly.
    session = Session(interaction=interaction, engine=engine, output_prefix="")
    session.run()


if __name__ == "__main__":
    main()