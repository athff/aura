"""AURA voice I/O abstraction: provider-independent speech contracts.

Exposes the two separate speech abstractions -- ``SpeechInput`` (audio -> text /
ASR, e.g. Whisper) and ``SpeechOutput`` (text -> audio / TTS, e.g. Kokoro) --
along with the lightweight ``Audio`` value object, the ``SpeechError`` base,
and deterministic offline stubs. No microphone, speakers, streaming, audio
drivers, or OS-specific code lives here.
"""

from aura.interaction.voice.base import Audio, SpeechError, SpeechInput, SpeechOutput
from aura.interaction.voice.hardware import SoundDeviceHardware
from aura.interaction.voice.interaction import (
    AudioCapture,
    AudioPlayback,
    VoiceInteraction,
)
from aura.interaction.voice.providers import KokoroSpeechOutput, VoskSpeechInput
from aura.interaction.voice.stubs import StubSpeechInput, StubSpeechOutput

__all__ = [
    "Audio",
    "SpeechError",
    "SpeechInput",
    "SpeechOutput",
    "VoiceInteraction",
    "AudioCapture",
    "AudioPlayback",
    "SoundDeviceHardware",
    "VoskSpeechInput",
    "KokoroSpeechOutput",
    "StubSpeechInput",
    "StubSpeechOutput",
]