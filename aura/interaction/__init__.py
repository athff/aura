"""AURA interaction layer: provider-independent text & voice interaction foundation.

Exposes the abstract ``Interaction`` contract (a modality through which a human
reads and writes with AURA), the ``InteractionEnd`` signal, the concrete
``TextInteraction`` implementation, the reusable ``Session`` driver, and the
provider-independent speech contracts (``SpeechInput`` / ``SpeechOutput``) with offline stubs for future voice
modalities.

This layer is deliberately independent of FastAPI, audio libraries, and the
brain/memory engine -- text, streaming, and voice backends can be added later by
implementing the relevant contract without redesign.
"""

from aura.interaction.base import Interaction, InteractionEnd
from aura.interaction.session import Session
from aura.interaction.text import TextInteraction
from aura.interaction.voice import (
    Audio,
    KokoroSpeechOutput,
    SoundDeviceHardware,
    SpeechError,
    SpeechInput,
    SpeechOutput,
    StubSpeechInput,
    StubSpeechOutput,
    VoiceInteraction,
    VoskSpeechInput,
)

__all__ = [
    "Interaction",
    "InteractionEnd",
    "Session",
    "TextInteraction",
    "Audio",
    "SpeechError",
    "SpeechInput",
    "SpeechOutput",
    "VoiceInteraction",
    "VoskSpeechInput",
    "KokoroSpeechOutput",
    "StubSpeechInput",
    "StubSpeechOutput",
    "SoundDeviceHardware",
]