"""
interaction/voice/base.py
-------------------------
Provider-independent CONTRACTS for speech I/O.

Two separate abstractions, so future engines can be swapped independently:

  * ``SpeechInput``  -- audio -> text   (speech-to-text / ASR, e.g. Whisper)
  * ``SpeechOutput`` -- text -> audio   (text-to-speech / TTS, e.g. Kokoro)

``Audio`` is intentionally a tiny, dependency-free value object (raw bytes plus
a format hint). No microphone, speakers, streaming, audio libraries, or OS
drivers are assumed here -- concrete providers decide how audio is captured or
played. These contracts only describe the *transformation* that a speech engine
performs, keeping AURA independent of any specific engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class SpeechError(Exception):
    """
    Base for EXPECTED speech I/O failures raised by concrete providers (a
    transcription/synthesis failure, an unusable payload, etc.). Keeps provider
    internals out of user-facing errors.
    """


@dataclass(frozen=True)
class Audio:
    """
    An immutable audio payload: raw ``data`` bytes plus a ``format`` hint.

    ``format`` names the container/codec so a provider knows how to interpret
    the bytes (e.g. ``\"wav\"``, ``\"pcm16\"``, ``\"mp3\"``). It carries no
    opinion about where the audio came from or how it will be played.
    """

    data: bytes
    format: str = "wav"


class SpeechInput(ABC):
    """
    Speech *input* contract: turns captured audio into text (audio -> text).

    Concrete providers (e.g. a Whisper-based STT) implement ``transcribe`` to
    interpret an ``Audio`` payload and return the resulting plain-text
    transcript. Callers raise ``SpeechError`` to signal expected failures.
    """

    @abstractmethod
    def transcribe(self, audio: Audio) -> str:
        """
        Transcribe ``audio`` into plain text.

        Parameters
        ----------
        audio:
            The captured utterance, with a ``format`` hint describing its bytes.

        Returns
        -------
        str :
            The spoken text. May be empty if nothing intelligible was heard.
        """
        raise NotImplementedError


class SpeechOutput(ABC):
    """
    Speech *output* contract: turns text into spoken audio (text -> audio).

    Concrete providers (e.g. Kokoro) implement ``synthesize`` to
    render ``text`` as an ``Audio`` payload ready for later playback.
    """

    @abstractmethod
    def synthesize(self, text: str) -> Audio:
        """
        Synthesize spoken audio for ``text``.

        Parameters
        ----------
        text:
            The text AURA wants to speak aloud.

        Returns
        -------
        Audio :
            The synthesized speech, with a ``format`` hint describing its bytes.
        """
        raise NotImplementedError