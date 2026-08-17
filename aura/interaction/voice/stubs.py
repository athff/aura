"""
interaction/voice/stubs.py
--------------------------
Deterministic, dependency-free stand-ins for the speech contracts -- for
offline testing and for wiring AURA before real STT/TTS engines exist.

  * ``StubSpeechInput``  -- returns a scripted transcript, or (if none was
                           configured) echoes the audio payload bytes back as
                           text so tests can drive speech-to-text offline.
  * ``StubSpeechOutput`` -- returns an ``Audio`` whose payload is the UTF-8
                           bytes of the input text, so text <-> audio can
                           round-trip without any audio engine.

Both record every call they receive, making them easy to assert against.
"""

from collections.abc import Sequence

from aura.interaction.voice.base import Audio, SpeechInput, SpeechOutput


class StubSpeechInput(SpeechInput):
    """A fake speech-to-text engine: deterministic and fully offline.

    Parameters
    ----------
    transcripts:
        Optional script of transcriptions to return in order. When exhausted
        (or omitted), the payload bytes are treated as the transcription text.
    """

    def __init__(self, transcripts: Sequence[str] | None = None) -> None:
        self._transcripts = list(transcripts) if transcripts is not None else []
        self.calls: list[Audio] = []

    def transcribe(self, audio: Audio) -> str:
        self.calls.append(audio)
        if self._transcripts:
            return self._transcripts.pop(0)
        # Deterministic fallback: treat the payload as the transcription text.
        return audio.data.decode("utf-8", errors="replace").strip()


class StubSpeechOutput(SpeechOutput):
    """A fake text-to-speech engine: deterministic and fully offline.

    Parameters
    ----------
    format:
        The ``Audio.format`` hint to attach to every synthesized payload.
    """

    def __init__(self, format: str = "wav") -> None:
        self._format = format
        self.calls: list[str] = []

    def synthesize(self, text: str) -> Audio:
        self.calls.append(text)
        return Audio(data=text.encode("utf-8"), format=self._format)