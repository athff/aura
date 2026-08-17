"""
interaction/voice/interaction.py
--------------------------------
A ``VoiceInteraction``: composes the existing speech providers with the existing
``Interaction`` contract to make a voice modality that ``Session``/``AuraEngine``
can drive WITHOUT any change to them.

It implements the two tiny verbs every interaction needs:

    read()   =  capture audio -> SpeechInput.transcribe()  -> text
    write()  =  SpeechOutput.synthesize(text) -> playback audio

Microphone and speaker access are deliberately kept behind two small callables
(``capture`` / ``playback``), so real hardware implementations can be added
later without touching this class.

No wake-word detection, continuous listening, or streaming is done here: each
``read`` captures exactly one utterance and ``write`` speaks exactly one reply.
"""

from collections.abc import Callable

from aura.interaction.base import Interaction, InteractionEnd
from aura.interaction.voice.base import Audio, SpeechError, SpeechInput, SpeechOutput

# The audio source: blocks until one utterance is captured and returns it, or
# raises ``InteractionEnd`` to signal the conversation is over.
AudioCapture = Callable[[], Audio]

# The audio sink: receives an ``Audio`` payload to play back to the human.
AudioPlayback = Callable[[Audio], None]


class VoiceInteraction(Interaction):
    """A voice modality built from injected speech input/output and audio I/O.

    Parameters
    ----------
    speech_input:
        Speech-to-text engine (``SpeechInput.transcribe``). Swappable, e.g.
        ``VoskSpeechInput``, ``StubSpeechInput``.
    speech_output:
        Text-to-speech engine (``SpeechOutput.synthesize``). Swappable, e.g.
        ``KokoroSpeechOutput``, ``StubSpeechOutput``.
    capture:
        Callable that returns one captured ``Audio`` utterance (microphone).
        May raise ``InteractionEnd`` to end the conversation.
    playback:
        Callable that plays an ``Audio`` payload back (speaker).
    """

    kind = "voice"

    def __init__(
        self,
        speech_input: SpeechInput,
        speech_output: SpeechOutput,
        capture: AudioCapture,
        playback: AudioPlayback,
    ) -> None:
        self._speech_input = speech_input
        self._speech_output = speech_output
        self._capture = capture
        self._playback = playback

    @property
    def speech_input(self) -> SpeechInput:
        """The speech-to-text engine driving ``read``."""
        return self._speech_input

    @property
    def speech_output(self) -> SpeechOutput:
        """The text-to-speech engine driving ``write``."""
        return self._speech_output

    def read(self) -> str:
        """Capture one utterance and return its transcription.

        If the speech engine reports nothing intelligible (``SpeechError``), a
        blank utterance is returned so the session simply skips the turn instead
        of crashing. ``capture`` raising ``InteractionEnd`` propagates so the
        conversation can be ended by the input side.
        """
        audio = self._capture()
        try:
            return self._speech_input.transcribe(audio)
        except SpeechError:
            return ""

    def write(self, text: str) -> None:
        """Synthesize ``text`` and play it back.

        Speech failures here (synthesis or playback) surface as ``SpeechError``
        because the reply was genuinely not delivered.
        """
        audio = self._speech_output.synthesize(text)
        self._playback(audio)