"""
interaction/session.py
----------------------
Reusable orchestrator that drives an ``Interaction`` against any AURA
\"responder\" -- normally ``AuraEngine``.

Session is the piece that makes future modalities cheap: it depends ONLY on
the ``Interaction`` contract and a tiny ``Responder`` protocol (anything with
``send(user_message) -> reply``). A voice/streaming/whatever backend simply
implements ``Interaction`` and it can be driven here with zero redesign.

Session applies conversation policy that is orthogonal to the transport:
  * blank input -> skip the turn (don't call the responder)
  * the user types an exit word -> stop
  * input is exhausted (``InteractionEnd``) -> say goodbye and stop
All presentation (``You: `` prompt, ``AURA: `` reply prefix) is delegated to
the interaction's writer so the layer stays provider- and format-agnostic.
"""

from typing import Protocol

from aura.interaction.base import Interaction, InteractionEnd


class Responder(Protocol):
    """
    Minimal contract for the thing that turns a user message into a reply.
    ``AuraEngine.send`` satisfies this structurally, so Session never needs to
    import or depend on AuraEngine directly.
    """

    def send(self, user_message: str) -> str: ...


class Session:
    """Drives an ``Interaction`` with a ``Responder`` until the chat ends."""

    # Words (case-insensitive) that end the session when typed by the user.
    EXIT_WORDS = frozenset({"exit", "quit"})

    def __init__(
        self,
        interaction: Interaction,
        engine: Responder,
        output_prefix: str = "AURA: ",
        farewell: str = "Goodbye.",
    ) -> None:
        self._interaction = interaction
        self._engine = engine
        self._output_prefix = output_prefix
        self._farewell = farewell

    @property
    def interaction(self) -> Interaction:
        """The input/output modality this session is driving."""
        return self._interaction

    @property
    def engine(self) -> Responder:
        """The responder this session sends user messages to."""
        return self._engine

    def run(self) -> None:
        """
        Run the conversation until the user exits, quits, or input ends.

        Each non-empty utterance is sent to the responder and its reply is
        written back through the interaction.
        """
        while True:
            try:
                text = self._interaction.read()
            except InteractionEnd:
                # Input closed (EOF / Ctrl-C): farewell and stop. The leading
                # newline keeps the farewell on its own line after the prompt.
                self._interaction.write(f"\n{self._output_prefix}{self._farewell}")
                return

            text = text.strip()
            if text.lower() in self.EXIT_WORDS:
                self._interaction.write(f"{self._output_prefix}{self._farewell}")
                return
            if not text:
                # Empty utterance: no command to process, just ask again.
                continue

            self._interaction.write(f"{self._output_prefix}{self._engine.send(text)}")