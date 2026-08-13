"""
core/engine.py
--------------
The AURA Core: the orchestrator.

Its job is to manage a conversation -- remember what has been said, hand it to
the Brain, and return the reply. It is deliberately small right now, but this
is the central place where we will later plug in long-term memory, tools,
and routing between different brains.
"""

from aura.brain.base import Brain
from aura.core.errors import BrainError
from aura.core.logging import get_logger

logger = get_logger(__name__)


class AuraEngine:
    """Coordinates a conversation between the user and a Brain."""

    def __init__(self, brain: Brain) -> None:
        # We RECEIVE a brain from the outside instead of creating one here.
        # This is "dependency injection": the engine doesn't care which brain
        # it uses, which makes it easy to test and ready for your hybrid plan.
        self._brain = brain
        self._history: list[dict] = []
        logger.debug("AuraEngine ready with brain %s", type(brain).__name__)

    def send(self, user_message: str) -> str:
        """
        Process one user message and return AURA's reply.

        Raises
        ------
        BrainError
            If the brain returns nothing usable (``None`` or an empty reply).
            Provider/transport failures and unexpected brain exceptions are
            logged here and re-raised unchanged; the web layer is responsible
            for translating them into user-facing HTTP responses.
        """
        self._history.append({"role": "user", "content": user_message})

        try:
            reply = self._brain.think(self._history)
        except Exception:
            # Log every brain failure through AURA's central logger, then let
            # the error propagate unchanged so the web layer (which already has
            # a generic 502 safety net) can translate it for the user.
            logger.exception("Brain.think failed while processing a user message")
            raise

        # Robustness: never accept a brain that returns nothing usable.
        if reply is None or not str(reply).strip():
            logger.warning("Brain returned an empty reply; rejecting it as BrainError")
            raise BrainError("The brain returned an empty reply.")

        # Normalize: if a provider hands back non-text, coerce it safely.
        reply = str(reply).strip()
        self._history.append({"role": "assistant", "content": reply})
        return reply
