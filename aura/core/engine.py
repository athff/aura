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


class AuraEngine:
    """Coordinates a conversation between the user and a Brain."""

    def __init__(self, brain: Brain) -> None:
        # We RECEIVE a brain from the outside instead of creating one here.
        # This is "dependency injection": the engine doesn't care which brain
        # it uses, which makes it easy to test and ready for your hybrid plan.
        self._brain = brain
        self._history: list[dict] = []

    def send(self, user_message: str) -> str:
        """Process one user message and return AURA's reply."""
        self._history.append({"role": "user", "content": user_message})
        reply = self._brain.think(self._history)
        self._history.append({"role": "assistant", "content": reply})
        return reply
