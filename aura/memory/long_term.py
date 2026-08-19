"""
memory/long_term.py
-------------------
Contract for AURA's long-term semantic memory.

Long-term memory is intentionally separate from conversation memory.
ConversationMemory handles the recent conversation window sent directly
to the Brain; LongTermMemory stores and retrieves older information
semantically through a vector backend.
"""

from abc import ABC, abstractmethod


class LongTermMemory(ABC):
    """Provider-independent contract for semantic long-term memory."""

    @abstractmethod
    def remember(self, text: str, metadata: dict | None = None) -> None:
        """Store one piece of information in long-term memory."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search long-term memory semantically.

        Returns a list of dictionaries containing the stored text and
        implementation-defined metadata/score information.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all long-term memories."""
        raise NotImplementedError
