"""
memory/base.py
--------------
The CONTRACT that every AURA conversation memory must follow.

A memory stores conversation messages in the established ``{role, content}``
format (see ``Brain.think``) and hands them back to callers. Like the brain,
AURA depends only on this abstract interface -- never on a concrete storage
backend -- so we can swap in durable storage, RAG, or vector search later
WITHOUT changing the engine or the web layer.
"""

from abc import ABC, abstractmethod


class Memory(ABC):
    """Abstract base class: the blueprint that all memory backends implement."""

    @abstractmethod
    def add(self, role: str, content: str) -> None:
        """
        Append one message turn to the end of the conversation.

        Parameters
        ----------
        role:
            Who sent the message, e.g. ``"user"`` or ``"assistant"``.
        content:
            The plain-text message body.
        """
        raise NotImplementedError

    @abstractmethod
    def messages(self) -> list[dict]:
        """
        Return all conversation messages as a list of ``{"role", "content"}``
        dicts, in the order they were added. Implementations must return a copy
        so callers cannot mutate the memory's internal state.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove the entire conversation history."""
        raise NotImplementedError