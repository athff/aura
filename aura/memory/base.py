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
    """Abstract base class: the blueprint that all memory backends implement.

    Parameters
    ----------
    max_messages:
        Optional ceiling on how many conversation messages a backend will
        retain at any moment. When the store exceeds this cap it MUST evict
        the *oldest* messages first while preserving the order of the messages
        that remain. ``None`` (the default) means unlimited history.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        if max_messages is not None:
            if isinstance(max_messages, bool) or not isinstance(max_messages, int):
                raise TypeError("max_messages must be an int or None")
            if max_messages < 1:
                raise ValueError("max_messages must be >= 1 when provided")
        self._max_messages = max_messages

    @property
    def max_messages(self) -> int | None:
        """
        The maximum number of messages this memory will retain, or ``None``
        for unlimited history. Implementations enforce this on ``add``.
        """
        return self._max_messages

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