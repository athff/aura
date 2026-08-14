"""
memory/conversation.py
----------------------
An in-memory ``Memory`` implementation that stores conversation messages in a
plain list. This is the default backend for AuraEngine.

It keeps messages in insertion order, supports appending new turns, returns the
conversation as a copy of ``{role, content}`` dicts, and can clear all history.
Internal state is never exposed to callers: ``messages()`` returns a defensive
copy so mutations can't leak into the stored conversation.
"""

from aura.memory.base import Memory


class ConversationMemory(Memory):
    """In-memory conversation store using the standard ``{role, content}`` format."""

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Append one message turn to the end of the conversation."""
        self._messages.append({"role": role, "content": content})

    def messages(self) -> list[dict]:
        """
        Return the conversation as a defensive copy.

        Each message dict is copied so the caller can't mutate the stored list
        or its entries (``content`` is a plain string and is already immutable).
        """
        return [dict(message) for message in self._messages]

    def clear(self) -> None:
        """Remove the entire conversation history."""
        self._messages.clear()