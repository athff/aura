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
    """In-memory conversation store using the standard ``{role, content}`` format.

    When ``max_messages`` is provided, the store keeps at most that many
    messages: adding a new turn past the cap evicts the oldest messages first
    while preserving the relative order of the messages that remain.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        super().__init__(max_messages)
        self._messages: list[dict] = []

    def add(self, role: str, content: str) -> None:
        """Append one message turn to the end of the conversation.

        If the conversation now exceeds ``max_messages``, the oldest messages
        are removed first so the newest ``max_messages`` turns are kept, in
        order.
        """
        self._messages.append({"role": role, "content": content})
        self._trim()

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

    def _trim(self) -> None:
        """Evict oldest messages first once the store exceeds the cap."""
        limit = self.max_messages
        if limit is None:
            return
        excess = len(self._messages) - limit
        if excess > 0:
            # Dropping from the front removes the oldest turns while keeping
            # the relative order of what remains.
            del self._messages[:excess]