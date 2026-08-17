"""
brain/base.py
-------------
Defines the CONTRACT that every "brain" in AURA must follow.

A brain is anything that can take a conversation and produce a reply.
Today we have one (a cloud LLM); later we may add a local model for your
'hybrid' plan. Because the rest of AURA depends on this abstract contract
-- not on any specific brain -- we can add or swap brains later WITHOUT
changing the engine or the app. This is called "programming to an interface".
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class Brain(ABC):
    """Abstract base class: the blueprint that all brains implement."""

    @abstractmethod
    def think(self, messages: list[dict]) -> str:
        """
        Take the conversation so far and return AURA's reply as plain text.

        `messages` is a list of turns, e.g.:
            [{"role": "user", "content": "Hello"},
             {"role": "assistant", "content": "Hi!"}]
        """
        raise NotImplementedError

    def think_stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield AURA's reply incrementally, token/fragment by fragment.

        Streaming lets callers begin TTS/playback before the full reply is done.
        This default implementation is the safe, non-streaming fallback: it yields
        the whole reply from ``think()`` as a single chunk. Providers whose API
        supports true token streaming (e.g. Nemotron/OpenAI-compatible) override
        this to yield tokens as they arrive.
        """
        reply = self.think(messages)
        if reply:
            yield reply
