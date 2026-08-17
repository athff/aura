"""
core/engine.py
--------------
The AURA Core: the orchestrator.

Its job is to manage a conversation -- remember what has been said, hand it to
the Brain, and return the reply. It is deliberately small right now, but this
is the central place where we will later plug in long-term memory, tools,
and routing between different brains.
"""

from time import perf_counter
from collections.abc import Iterator

from aura.brain.base import Brain
from aura.core.errors import BrainError
from aura.core.logging import get_logger
from aura.memory.base import Memory
from aura.memory.conversation import ConversationMemory

logger = get_logger(__name__)

# Default size of the conversation window held in memory when no Memory is
# injected. ~4 full user/assistant turns (8 messages). Bounding context keeps
# OLD turns from overriding the latest question (correctness) and shrinks the
# prompt sent to the provider (latency). The most recent user message is always
# appended last, so the current question is always present.
DEFAULT_CONTEXT_MESSAGES = 8


class AuraEngine:
    """Coordinates a conversation between the user and a Brain."""

    def __init__(self, brain: Brain, memory: Memory | None = None) -> None:
        # We RECEIVE a brain from the outside instead of creating one here.
        # This is "dependency injection": the engine doesn't care which brain
        # it uses, which makes it easy to test and ready for your hybrid plan.
        self._brain = brain
        # Conversation history is stored behind the `Memory` abstraction so the
        # engine only depends on the contract -- never on a concrete backend.
        # When omitted, default to a bounded ConversationMemory so stale context
        # cannot override the current question and requests stay small.
        self._memory = (
            memory
            if memory is not None
            else ConversationMemory(max_messages=DEFAULT_CONTEXT_MESSAGES)
        )
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
        user_message = str(user_message).strip()
        # Diagnostic: the exact text passed into the engine (post-normalization).
        logger.info("User text -> AuraEngine (normalized): %r", user_message)
        self._memory.add("user", user_message)

        history = self._memory.messages()
        # Diagnostic: the conversation window handed to the brain (the last item
        # is always the current question). Only the message COUNT is logged on the
        # hot path; the full transcript is DEBUG-level so we never re-serialize
        # the whole history to a string on every request right before the API call.
        logger.info("Conversation history sent to brain (%d messages)", len(history))
        logger.debug("Conversation history detail sent to brain: %r", history)

        started = perf_counter()
        try:
            reply = self._brain.think(history)
        except Exception:
            # Log every brain failure through AURA's central logger, then let
            # the error propagate unchanged so the web layer (which already has
            # a generic 502 safety net) can translate it for the user.
            logger.exception("Brain.think failed while processing a user message")
            raise

        logger.info(
            "Brain round-trip for %r took %.0f ms",
            user_message,
            (perf_counter() - started) * 1000,
        )

        # Robustness: never accept a brain that returns nothing usable.
        if reply is None or not str(reply).strip():
            logger.warning("Brain returned an empty reply; rejecting it as BrainError")
            raise BrainError("The brain returned an empty reply.")

        # Normalize: if a provider hands back non-text, coerce it safely.
        reply = str(reply).strip()
        self._memory.add("assistant", reply)
        return reply

    def send_stream(self, user_message: str) -> Iterator[str]:
        """Process one user message, yielding reply fragments as they arrive.

        Behaves like ``send`` (history, robustness, normalization) but yields each
        token/fragment from the brain so the caller can start TTS/playback before
        the full reply is finished. The complete reply (the concatenation of all
        yielded fragments) is still recorded to conversation history, so the
        current question stays answered and memory is preserved.

        Raises ``BrainError`` if the brain yields nothing usable.
        """
        user_message = str(user_message).strip()
        logger.info("User text -> AuraEngine (normalized): %r", user_message)
        self._memory.add("user", user_message)

        history = self._memory.messages()
        logger.info("Conversation history sent to brain (%d messages)", len(history))

        collected: list[str] = []
        started = perf_counter()
        try:
            for token in self._brain.think_stream(history):
                if token:
                    collected.append(str(token))
                    yield str(token)
        finally:
            logger.info(
                "Brain streaming round-trip for %r took %.0f ms",
                user_message,
                (perf_counter() - started) * 1000,
            )

        # Robustness: never accept a brain that produces nothing usable.
        reply = "".join(collected).strip()
        if not reply:
            logger.warning("Brain returned an empty reply; rejecting it as BrainError")
            raise BrainError("The brain returned an empty reply.")

        self._memory.add("assistant", reply)
