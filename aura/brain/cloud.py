"""
brain/cloud.py
--------------
A concrete Brain that "thinks" by calling a cloud LLM (Anthropic's Claude).

This is our first implementation of the Brain contract from base.py. It is the
ONLY file that knows the details of the Anthropic API; the rest of AURA just
calls `think()` and gets text back.
"""

from time import perf_counter

from anthropic import Anthropic

from aura.brain.base import Brain
from aura.config.settings import settings
from aura.core.errors import ProviderError
from aura.core.logging import get_logger

# The "system prompt" defines AURA's personality and standing instructions.
# These rules keep normal responses short and natural because in voice mode the
# text is read aloud by TTS. The "Conversation-priority rules" make the most
# recent user message the operative question so old context can never override
# the current question (this fixes wrong/unrelated answers to factual questions).
SYSTEM_PROMPT = (
    "You are AURA (Artificial Universal Reasoning Assistant), a fast, friendly "
    "personal AI voice assistant. Answer conversationally in plain spoken "
    "language; never use markdown or bullet lists unless the user asks for it.\n"
    "Concise-response rules:\n"
    "1. For normal questions, reply in 1-2 short sentences (about 20-50 words "
    "maximum).\n"
    "2. Be direct and natural. Do not repeat the user's question, and do not "
    "use filler such as 'Sure, I'd be happy to explain...'.\n"
    "3. If the user explicitly asks for more detail (e.g. 'explain in detail', "
    "'tell me more', 'give me an example'), you may give a longer, thorough "
    "answer.\n"
    "4. Optimize every answer for being spoken aloud by text-to-speech.\n"
    "Conversation-priority rules:\n"
    "5. The user's MOST RECENT message is always the current question. Answer "
    "it FIRST, directly and completely, and always satisfy it on its own."
    "6. Earlier messages are only background context. Never let an older topic "
    "or an earlier question override, replace, or drag your reply away from the "
    "most recent message.\n"
    "7. For simple factual questions, give the direct, correct factual answer "
    "immediately. Do not drift into a story, a tangent, or a previous topic."
)

logger = get_logger(__name__)


class CloudBrain(Brain):
    """Generates replies using the Anthropic API."""

    def __init__(self) -> None:
        # The client uses the API key we loaded and validated in settings.
        self._client = Anthropic(api_key=settings.anthropic_api_key)

    def think(self, messages: list[dict]) -> str:
        started = perf_counter()
        try:
            response = self._client.messages.create(
                model=settings.model,
                max_tokens=256,
                temperature=settings.temperature,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            # Transport/API errors should never crash AURA for the user. Log the
            # details, then raise a typed error that hides provider internals.
            logger.exception("Anthropic API request failed")
            raise ProviderError("The Anthropic provider request failed.") from exc

        # Robustness: safely extract the text. Never assume the payload shape;
        # content is a list of blocks, and we expect a .text on the first one.
        try:
            text = response.content[0].text
        except (AttributeError, IndexError, TypeError) as exc:
            logger.exception("Anthropic returned a malformed response payload")
            raise ProviderError("The Anthropic provider returned a malformed response.") from exc

        # Reject empty replies the same way we reject transport failures.
        if text is None or not str(text).strip():
            logger.warning("Anthropic returned an empty reply")
            raise ProviderError("The Anthropic provider returned an empty reply.")

        latency_ms = (perf_counter() - started) * 1000
        logger.info(
            "Anthropic reply for latest question %r (%d ms): %r",
            messages[-1],
            latency_ms,
            str(text).strip(),
        )
        return str(text).strip()
