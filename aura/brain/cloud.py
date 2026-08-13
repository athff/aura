"""
brain/cloud.py
--------------
A concrete Brain that "thinks" by calling a cloud LLM (Anthropic's Claude).

This is our first implementation of the Brain contract from base.py. It is the
ONLY file that knows the details of the Anthropic API; the rest of AURA just
calls `think()` and gets text back.
"""

from anthropic import Anthropic

from aura.brain.base import Brain
from aura.config.settings import settings
from aura.core.errors import ProviderError
from aura.core.logging import get_logger

# The "system prompt" defines AURA's personality and standing instructions.
# We will grow this as AURA gains capabilities.
SYSTEM_PROMPT = (
    "You are AURA (Artificial Universal Reasoning Assistant), a helpful, "
    "concise, and capable personal AI assistant. You are calm, precise, and "
    "friendly, and you explain things clearly."
)

logger = get_logger(__name__)


class CloudBrain(Brain):
    """Generates replies using the Anthropic API."""

    def __init__(self) -> None:
        # The client uses the API key we loaded and validated in settings.
        self._client = Anthropic(api_key=settings.anthropic_api_key)

    def think(self, messages: list[dict]) -> str:
        try:
            response = self._client.messages.create(
                model=settings.model,
                max_tokens=1024,
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

        return str(text).strip()
