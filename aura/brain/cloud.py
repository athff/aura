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

# The "system prompt" defines AURA's personality and standing instructions.
# We will grow this as AURA gains capabilities.
SYSTEM_PROMPT = (
    "You are AURA (Artificial Universal Reasoning Assistant), a helpful, "
    "concise, and capable personal AI assistant. You are calm, precise, and "
    "friendly, and you explain things clearly."
)


class CloudBrain(Brain):
    """Generates replies using the Anthropic API."""

    def __init__(self) -> None:
        # The client uses the API key we loaded and validated in settings.
        self._client = Anthropic(api_key=settings.anthropic_api_key)

    def think(self, messages: list[dict]) -> str:
        response = self._client.messages.create(
            model=settings.model,
            max_tokens=1024,
            temperature=settings.temperature,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        # The API returns content as a list of blocks; we return the text.
        return response.content[0].text
