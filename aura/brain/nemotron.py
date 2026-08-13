"""
brain/nemotron.py
-----------------
A concrete Brain that "thinks" by calling NVIDIA's Nemotron free developer
endpoint (build.nvidia.com / NIM). That endpoint is OpenAI-compatible, so we
use the official `openai` client pointed at NVIDIA's base URL.

Like cloud.py, this is the ONLY file that knows the details of this endpoint;
the rest of AURA just calls `think()` and gets plain text back. The provider
is selected via the factory in brain/factory.py, so this file never has to
change when we add more providers.
"""

from openai import OpenAI

from aura.brain.base import Brain
from aura.config.settings import settings
from aura.core.errors import ProviderError
from aura.core.logging import get_logger

# Standing instructions / personality for AURA. We keep a local copy here so
# this module stays independent of cloud.py (no cross-provider coupling).
SYSTEM_PROMPT = (
    "You are AURA (Artificial Universal Reasoning Assistant), a helpful, "
    "concise, and capable personal AI assistant. You are calm, precise, and "
    "friendly, and you explain things clearly."
)

logger = get_logger(__name__)


class NemotronBrain(Brain):
    """Generates replies using NVIDIA's OpenAI-compatible Nemotron endpoint."""

    def __init__(self) -> None:
        if not settings.nvidia_api_key:
            raise RuntimeError(
                "AURA_LLM_PROVIDER is 'nemotron' but AURA_NVIDIA_API_KEY is "
                "missing. Paste your free NVIDIA API key into your .env "
                "(get one at https://build.nvidia.com/)."
            )
        # Pointing the standard OpenAI client at NVIDIA's base URL makes it
        # talk to NVIDIA. The client is lazy: no network happens until think().
        self._client = OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )

    def think(self, messages: list[dict]) -> str:
        # OpenAI-compatible APIs pass the system prompt as a special first
        # "system" message, followed by the conversation turns.
        request_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

        try:
            response = self._client.chat.completions.create(
                model=settings.model,
                max_tokens=1024,
                temperature=settings.temperature,
                messages=request_messages,
            )
        except Exception as exc:
            # Transport/API errors should never crash AURA for the user. Log the
            # details, then raise a typed error that hides provider internals.
            logger.exception("Nemotron API request failed")
            raise ProviderError("The Nemotron provider request failed.") from exc

        # Robustness: safely extract the reply. Never assume the payload shape.
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            logger.exception("Nemotron returned a malformed response payload")
            raise ProviderError("The Nemotron provider returned a malformed response.") from exc

        # Reject empty replies the same way we reject transport failures.
        if content is None or not str(content).strip():
            logger.warning("Nemotron returned an empty reply")
            raise ProviderError("The Nemotron provider returned an empty reply.")

        return str(content).strip()