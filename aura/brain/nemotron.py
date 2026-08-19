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

from time import perf_counter, sleep
from collections.abc import Iterator

from openai import OpenAI, APIError

from aura.brain.base import Brain
from aura.config.settings import settings
from aura.core.errors import ProviderError
from aura.core.logging import get_logger

# Standing instructions / personality for AURA. We keep a local copy here so
# this module stays independent of cloud.py (no cross-provider coupling).
# These rules keep normal responses short and natural because in voice mode the
# text is read aloud by TTS.
#
# The "Conversation-priority rules" address the voice bug where simple factual
# questions sometimes produced unrelated/wrong answers: the model MUST treat the
# user's most recent message as the current question and answer it directly,
# never letting earlier turns or a previous topic override it. Combined with a
# bounded conversation window and a lower temperature, this keeps factual
# questions anchored on the question actually asked.
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

# ---------------------------------------------------------------------------
# Transient retry for NVIDIA's per-worker "ResourceExhausted" quota.
# ---------------------------------------------------------------------------
# The NVIDIA free endpoint occasionally rejects a request with a transient
# "ResourceExhausted: Worker local total request limit reached (16/16)" when a
# per-worker request quota is momentarily exhausted. It self-resets in seconds.
# We retry ONLY this exact condition, with a small exponential backoff. All
# other errors (auth, invalid requests, genuine rate limits, transport) fail
# fast and are never retried.
_RESOURCE_EXHAUSTED_RETRIES = 3        # total attempts = retries + 1 (max 4)
_RESOURCE_EXHAUSTED_BASE_DELAY = 0.5   # seconds; doubles per retry (0.5/1/2)


def _is_resource_exhausted(exc: BaseException) -> bool:
    """True only for the transient 16/16 per-worker ResourceExhausted signal."""
    if not isinstance(exc, APIError):
        return False
    msg = str(exc).lower()
    return "resourceexhausted" in msg and "limit reached" in msg


def _backoff_delay(attempt: int) -> float:
    """Exponential delay (seconds) before retry number ``attempt`` (0-based)."""
    return _RESOURCE_EXHAUSTED_BASE_DELAY * (2 ** attempt)



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
        # We bound the request with an explicit timeout so a stalled endpoint
        # can't hold a voice turn hostage for the SDK's default (600s); normal
        # responses are short (a few seconds) and stay well under this ceiling.
        self._client = OpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
            timeout=60.0,
        )

    def think(self, messages: list[dict]) -> str:
        # OpenAI-compatible APIs pass the system prompt as a special first
        # "system" message, followed by the conversation turns.
        request_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]

        # Log exactly what AURA is asking the model (the current question is the
        # last user message) so real-voice correctness bugs are visible at the
        # API boundary. Only the message COUNT is logged on the hot path; the
        # full transcript is DEBUG-level to avoid re-serializing the whole
        # request right before the network call.
        logger.info("Nemotron request messages (%d)", len(request_messages))
        logger.debug("Nemotron request messages detail: %r", request_messages)

        started = perf_counter()
        try:
            for attempt in range(_RESOURCE_EXHAUSTED_RETRIES + 1):
                try:
                    response = self._client.chat.completions.create(
                        model=settings.model,
                        # Cap generation length. Default answers are 20-50 words, but
                        # this still leaves plenty of headroom for explicitly-requested
                        # detailed answers while bounding the worst-case latency (the
                        # bulk of the 2-4s round trip is token generation on the 550B).
                        max_tokens=256,
                        temperature=settings.temperature,
                        messages=request_messages,
                    )
                    break
                except Exception as exc:
                    # Retry ONLY the transient 16/16 ResourceExhausted condition.
                    if not _is_resource_exhausted(exc) or attempt == _RESOURCE_EXHAUSTED_RETRIES:
                        raise
                    logger.warning(
                        "Nemotron resource-exhausted (16/16); retry %d/%d in %.1fs",
                        attempt + 1, _RESOURCE_EXHAUSTED_RETRIES, _backoff_delay(attempt))
                    sleep(_backoff_delay(attempt))
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

        latency_ms = (perf_counter() - started) * 1000
        logger.info(
            "Nemotron reply for latest question %r (%d ms): %r",
            messages[-1],
            latency_ms,
            str(content).strip(),
        )
        return str(content).strip()

    def think_stream(self, messages: list[dict]) -> Iterator[str]:
        """Stream Nemotron's reply token-by-token via the OpenAI-compatible API.

        Yields each ``delta.content`` fragment as it arrives so the caller can
        start TTS/playback before the full reply is finished. Conversation
        history/final reply are handled by the caller (``AuraEngine.send_stream``).
        Raises ``ProviderError`` if the request fails or no content is produced.
        """
        request_messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        logger.info("Nemotron streaming request messages (%d)", len(request_messages))
        logger.debug("Nemotron streaming request messages detail: %r", request_messages)

        started = perf_counter()
        produced_any = False
        try:
            for attempt in range(_RESOURCE_EXHAUSTED_RETRIES + 1):
                try:
                    stream = self._client.chat.completions.create(
                        model=settings.model,
                        max_tokens=256,
                        temperature=settings.temperature,
                        messages=request_messages,
                        stream=True,
                    )
                    for chunk in stream:
                        choices = chunk.choices
                        delta = None
                        if choices and choices[0].delta is not None:
                            delta = choices[0].delta.content
                        if not delta:
                            continue
                        produced_any = True
                        yield str(delta)
                    break  # stream finished cleanly; exit the retry loop
                except Exception as exc:
                    # Retry ONLY the transient 16/16 ResourceExhausted, and only
                    # if no token has been produced yet (never retry mid-stream).
                    if produced_any or not _is_resource_exhausted(exc) or attempt == _RESOURCE_EXHAUSTED_RETRIES:
                        raise
                    logger.warning(
                        "Nemotron streaming resource-exhausted (16/16); retry %d/%d in %.1fs",
                        attempt + 1, _RESOURCE_EXHAUSTED_RETRIES, _backoff_delay(attempt))
                    sleep(_backoff_delay(attempt))
        except Exception as exc:
            logger.exception("Nemotron streaming request failed")
            raise ProviderError("The Nemotron provider request failed.") from exc
        finally:
            logger.info(
                "Nemotron streamed reply for latest question %r (%d ms)",
                messages[-1],
                (perf_counter() - started) * 1000,
            )

        if not produced_any:
            logger.warning("Nemotron streaming returned no content")
            raise ProviderError("The Nemotron provider returned an empty reply.")