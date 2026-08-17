"""
Offline tests for AURA's concise, voice-optimized SYSTEM_PROMPT.

These run fully offline -- importing the brain modules only loads the provider
clients (never making a network call) and never touches any audio or models.
"""

from aura.brain.cloud import SYSTEM_PROMPT as CLOUD_PROMPT
from aura.brain.nemotron import SYSTEM_PROMPT as NEMOTRON_PROMPT

_ALL_PROMPTS = (NEMOTRON_PROMPT, CLOUD_PROMPT)


def test_system_prompt_requests_short_concise_answers() -> None:
    for prompt in _ALL_PROMPTS:
        lower = prompt.lower()
        assert "concise" in lower
        assert "1-2" in prompt and "sentences" in lower
        assert "20-50" in prompt and "words" in lower


def test_system_prompt_allows_longer_answers_when_detail_is_requested() -> None:
    for prompt in _ALL_PROMPTS:
        lower = prompt.lower()
        assert "explain in detail" in lower or "more detail" in lower
        assert "tell me more" in lower or "longer" in lower


def test_system_prompt_is_optimized_for_spoken_text_to_speech() -> None:
    for prompt in _ALL_PROMPTS:
        lower = prompt.lower()
        # No markdown / bullet lists by default, and TTS-optimized wording.
        assert "markdown" in lower or "bullet" in lower
        assert "spoken" in lower or "text-to-speech" in lower


def test_system_prompt_discourages_filler_and_question_repetition() -> None:
    for prompt in _ALL_PROMPTS:
        lower = prompt.lower()
        assert "repeat the" in lower and "user's question" in lower
        assert "filler" in lower


def test_system_prompt_makes_latest_question_primary() -> None:
    # Correctness rule: the most recent user message is always the operative
    # question; older context must never override the current question. This
    # addresses wrong/unrelated answers to simple factual questions in voice.
    for prompt in _ALL_PROMPTS:
        lower = prompt.lower()
        assert "most recent" in lower
        assert "current question" in lower
        assert "factual" in lower and "direct" in lower
