"""Focused tests for the Nemotron provider's transient ResourceExhausted retry.

Verifies the localized retry/backoff added in brain/nemotron.py:

  * retry succeeds  -- the 16/16 ResourceExhausted error is retried and the
                       request eventually completes;
  * retry exhaustion -- after the bounded retries, the original error is
                        re-wrapped into a ProviderError and surfaced;
  * non-ResourceExhausted errors -- auth / invalid request / generic API
                       errors are NOT retried (single attempt).

The real OpenAI client is replaced with a fake whose ``create()`` returns a
programmed sequence of outcomes, so everything runs offline with no keys/network.
"""

import httpx
import pytest
from openai import APIError

import aura.brain.nemotron as nemotron
from aura.brain.nemotron import NemotronBrain
from aura.core.errors import ProviderError


def _mk_api_error(message: str) -> APIError:
    return APIError(message, request=httpx.Request("POST", "https://example.com/v1"),
                    body={"error": message})


def _resource_exhausted() -> APIError:
    return _mk_api_error("ResourceExhausted: Worker local total request limit reached (16/16)")


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _RespChoice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:  # non-streaming think() success payload
    def __init__(self, content: str) -> None:
        self.choices = [_RespChoice(content)]


class _Delta:
    def __init__(self, content: str) -> None:
        self.content = content


class _StreamChoice:
    def __init__(self, content: str) -> None:
        self.delta = _Delta(content)


class _Chunk:  # one streaming chunk
    def __init__(self, content: str) -> None:
        self.choices = [_StreamChoice(content)]


class _Completions:
    """Yields a programmed sequence of outcomes; each is an Exception or a return."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self._outcomes:
            raise AssertionError("create() called more times than outcomes provided")
        out = self._outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, outcomes: list) -> None:
        self.chat = _Chat(_Completions(outcomes))


def _make_brain(outcomes: list, monkeypatch: pytest.MonkeyPatch):
    """Build a NemotronBrain with a fake client and a recorded no-op sleep."""
    delays: list[float] = []
    monkeypatch.setattr(nemotron, "sleep", lambda seconds: delays.append(seconds))
    from aura.config.settings import settings
    monkeypatch.setattr(settings, "nvidia_api_key", "test-key")
    brain = NemotronBrain()
    fake = _FakeClient(outcomes)
    brain._client = fake
    return brain, fake.chat.completions, delays


USER = [{"role": "user", "content": "hello"}]
# --------------------------------------------------------------------------- #
# 1) Retry succeeds.
# --------------------------------------------------------------------------- #
def test_think_retries_resource_exhausted_then_succeeds(monkeypatch) -> None:
    brain, comp, delays = _make_brain(
        [_resource_exhausted(), _resource_exhausted(), _Response("Hi")], monkeypatch)
    assert brain.think(USER) == "Hi"
    assert comp.calls == 3
    assert delays == [0.5, 1.0]  # exponential backoff: base 0.5, doubled


def test_think_stream_retries_resource_exhausted_then_succeeds(monkeypatch) -> None:
    brain, comp, delays = _make_brain(
        [_resource_exhausted(), _resource_exhausted(),
         [_Chunk("He"), _Chunk("llo")]], monkeypatch)
    assert "".join(brain.think_stream(USER)) == "Hello"
    assert comp.calls == 3
    assert delays == [0.5, 1.0]


# --------------------------------------------------------------------------- #
# 2) Retry exhaustion.
# --------------------------------------------------------------------------- #
def test_think_retry_exhaustion_raises_provider_error(monkeypatch) -> None:
    brain, comp, delays = _make_brain(
        [_resource_exhausted()] * (nemotron._RESOURCE_EXHAUSTED_RETRIES + 1), monkeypatch)
    with pytest.raises(ProviderError):
        brain.think(USER)
    assert comp.calls == nemotron._RESOURCE_EXHAUSTED_RETRIES + 1
    assert len(delays) == nemotron._RESOURCE_EXHAUSTED_RETRIES


def test_think_stream_retry_exhaustion_raises_provider_error(monkeypatch) -> None:
    brain, comp, delays = _make_brain(
        [_resource_exhausted()] * (nemotron._RESOURCE_EXHAUSTED_RETRIES + 1), monkeypatch)
    with pytest.raises(ProviderError):
        list(brain.think_stream(USER))
    assert comp.calls == nemotron._RESOURCE_EXHAUSTED_RETRIES + 1
    assert len(delays) == nemotron._RESOURCE_EXHAUSTED_RETRIES


# --------------------------------------------------------------------------- #
# 3) Non-ResourceExhausted errors are NOT retried.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("error", [
    _mk_api_error("BadRequest: model not found"),
    _mk_api_error("AuthenticationError: invalid api key"),
    _mk_api_error("Some unexpected API error"),
    ProviderError("A generic provider error"),
])
def test_think_does_not_retry_non_resource_exhausted(error, monkeypatch) -> None:
    brain, comp, delays = _make_brain([error], monkeypatch)
    with pytest.raises(ProviderError):
        brain.think(USER)
    assert comp.calls == 1
    assert delays == []


@pytest.mark.parametrize("error", [
    _mk_api_error("BadRequest: model not found"),
    _mk_api_error("AuthenticationError: invalid api key"),
    _mk_api_error("Some unexpected API error"),
])
def test_think_stream_does_not_retry_non_resource_exhausted(error, monkeypatch) -> None:
    brain, comp, delays = _make_brain([error], monkeypatch)
    with pytest.raises(ProviderError):
        list(brain.think_stream(USER))
    assert comp.calls == 1
    assert delays == []

