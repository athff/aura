"""
Tests for AURA's web layer (FastAPI), using a STUB brain so no real LLM, API
key, or network is ever needed. We inject a stub-backed AuraEngine into the
app exactly the way the real server does at startup -- so these tests confirm
the engine is reused (conversation history persists) but never call NVIDIA or
Anthropic.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import aura.web.app as web_app
from aura.brain.base import Brain
from aura.core.engine import AuraEngine

from conftest import StubBrain
from aura.core.errors import ProviderError


class FailingBrain(Brain):
    """A brain that always raises, to exercise the 502 error path."""

    def think(self, messages: list[dict]) -> str:
        raise RuntimeError("boom")


class BrokenProviderBrain(Brain):
    """A brain that simulates a provider failure via AURA's typed error."""

    def think(self, messages: list[dict]) -> str:
        raise ProviderError("provider internals must stay hidden")


def _app_with(brain: Brain) -> FastAPI:
    """Build a FastAPI app whose engine is backed by the given brain."""
    return web_app.create_app(engine=AuraEngine(brain=brain))


def test_index_serves_web_interface() -> None:
    app = _app_with(StubBrain())
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AURA" in response.text


def test_chat_returns_reply() -> None:
    app = _app_with(StubBrain())
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Hello AURA"})
    assert response.status_code == 200
    assert response.json() == {"reply": StubBrain.REPLY}


def test_chat_rejects_blank_message() -> None:
    app = _app_with(StubBrain())
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 400


def test_conversation_history_persists_across_requests() -> None:
    brain = StubBrain()
    app = _app_with(brain)
    with TestClient(app) as client:
        client.post("/api/chat", json={"message": "first"})
        client.post("/api/chat", json={"message": "second"})

    # Two chats => two calls, and the shared engine remembered both turns.
    assert len(brain.calls) == 2
    second_messages = brain.calls[1]
    assert len(second_messages) == 3
    assert second_messages[0] == {"role": "user", "content": "first"}
    assert second_messages[1] == {"role": "assistant", "content": StubBrain.REPLY}
    assert second_messages[2] == {"role": "user", "content": "second"}


def test_chat_returns_502_on_brain_failure() -> None:
    app = _app_with(FailingBrain())
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "hi"})
    # An unexpected brain failure is a server error from the client's view.
    assert response.status_code == 502
    # The raw internal exception message must never be exposed to the browser.
    body = response.json()
    assert "boom" not in body.get("detail", "")
    assert "boom" not in response.text


def test_chat_returns_503_when_provider_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a missing key at server start: create_brain() raises, but the
    # server must still start and /api/chat must explain the problem.
    def no_key() -> Brain:
        raise RuntimeError("AURA_NVIDIA_API_KEY is missing")

    monkeypatch.setattr(web_app, "create_brain", no_key)

    app = web_app.create_app(engine=None)
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert "missing" in response.json()["detail"]
def test_health_returns_ok_with_valid_engine() -> None:
    app = _app_with(StubBrain())
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "aura"
    assert body["configured"] is True
    assert body["engine_ready"] is True


def test_health_reports_unconfigured_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    # create_brain() raises at startup (e.g. missing key): /api/health must
    # still return 200 "ok" for liveness, but report the engine is not ready.
    def no_key() -> Brain:
        raise RuntimeError("AURA_NVIDIA_API_KEY is missing")

    monkeypatch.setattr(web_app, "create_brain", no_key)
    app = web_app.create_app(engine=None)
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["configured"] is False
    assert body["engine_ready"] is False


def test_chat_returns_502_without_leaking_provider_internals() -> None:
    # A typed ProviderError must become a clean 502 that never reveals the
    # provider's raw message to the browser.
    app = _app_with(BrokenProviderBrain())
    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "provider internals must stay hidden" not in response.text