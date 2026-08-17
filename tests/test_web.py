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

import time


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


def test_ws_live_text_streaming() -> None:
    """Connect to /ws/live, send a text message, and receive status + partials + complete.

    Uses StubBrain so no external provider is required.
    """
    app = _app_with(StubBrain())
    with TestClient(app) as client:
        with client.websocket_connect("/ws/live") as ws:
            # send the expected JSON payload
            ws.send_json({"type": "text", "text": "hello"})

            # first message must be the 'thinking' status
            msg = ws.receive_json()
            assert msg.get("type") == "status"
            assert msg.get("state") == "thinking"

            # collect partials until the complete message arrives
            partials: list[str] = []
            complete_text: str | None = None
            # loop until we get a complete message or time out via TestClient
            while True:
                msg = ws.receive_json()
                assert isinstance(msg, dict)
                t = msg.get("type")
                if t == "partial":
                    partials.append(msg.get("text", ""))
                elif t == "complete":
                    complete_text = msg.get("text")
                    break
                elif t == "error":
                    pytest.fail(f"received error from WS: {msg.get('detail')}")

            # The complete reply must equal the stubbed brain's reply
            assert complete_text == StubBrain.REPLY
            # At least one partial should have been sent
            assert len(partials) >= 1
            # Each partial should be a substring of the final reply
            for p in partials:
                assert p in complete_text


def test_ws_live_audio_transcription(monkeypatch) -> None:
    """Send a small WAV over /ws/live and verify transcript + reply flow.

    We monkeypatch the build_speech_input helper to return a fake STT that
    deterministically returns a fixed transcript so the test is offline.
    """
    class FakeSTT:
        def transcribe(self, audio):
            # simple deterministic stub
            return "hello from audio"

    # patch the builder used by the WS handler
    import aura.voice.__main__ as voice_main

    monkeypatch.setattr(voice_main, 'build_speech_input', lambda: FakeSTT())

    # Patch TTS to return a tiny WAV without loading Kokoro
    class FakeTTS:
        def synthesize(self, text):
            import io, wave
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b'\x00\x00' * 1600)
            return type('A', (), {'data': buf.getvalue(), 'format': 'wav'})()

    import aura.voice.__main__ as voice_main
    monkeypatch.setattr(voice_main, 'build_speech_output', lambda: FakeTTS())

    app = _app_with(StubBrain())
    # Build a tiny 16kHz mono WAV with 0.1s silence (input)
    import io, wave
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        # 0.1s of silence
        w.writeframes(b'\x00\x00' * 1600)
    b = wav_buf.getvalue()
    import base64
    b64 = base64.b64encode(b).decode('ascii')

    with TestClient(app) as client:
        with client.websocket_connect('/ws/live') as ws:
            # send audio payload
            ws.send_json({'type': 'audio', 'format': 'wav', 'data': b64})

            # Expect transcribing status first
            msg = ws.receive_json()
            assert msg.get('type') == 'status' and msg.get('state') == 'transcribing'

            # Then transcript
            msg = ws.receive_json()
            assert msg.get('type') == 'transcript'
            assert 'hello from audio' in msg.get('text')

            # Then thinking status
            msg = ws.receive_json()
            assert msg.get('type') == 'status' and msg.get('state') == 'thinking'

            # Then partials/complete/audio; capture final
            final = None
            audio_payload = None
            audio_done = False
            final = None
            audio_payload = None
            while True:
                msg = ws.receive_json()
                t = msg.get('type')
                if t == 'partial':
                    continue
                if t == 'complete':
                    final = msg.get('text')
                    continue
                if t == 'audio':
                    audio_payload = msg
                    continue
                if t == 'audio_done':
                    audio_done = True
                    # break when we've received both complete and audio_done
                    if final is not None and audio_payload is not None:
                        break
                    continue
                if t == 'error':
                    pytest.fail('error from ws: ' + str(msg))

            assert final == StubBrain.REPLY
            assert audio_payload is not None
            assert audio_payload.get('format') == 'wav'
            # verify base64 decodes to a WAV container
            import base64 as _b64
            decoded = _b64.b64decode(audio_payload.get('data'))
            assert decoded[:4] == b'RIFF'

