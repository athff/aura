"""
web/app.py
----------
The FastAPI web layer for AURA.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

Architecture (the browser NEVER talks to a cloud provider directly):

    Browser  ->  FastAPI (this file)  ->  AuraEngine  ->  Brain
                                                         -> Nemotron / Anthropic

Routes:
    GET    /            -> serves the web interface (dark chat UI)
    GET    /api/health  -> JSON liveness/status probe (no secrets)
    POST   /api/chat    -> {"message": "Hello AURA"}  ->  {"reply": "..."}

Notes
-----
* We reuse the SAME AuraEngine for every request so conversation history is
  preserved across the whole session. The engine is created once when the app
  is built (server start) via create_brain(), which picks the provider from
  .env (AURA_LLM_PROVIDER).
* No API keys ever reach the frontend; the browser only talks to us.
* If the selected provider isn't configured yet (e.g. no NVIDIA key in .env),
  the server still starts; /api/chat then returns a friendly 503 so the UI can
  show a clear error instead of crashing.
* Expected AURA errors (BrainError / ProviderError) are translated to a clean
  502 that never leaks provider internals to the browser.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aura.brain.factory import create_brain
from aura.config.settings import settings
from aura.core.engine import AuraEngine
from aura.core.errors import AuraError
from aura.core.logging import get_logger

# This folder holds the web frontend files (index.html, style.css, app.js).
WEB_DIR = Path(__file__).resolve().parent

logger = get_logger(__name__)

# Punctuation that reliably ends a spoken sentence. Streaming splits the brain
# reply on these so completed sentences are handed to Kokoro as soon as they are
# available, letting audio/playback begin before the whole response has finished.
_SENTENCE_ENDERS = (".", "!", "?")


def _has_sentence_end(text: str) -> bool:
    """Return True if ``text`` contains a terminal sentence boundary (. ! ?)."""
    return any(ch in text for ch in _SENTENCE_ENDERS)


def build_engine() -> AuraEngine:
    """Build AURA's single engine + brain for the running provider."""
    return AuraEngine(brain=create_brain())


def create_app(engine: AuraEngine | None = None) -> FastAPI:
    """
    Build the FastAPI application.

    Pass `engine` for tests (e.g. one backed by a stub brain). When `engine`
    is None (normal server start) we build the real engine exactly once. If
    that fails (provider key missing), we record the reason and keep the app
    running so the UI still loads and can report the error.
    """
    app_state: dict = {
        "engine": engine,
        "error": None,
        "stt": None,
        "stt_error": None,
        "tts": None,
        "tts_error": None,
    }
    if engine is None:
        try:
            app_state["engine"] = build_engine()
            logger.info("AURA engine built at startup (provider=%s)", settings.llm_provider)
        except Exception as exc:  # e.g. missing API key for the provider
            # Keep the server running so the UI still loads and can report it.
            app_state["error"] = str(exc)
            logger.error("AURA engine could not be built at startup: %s", exc)

    try:
        from aura.voice.__main__ import build_speech_input, build_speech_output

        app_state["stt"] = build_speech_input()
        logger.info("AURA STT engine built at startup")
    except Exception as exc:
        app_state["stt_error"] = str(exc)
        logger.error("AURA STT engine could not be built at startup: %s", exc)

    try:
        from aura.voice.__main__ import build_speech_output

        app_state["tts"] = build_speech_output()
        logger.info("AURA TTS engine built at startup")
    except Exception as exc:
        app_state["tts_error"] = str(exc)
        logger.error("AURA TTS engine could not be built at startup: %s", exc)

    app = FastAPI(
        title="AURA",
        docs_url=None,  # keep the served surface minimal
        redoc_url=None,
    )

    class ChatRequest(BaseModel):
        message: str

    class ChatResponse(BaseModel):
        reply: str

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    # Static assets for the page (same origin, no CORS needed).
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/api/health")
    def health() -> dict:
        """Liveness/status probe. Never exposes API keys or secrets, and never
        touches the provider -- this works even when the brain is unconfigured.
        """
        return {
            "status": "ok",
            "service": "aura",
            "provider": settings.llm_provider,
            "engine_ready": app_state["engine"] is not None,
            "configured": app_state["error"] is None,
        }

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        if app_state["error"] is not None:
            raise HTTPException(
                status_code=503,
                detail=f"Brain is not configured: {app_state['error']}",
            )

        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        engine: AuraEngine = app_state["engine"]
        try:
            reply = engine.send(message)
        except AuraError as exc:
            # Expected AURA failures: never leak provider internals to the user.
            logger.warning("Chat failed with an expected AURA error: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="AURA could not produce a reply right now.",
            )
        except Exception as exc:
            # Unexpected failures must NEVER leak internals to the browser.
            # Log the FULL exception server-side via AURA's central logger,
            # then return a clean, generic error to the client.
            logger.exception("Chat failed with an unexpected error: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="AURA could not produce a reply right now.",
            )

        return ChatResponse(reply=reply)

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket) -> None:
        """Minimal Live Mode WebSocket: accepts {"type":"text","text":"..."}.

        Sends a status then posts the reply in post-hoc partial chunks followed by
        a complete message. Reuses the same AuraEngine built into the app.
        """
        await ws.accept()

        engine: AuraEngine | None = app_state.get("engine")
        if engine is None:
            # Engine not ready at startup: tell the client and close.
            await ws.send_json({"type": "error", "detail": "Engine is not configured"})
            await ws.close()
            return

        try:
            while True:
                # We accept either JSON with text/audio or binary frames; prefer
                # JSON for simplicity. Use receive_json which will raise if a
                # binary frame arrives — catch and handle separately if needed.
                payload = await ws.receive_json()
                if not isinstance(payload, dict):
                    await ws.send_json({"type": "error", "detail": "invalid payload"})
                    continue

                ptype = payload.get("type")

                # ----- Audio upload path: client sends base64-encoded WAV blob -----
                if ptype == "audio":
                    # Notify client we're transcribing
                    logger.info('/ws/live: received audio payload, starting transcribe')
                    await ws.send_json({"type": "status", "state": "transcribing"})
                    b64 = payload.get("data")
                    fmt = payload.get("format", "wav")
                    if not b64:
                        await ws.send_json({"type": "error", "detail": "no audio data"})
                        continue

                    try:
                        import base64

                        audio_bytes = base64.b64decode(b64)
                    except Exception:
                        await ws.send_json({"type": "error", "detail": "invalid audio encoding"})
                        continue

                    # Build an Audio object accepted by existing STT providers
                    from aura.interaction.voice.base import Audio as AudioObj

                    audio_obj = AudioObj(data=bytes(audio_bytes), format=fmt)

                    # Use the preloaded VoskSpeechInput from startup; if it
                    # failed to initialize, the stored error is sent below.
                    stt = app_state.get("stt")
                    if stt is None:
                        logger.error("STT engine unavailable for /ws/live: %s", app_state.get("stt_error"))
                        await ws.send_json(
                            {
                                "type": "error",
                                "detail": app_state.get("stt_error") or "STT engine unavailable on server",
                            }
                        )
                        continue

                    # Call STT (may raise SpeechError -> treat as client-visible error)
                    try:
                        transcript = stt.transcribe(audio_obj)
                        logger.info('/ws/live: STT transcript: %s', transcript)
                    except Exception as exc:  # include SpeechError
                        logger.exception("STT failed: %s", exc)
                        await ws.send_json({"type": "error", "detail": "speech recognition failed"})
                        continue

                    # Send the transcript back to the client and proceed as text
                    await ws.send_json({"type": "transcript", "text": transcript})

                    # Feed the transcript into the same engine path as typed text
                    user_text = transcript.strip()
                    if not user_text:
                        # empty transcript -> no reply
                        await ws.send_json({"type": "complete", "text": ""})
                        continue

                    # fall through to the text->engine path below by setting ptype
                    ptype = "text"
                    payload = {"text": user_text}

                # ----- Text path (typed or post-STT) -----------------------------
                if ptype == "text":
                    user_text = str(payload.get("text", "")).strip()
                    if not user_text:
                        await ws.send_json({"type": "error", "detail": "text cannot be empty"})
                        continue

                    # Indicate work has started.
                    await ws.send_json({"type": "status", "state": "thinking"})

                    tts = app_state.get("tts")
                    if tts is None:
                        logger.error("TTS engine unavailable for /ws/live: %s", app_state.get("tts_error"))
                        await ws.send_json(
                            {
                                "type": "error",
                                "detail": app_state.get("tts_error") or "TTS engine unavailable on server",
                            }
                        )
                        continue

                    import base64 as _b64
                    import asyncio as _asyncio
                    import queue as _queue
                    import threading as _threading

                    # Bounded producer/consumer pipeline so Nemotron token streaming and
                    # Kokoro TTS overlap in time. The producer keeps consuming tokens (so
                    # the brain streams the NEXT sentence) WHILE a background thread
                    # synthesizes the already-completed sentence into audio. A single FIFO
                    # worker + ordered queue guarantee audio chunks never overlap or play
                    # out of order. Both queues are bounded so memory stays capped; the
                    # producer only back-pressures (briefly) if the worker falls far
                    # behind a burst of very short sentences.
                    _STOP = object()
                    sent_q = _queue.Queue(maxsize=8)   # completed sentences -> Kokoro worker
                    audio_q = _queue.Queue(maxsize=8)  # (format, wav-bytes) -> websocket
                    _interrupt_event = _threading.Event()  # set from WS handler when barge_in received

                    def _kokoro_worker() -> None:
                        """Kokoro synthesizer: sentence in, WAV out. One worker = FIFO order."""
                        while True:
                            # Wait for next item; interrupt lets us exit cleanly without
                            # blocking forever if the caller has already set the event.
                            try:
                                # Single dequeue: the item we get here IS the one we
                                # process (previously a first get() was discarded and a
                                # second get() consumed the next item, dropping sentences).
                                item = sent_q.get(
                                    timeout=0.5 if _interrupt_event.is_set() else None
                                )
                            except _queue.Empty:
                                if _interrupt_event.is_set():
                                    return
                                continue
                            if _interrupt_event.is_set():
                                # Barge-in requested: abandon the not-yet-synthesized
                                # sentence and stop.
                                sent_q.task_done()
                                return
                            if item is _STOP:
                                sent_q.task_done()
                                return
                            sent_q.task_done()
                            try:
                                audio_obj = tts.synthesize(item)
                                fmt = getattr(audio_obj, "format", "wav") or "wav"
                                audio_q.put((fmt, bytes(audio_obj.data)))
                            except Exception:
                                logger.exception("Kokoro worker failed to synthesize a chunk")

                    worker = _threading.Thread(
                        target=_kokoro_worker, name="kokoro-stream-worker", daemon=True
                    )
                    worker.start()

                    async def _drain_audio() -> None:
                        """Send every audio chunk already synthesized, in FIFO order.

                        Respects _interrupt_event: if interrupt is set, drain only what
                        has already been queued then return immediately — no new chunks
                        are fetched.
                        """
                        while True:
                            # If interrupt has been requested, drain whatever is already
                            # in the queue and then return right away.
                            if _interrupt_event.is_set():
                                try:
                                    fmt, data = audio_q.get_nowait()
                                except _queue.Empty:
                                    return
                            else:
                                try:
                                    fmt, data = audio_q.get_nowait()
                                except _queue.Empty:
                                    return
                            await ws.send_json(
                                {
                                    "type": "audio",
                                    "format": fmt,
                                    "data": _b64.b64encode(data).decode("ascii"),
                                }
                            )
                            # After one chunk, re-check interrupt before fetching more.
                            if _interrupt_event.is_set():
                                return

                    async def _enqueue(text: str) -> None:
                        """Push one completed sentence to the Kokoro worker (bounded)."""
                        while True:
                            try:
                                sent_q.put_nowait(text)
                                return
                            except _queue.Full:
                                # Worker momentarily behind: drain its finished audio and
                                # yield so it can pick up the next item and free a slot.
                                await _drain_audio()
                                await _asyncio.sleep(0.01)

                    async def _stop_worker(*, abandon: bool = True) -> None:
                        """Signal the worker to finish; never blocks the event loop.
                        
                        ``abandon=True`` (default, used by barge-in / error paths) also
                        sets the interrupt event so the worker drops any not-yet-
                        synthesized sentence. ``abandon=False`` (clean end-of-reply)
                        only posts the ``_STOP`` sentinel so the worker drains every
                        queued sentence into audio before exiting -- i.e. TTS/audio
                        still complete while partial/complete/audio_done are still sent.
                        """
                        if abandon:
                            _interrupt_event.set()
                        try:
                            sent_q.put_nowait(_STOP)
                        except _queue.Full:
                            pass  # Worker will pick up sentinel on next iteration; interrupt event also forces exit.

                    full = ""
                    pending = ""  # text accumulated since the last sentence boundary
                    try:
                        # Stream the brain reply incrementally. Completed sentences are
                        # handed to the Kokoro worker immediately so synthesis runs in
                        # parallel with the brain still streaming the rest of the reply.
                        for token in engine.send_stream(user_text):
                            full += token
                            pending += token
                            # Before processing the next sentence, check whether a
                            # barge_in was requested. If so, abandon this reply entirely.
                            if _interrupt_event.is_set():
                                # Drain any already-synthesized audio one last time, then
                                # stop. Do NOT send queued audio after interruption.
                                await _drain_audio()
                                await ws.send_json({"type": "barge_in"})
                                # Wait for Kokoro worker to finish, then send audio_done.
                                await _asyncio.to_thread(worker.join)
                                try:
                                    await ws.send_json({"type": "audio_done"})
                                except Exception:
                                    pass
                                pending = ""
                                # Do NOT send partial/complete/reply — the interruption
                                # has already been signalled. Fall through to the
                                # post-loop cleanup that returns to listening.
                                break
                            if _has_sentence_end(pending):
                                # Stream the text so far for live display, then hand the
                                # completed sentence to the Kokoro worker immediately so it
                                # synthesizes in parallel with the brain still streaming.
                                await ws.send_json({"type": "partial", "text": full})
                                await _enqueue(pending.strip())
                                pending = ""
                            # Every iteration (whichever branch) flush finished audio so
                            # synthesized sentences stream out progressively -- especially
                            # when each brain delta is itself a complete sentence and there
                            # is no token-level gap to drain in.
                            await _drain_audio()
                    except AuraError:
                        await _stop_worker(abandon=True)
                        await ws.send_json({"type": "error", "detail": "AURA could not produce a reply right now."})
                        continue
                    except Exception:
                        await _stop_worker(abandon=True)
                        logger.exception("WebSocket handler failed while producing a reply")
                        await ws.send_json({"type": "error", "detail": "internal server error"})
                        continue

                    reply = full.strip()
                    if not reply:
                        await _stop_worker(abandon=True)
                        await ws.send_json({"type": "error", "detail": "AURA could not produce a reply right now."})
                        continue

                    # Flush any trailing text with no clear sentence boundary (e.g. a
                    # single-word reply) as final audio, then end the stream.
                    if pending.strip() and not _interrupt_event.is_set():
                        await _enqueue(pending.strip())
                    await _stop_worker(abandon=False)
                    # Wait for Kokoro to finish every queued sentence, then send all the
                    # produced audio before the complete reply, keeping the client's
                    # partial/complete and audio/audio_done contract intact.
                    await _asyncio.to_thread(worker.join)
                    await _drain_audio()

                    # A final partial (the complete streamed text) then the complete
                    # reply, keeping the client's partial/complete contract intact.
                    # Only send these if we weren't interrupted.
                    if not _interrupt_event.is_set():
                        await ws.send_json({"type": "partial", "text": full})
                        logger.info("/ws/live: sending complete reply")
                        await ws.send_json({"type": "complete", "text": reply})

                        # Deterministic end-of-audio marker for clients/tests.
                        try:
                            await ws.send_json({"type": "audio_done"})
                        except Exception:
                            pass

                    continue

                # Unknown payload type
                await ws.send_json({"type": "error", "detail": "unknown message type"})
                continue

        except WebSocketDisconnect:
            logger.info("/ws/live client disconnected")
        except Exception:
            logger.exception("Unexpected error in /ws/live WebSocket handler")
            try:
                await ws.close()
            except Exception:
                pass

    return app


if __name__ == "__main__":
    import uvicorn

    # Create the app (and thus AURA's single engine) once, then serve it.
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)