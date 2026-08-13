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

from fastapi import FastAPI, HTTPException
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
    app_state: dict = {"engine": engine, "error": None}
    if engine is None:
        try:
            app_state["engine"] = build_engine()
            logger.info("AURA engine built at startup (provider=%s)", settings.llm_provider)
        except Exception as exc:  # e.g. missing API key for the provider
            # Keep the server running so the UI still loads and can report it.
            app_state["error"] = str(exc)
            logger.error("AURA engine could not be built at startup: %s", exc)

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

    return app


if __name__ == "__main__":
    import uvicorn

    # Create the app (and thus AURA's single engine) once, then serve it.
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)