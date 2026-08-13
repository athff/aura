"""
web/app.py
----------
The FastAPI web layer for AURA.

Architecture (the browser NEVER talks to a cloud provider directly):

    Browser  ->  FastAPI (this file)  ->  AuraEngine  ->  Brain
                                                         -> Nemotron / Anthropic

Routes:
    GET  /           -> serves the web interface (dark chat UI)
    POST /api/chat   -> {"message": "Hello AURA"}  ->  {"reply": "..."}

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
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aura.brain.factory import create_brain
from aura.core.engine import AuraEngine

# This folder holds the web frontend files (index.html, style.css, app.js).
WEB_DIR = Path(__file__).resolve().parent


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
        except Exception as exc:  # e.g. missing API key for the provider
            app_state["error"] = str(exc)

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
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"AURA hit an error: {exc}")

        return ChatResponse(reply=reply)

    return app


if __name__ == "__main__":
    import uvicorn

    # Create the app (and thus AURA's single engine) once, then serve it.
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)