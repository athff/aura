# AURA — Artificial Universal Reasoning Assistant

A JARVIS-inspired personal AI assistant, built step by step with a clean,
modular architecture so capabilities (memory, voice, vision, agents, IoT)
can be added over time without rewrites.

## Current status

**Phase 1 — Conversational Core.** AURA can hold a text conversation in your
terminal using a cloud LLM, and also from a local web chat GUI.

## Architecture (in one picture)

```
You (terminal)  ->  AuraEngine (orchestrator)  ->  Brain (the LLM)
                          ^                            |
                          +----------- reply ----------+
```

Web GUI:

```
Browser  ->  FastAPI  ->  AuraEngine  ->  Brain
                                        -> Nemotron / Anthropic
```

The browser only ever talks to FastAPI — it never contacts NVIDIA or Anthropic
directly, and no API keys reach the browser.

- `aura/config/` — loads settings & secrets from `.env`
- `aura/brain/`  — the "brain" contract (`base.py`) + providers (`cloud.py`, `nemotron.py`) + `factory.py`
- `aura/core/`   — the orchestrator that manages the conversation
- `aura/main.py` — the terminal entry point (chat loop)
- `aura/web/`    — the local web GUI (FastAPI + HTML/CSS/vanilla JS)

## Setup (Windows)

```bat
:: 1. From the project folder, create a virtual environment
python -m venv .venv

:: 2. Activate it
.venv\Scripts\activate

:: 3. Install dependencies
pip install -r requirements.txt

:: 4. Create your .env from the template, then paste your API key into it
copy .env.example .env

:: 5. Run AURA
python -m aura.main
```

Type `exit` or `quit` to stop.

To run the **local web GUI** instead:

```bat
:: 1. (setup) activate the venv and install deps as above

:: 2. Run the web server
python -m aura.web.app

:: 3. Open the chat in your browser
::    http://127.0.0.1:8000
```

If the configured provider has no API key yet (no `.env` / no key pasted), the
server still starts and the page loads; AURA will show you a clear error when
you send a message.

## Tests

```bat
pip install -r requirements-dev.txt
pytest
```

The suite runs fully offline using stub brains (no API keys, no network).
