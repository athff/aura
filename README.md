# AURA — Artificial Universal Reasoning Assistant

A JARVIS-inspired personal AI assistant, built step by step with a clean,
modular architecture so capabilities (memory, voice, vision, agents, IoT)
can be added over time without rewrites.

## Current status

**Phase 1 — Conversational Core.** AURA can hold a text conversation in your
terminal using a cloud LLM.

## Architecture (in one picture)

```
You (terminal)  ->  AuraEngine (orchestrator)  ->  Brain (the LLM)
                          ^                            |
                          +----------- reply ----------+
```

- `aura/config/` — loads settings & secrets from `.env`
- `aura/brain/`  — the "brain" contract (`base.py`) + a cloud implementation (`cloud.py`)
- `aura/core/`   — the orchestrator that manages the conversation
- `aura/main.py` — the entry point that wires it together and runs the chat loop

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
