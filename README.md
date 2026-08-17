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

## Voice (optional)

AURA ships local/offline speech providers behind the `SpeechInput` /
`SpeechOutput` contracts (`aura/interaction/voice/`):

* **Speech-to-text** — `VoskSpeechInput` (Vosk, Kaldi-based)
* **Text-to-speech** — `KokoroSpeechOutput` (Kokoro-82M ONNX)
* **Microphone/Speaker** — `SoundDeviceHardware` (Windows, via `sounddevice`)

These are OPTIONAL: none of them are needed by the CLI, the web GUI, or the
offline test suite. Enable them only when you actually want voice:

```bat
:: 1. Install the optional voice engines + hardware adapter
pip install -r requirements-voice.txt

:: 2. Download models (these are separate downloads, not pip packages)
::    - a Vosk model  ->  https://alphacephei.com/vosk/models
::    - Kokoro model  ->  https://github.com/thewh1teagle/kokoro-onnx/releases
```

Then point the providers at those models:

```python
from aura.interaction.voice import KokoroSpeechOutput, VoskSpeechInput

stt = VoskSpeechInput(model_path="path/to/vosk-model-small-en-us")
tts = KokoroSpeechOutput(
    model_path="path/to/kokoro-v1.0.onnx",
    voices_path="path/to/voices-v1.0.bin",
    voice="af_sarah",
    lang="en-us",
    speed=1.0,
)
```

Both providers raise `SpeechError` when the engine/model is unavailable, so AURA
can surface a clear startup error until they are configured. `read()` captures
one bounded utterance and transcribes it; `write()` synthesizes and plays it
back through the default speaker.

To turn those providers into a voice modality (a `VoiceInteraction`), inject
the STT/TTS plus the real `SoundDeviceHardware` adapter (which supplies the
microphone `capture` and speaker `playback` callables). `Session` drives it
exactly like the text terminal:

```python
from aura.interaction import (
    Session,
    VoiceInteraction,
    VoskSpeechInput,
    KokoroSpeechOutput,
    SoundDeviceHardware,
)

hardware = SoundDeviceHardware()   # real Windows mic + speaker

voice = VoiceInteraction(
    speech_input=VoskSpeechInput(model_path="path/to/vosk-model"),
    speech_output=KokoroSpeechOutput(
        model_path="path/to/kokoro-v1.0.onnx",
        voices_path="path/to/voices-v1.0.bin",
    ),
    capture=hardware.capture,       # one bounded utterance per call
    playback=hardware.playback,     # plays one Audio payload
)
# Pass output_prefix="" so voice only speaks AURA's reply, not a text prefix.
Session(interaction=voice, engine=engine, output_prefix="").run()
```

`SoundDeviceHardware` is configurable (sample rate, channels, and a hard
`recording_limit`) and is the ONLY module that touches the microphone/speaker —
it records one bounded utterance per call (stopping on silence or after the
recording limit) and plays one `Audio` payload per call. It raises `SpeechError`
when the hardware/library is unavailable and `InteractionEnd` on Ctrl-C. No
wake-word, continuous listening, or streaming. Swap in the stubs or the real
providers without changing the callers.

### Run AURA by voice

A ready-made entry point wires all of the above together —
`SoundDeviceHardware` + `VoskSpeechInput` + `KokoroSpeechOutput` +
`VoiceInteraction` + `AuraEngine` + `Session` — with no code changes. It starts
one bounded voice conversation and
speaks AURA's replies through your speakers:

```bat
:: 1. Install the optional voice engines + hardware adapter
pip install -r requirements-voice.txt

:: 2. Download models (as shown above) and place them by default at:
::      models\vosk-model-small-en-us-0.15   (Vosk STT, unzip into this folder)
::      models\kokoro-v1.0.onnx              (Kokoro ONNX model)
::      models\voices-v1.0.bin               (Kokoro voices pack)
::    OR set the paths in .env / your shell (see below).

:: 3. Set up your LLM brain exactly like the text CLI (.env with your API key)

:: 4. Run it
python -m aura.voice
```

Speak, and AURA replies aloud. Say **"exit"** or **"quit"** to stop; recording
is bounded per turn (silence or up to `AURA_VOICE_RECORDING_LIMIT` seconds).

Configuration (optional; defaults shown). All may be set in the shell or in
`.env`:

```env
AURA_VOSK_MODEL=models/vosk-model-small-en-us-0.15
AURA_KOKORO_MODEL=models/kokoro-v1.0.onnx
AURA_KOKORO_VOICES=models/voices-v1.0.bin
AURA_KOKORO_VOICE=af_sarah
AURA_KOKORO_LANG=en-us
AURA_KOKORO_SPEED=1.0
AURA_VOICE_SAMPLE_RATE=16000
AURA_VOICE_CHANNELS=1
AURA_VOICE_RECORDING_LIMIT=5.0
```

If a model path is wrong or missing, the runner exits with a clear message
pointing to the download and the exact path it looked for — you never get a
mysterious crash.

## Tests

```bat
pip install -r requirements-dev.txt
pytest
```

The suite runs fully offline using stub brains (no API keys, no network).
