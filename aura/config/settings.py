"""
config/settings.py
------------------
Loads and VALIDATES AURA's configuration from environment variables and the
`.env` file. Every other file imports the single `settings` object from here,
so configuration lives in exactly one place.

AURA is provider-agnostic: you pick the LLM provider in `.env` with
AURA_LLM_PROVIDER and provide that provider's API key. Each provider's key is
optional at import time (so AURA can start even before you paste a key into
your `.env`); the provider factory (`brain/factory.py`) raises a clear error
if the *selected* provider's key is missing.

Why pydantic-settings (our chosen tech)?
  * It checks types for us (e.g. temperature must be a number).
  * It reads the `.env` file automatically.
  * A missing/optional secret does not crash configuration loading.
"""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All of AURA's configuration in one validated object."""

    # Read from a file called ".env", and expect every variable to start
    # with the prefix "AURA_" (so AURA_MODEL fills the `model` field below).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AURA_",
        extra="ignore",
    )

    # Which brain provider to use. Supported: "nemotron", "anthropic".
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "aura_memory"

    llm_provider: str = "nemotron"

    # --- Anthropic / Claude (used when llm_provider == "anthropic") ---
    anthropic_api_key: str | None = None

    # --- NVIDIA Nemotron (used when llm_provider == "nemotron") ---
    nvidia_api_key: str | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # The model for whatever provider is currently selected. Model names change
    # over time, so keep this in .env and update it if a provider reports an
    # unknown/"model not found" ID.
    model: str = "nvidia/nemotron-3-ultra-550b-a55b"

    # Response "creativity": 0.0 = focused/deterministic, 1.0 = more creative.
    # Kept low (0.2) so simple factual questions get direct, consistent answers
    # instead of creative drift (a voice-mode correctness bug).
    temperature: float = 0.2

    # --- Kokoro TTS (used by the local voice runner and /ws/live) ---
    kokoro_model: str = "models/kokoro-v1.0.onnx"
    kokoro_voices: str = "models/voices-v1.0.bin"
    kokoro_voice: str = "af_sarah"
    kokoro_lang: str = "en-us"
    kokoro_speed: float = 1.0

    # --- Validation (runs once at import / .env load) ---
    # These catch misconfiguration EARLY instead of sending a bad request to a
    # live LLM (which costs time, money, and can bake subtle bugs into history).

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("AURA_TEMPERATURE must be between 0.0 and 2.0")
        return v

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("AURA_MODEL must not be empty")
        return v

    @field_validator("kokoro_model", "kokoro_voices", "kokoro_voice", "kokoro_lang")
    @classmethod
    def _check_kokoro_text_fields(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("Kokoro settings must not be empty")
        return v.strip()

    @field_validator("kokoro_speed")
    @classmethod
    def _check_kokoro_speed(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("AURA_KOKORO_SPEED must be greater than 0")
        return v


# One shared instance imported by the rest of AURA.
# Creating it here means config is loaded & validated as soon as AURA starts.
settings = Settings()
