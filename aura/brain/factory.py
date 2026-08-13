"""
brain/factory.py
----------------
A tiny "factory" that turns AURA's configured provider name into the right
concrete Brain. Everything else in AURA depends only on the abstract `Brain`
contract from base.py -- never on a specific provider -- so this is the single
place that knows which concrete class maps to which provider name.

To add a provider later (e.g. DeepSeek, OpenAI, a local model) you only need:
  1. Add a new brain class under brain/ that implements Brain.think().
  2. Register it here with one line.
The engine, the entry point, and the tests stay unchanged.
"""

from aura.brain.base import Brain
from aura.brain.cloud import CloudBrain
from aura.brain.nemotron import NemotronBrain
from aura.config.settings import settings


def create_brain() -> Brain:
    """Return the concrete Brain for the configured AURA_LLM_PROVIDER."""
    provider = settings.llm_provider.strip().lower()

    if provider == "nemotron":
        return NemotronBrain()
    if provider == "anthropic":
        return CloudBrain()

    raise ValueError(
        f"Unknown AURA_LLM_PROVIDER: {settings.llm_provider!r}. "
        "Supported values: 'nemotron', 'anthropic'."
    )