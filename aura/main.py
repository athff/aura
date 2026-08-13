"""
main.py
-------
The entry point. Run this to talk to AURA in your terminal:

    python -m aura.main

It performs "composition" -- creating the parts and wiring them together:
    choose a Brain (via the provider factory)  ->  give it to the Engine  ->
    run a chat loop.

Which provider AURA uses (Nemotron, Claude, ...) comes from AURA_LLM_PROVIDER
in your .env file, so you don't edit this file to switch providers.
"""

from aura.brain.factory import create_brain
from aura.core.engine import AuraEngine
from aura.core.logging import get_logger

logger = get_logger(__name__)


def build_engine() -> AuraEngine:
    """Create AURA's parts and connect them."""
    brain = create_brain()  # picks the brain based on settings.llm_provider
    return AuraEngine(brain=brain)


def main() -> None:
    print("AURA is online. Type 'exit' or 'quit' to stop.\n")
    logger.info("AURA terminal session starting")
    engine = build_engine()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAURA: Goodbye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("AURA: Goodbye.")
            break

        if not user_input:
            continue

        reply = engine.send(user_input)
        print(f"AURA: {reply}\n")


if __name__ == "__main__":
    main()
