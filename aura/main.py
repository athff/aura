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
from qdrant_client import QdrantClient

from aura.brain.factory import create_brain
from aura.config.settings import settings
from aura.core.engine import AuraEngine
from aura.core.logging import get_logger
from aura.interaction.session import Session
from aura.interaction.text import TextInteraction
from aura.memory.embeddings import LocalEmbeddingProvider
from aura.memory.qdrant import QdrantLongTermMemory

logger = get_logger(__name__)


def build_engine() -> AuraEngine:
    """Create AURA's brain and persistent long-term memory."""
    brain = create_brain()

    qdrant = QdrantClient(url=settings.qdrant_url)
    embeddings = LocalEmbeddingProvider()

    long_term_memory = QdrantLongTermMemory(
        client=qdrant,
        embeddings=embeddings,
        collection_name=settings.qdrant_collection,
    )

    logger.info(
        "Long-term memory enabled: Qdrant collection=%s",
        settings.qdrant_collection,
    )

    return AuraEngine(
        brain=brain,
        long_term_memory=long_term_memory,
    )


def main() -> None:
    print("AURA is online. Type 'exit' or 'quit' to stop.\n")
    logger.info("AURA terminal session starting")

    engine = build_engine()

    session = Session(
        interaction=TextInteraction(),
        engine=engine,
    )
    session.run()


if __name__ == "__main__":
    main()
