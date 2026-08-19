from qdrant_client import QdrantClient

from aura.memory.embeddings import LocalEmbeddingProvider
from aura.memory.qdrant import QdrantLongTermMemory


COLLECTION = "aura_test_memory_unit"


def make_memory() -> tuple[QdrantClient, QdrantLongTermMemory]:
    client = QdrantClient(url="http://localhost:6333")
    embeddings = LocalEmbeddingProvider()
    memory = QdrantLongTermMemory(
        client=client,
        embeddings=embeddings,
        collection_name=COLLECTION,
    )
    return client, memory


def cleanup(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)


def test_remember_and_search() -> None:
    client, memory = make_memory()
    try:
        memory.remember("AURA is my personal AI assistant")
        memory.remember("My favorite programming language is Python")

        results = memory.search("What is AURA?", limit=1)

        assert results
        assert results[0]["text"] == "AURA is my personal AI assistant"
        assert results[0]["score"] > 0
    finally:
        cleanup(client)


def test_metadata_is_preserved() -> None:
    client, memory = make_memory()
    try:
        memory.remember(
            "I am building long-term memory for AURA",
            metadata={"source": "test"},
        )

        results = memory.search("long-term memory", limit=1)

        assert results
        assert results[0]["metadata"]["source"] == "test"
    finally:
        cleanup(client)


def test_empty_memory_is_ignored() -> None:
    client, memory = make_memory()
    try:
        memory.remember("   ")
        assert memory.search("anything", limit=5) == []
    finally:
        cleanup(client)


def test_invalid_search_limit_returns_empty() -> None:
    client, memory = make_memory()
    try:
        memory.remember("AURA is my personal AI assistant")

        assert memory.search("AURA", limit=0) == []
        assert memory.search("AURA", limit=-1) == []
    finally:
        cleanup(client)


def test_clear_removes_memories() -> None:
    client, memory = make_memory()
    try:
        memory.remember("AURA is my personal AI assistant")

        assert memory.search("AURA", limit=1)

        memory.clear()

        assert memory.search("AURA", limit=1) == []
    finally:
        cleanup(client)
