"""
memory/qdrant.py
----------------
Qdrant-backed long-term semantic memory for AURA.

Embeddings are generated locally; Qdrant stores and searches the vectors.
"""

from uuid import uuid4

from qdrant_client import QdrantClient, models

from aura.memory.embeddings import LocalEmbeddingProvider
from aura.memory.long_term import LongTermMemory


class QdrantLongTermMemory(LongTermMemory):
    """Long-term memory backed by a local Qdrant instance."""

    DEFAULT_COLLECTION = "aura_memory"

    def __init__(
        self,
        client: QdrantClient,
        embeddings: LocalEmbeddingProvider,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self._client = client
        self._embeddings = embeddings
        self._collection_name = collection_name

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if self._client.collection_exists(self._collection_name):
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=self._embeddings.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def remember(self, text: str, metadata: dict | None = None) -> None:
        """Store one text memory in Qdrant."""
        text = str(text).strip()
        if not text:
            return

        vector = self._embeddings.embed(text)

        payload = {
            "text": text,
            "metadata": metadata or {},
        }

        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Return semantically relevant memories."""
        query = str(query).strip()

        if not query or limit < 1:
            return []

        vector = self._embeddings.embed(query)

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=limit,
        ).points

        return [
            {
                "text": result.payload.get("text", ""),
                "metadata": result.payload.get("metadata", {}),
                "score": result.score,
            }
            for result in results
        ]

    def clear(self) -> None:
        """Delete all stored memories while keeping the collection."""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[]
                )
            ),
        )
