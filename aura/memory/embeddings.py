"""
memory/embeddings.py
--------------------
Local text embedding provider for AURA long-term memory.

The embedding model runs locally through sentence-transformers.
No external embedding API is required at runtime.
"""

from sentence_transformers import SentenceTransformer


class LocalEmbeddingProvider:
    """Generate vector embeddings locally."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self.DIMENSION

    def embed(self, text: str) -> list[float]:
        """Convert text into a normalized embedding vector."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()
