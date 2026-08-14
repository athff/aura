"""AURA memory layer: a provider-independent conversation memory foundation.

Exposes the abstract ``Memory`` contract and the concrete in-memory
``ConversationMemory`` implementation. No database, embeddings, or external
services are used here -- this is deliberately minimal so richer memory backends
(durable storage, RAG, etc.) can be added later by implementing ``Memory``.
"""

from aura.memory.base import Memory
from aura.memory.conversation import ConversationMemory

__all__ = ["Memory", "ConversationMemory"]