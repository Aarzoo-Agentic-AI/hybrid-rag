"""Vector store (Qdrant) for hybrid search."""

from __future__ import annotations

from qdrant_client import QdrantClient

from hybrid_rag.config import Settings
from hybrid_rag.embedding import EmbeddingModels, build_embeddings
from hybrid_rag.store.qdrant import (
    DENSE_VECTOR,
    PAYLOAD_INDEXES,
    SPARSE_VECTOR,
    CollectionConfigError,
    DocumentIndex,
    DocumentInfo,
    IndexResult,
    StoreError,
    ensure_collection,
)


def create_client(settings: Settings) -> QdrantClient:
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    return QdrantClient(url=settings.qdrant.url, api_key=api_key, timeout=30)


def build_document_index(
    settings: Settings,
    embeddings: EmbeddingModels | None = None,
    client: QdrantClient | None = None,
) -> DocumentIndex:
    """Connect to Qdrant and return the ``DocumentIndex`` for the configured collection."""
    return DocumentIndex(
        client=client or create_client(settings),
        collection=settings.qdrant.collection,
        embeddings=embeddings or build_embeddings(settings.embedding),
        batch_size=settings.embedding.batch_size,
    )


__all__ = [
    "DENSE_VECTOR",
    "PAYLOAD_INDEXES",
    "SPARSE_VECTOR",
    "CollectionConfigError",
    "DocumentIndex",
    "DocumentInfo",
    "IndexResult",
    "StoreError",
    "build_document_index",
    "create_client",
    "ensure_collection",
]
