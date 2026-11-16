"""Embeddings: dense + sparse vectors from one model, as LangChain interfaces."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.config import EmbeddingSettings
from hybrid_rag.embedding.adapters import DenseEmbeddings, SparseEmbeddingsAdapter
from hybrid_rag.embedding.encoders import (
    BgeM3Encoder,
    CachingEncoder,
    DenseSparseEncoder,
    Encoding,
    HashingEncoder,
)


@dataclass
class EmbeddingModels:
    """The dense and sparse LangChain embeddings, sharing one cached encoder."""

    encoder: CachingEncoder
    dense: DenseEmbeddings
    sparse: SparseEmbeddingsAdapter

    @property
    def dense_dim(self) -> int:
        return self.encoder.dense_dim


def build_embeddings(settings: EmbeddingSettings) -> EmbeddingModels:
    """Create the embeddings from config. Build once per process: the model is large."""
    inner: DenseSparseEncoder
    if settings.provider == "hashing":
        inner = HashingEncoder()
    else:
        inner = BgeM3Encoder(
            model_name=settings.model,
            device=settings.device,
            batch_size=settings.batch_size,
            max_length=settings.max_length,
        )
    encoder = CachingEncoder(inner, max_entries=settings.cache_size)
    return EmbeddingModels(
        encoder=encoder, dense=DenseEmbeddings(encoder), sparse=SparseEmbeddingsAdapter(encoder)
    )


__all__ = [
    "BgeM3Encoder",
    "CachingEncoder",
    "DenseEmbeddings",
    "DenseSparseEncoder",
    "EmbeddingModels",
    "Encoding",
    "HashingEncoder",
    "SparseEmbeddingsAdapter",
    "build_embeddings",
]
