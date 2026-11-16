"""LangChain wrappers over one shared encoder: ``Embeddings`` (dense) and ``SparseEmbeddings``."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector

from hybrid_rag.embedding.encoders import DenseSparseEncoder


class DenseEmbeddings(Embeddings):
    """Dense vectors from a ``DenseSparseEncoder``, as a LangChain ``Embeddings``."""

    def __init__(self, encoder: DenseSparseEncoder) -> None:
        self.encoder = encoder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.dense for e in self.encoder.encode(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.encoder.encode([text])[0].dense


class SparseEmbeddingsAdapter(SparseEmbeddings):
    """Sparse (lexical) vectors from a ``DenseSparseEncoder``.

    Implements ``langchain_qdrant.SparseEmbeddings``.
    """

    def __init__(self, encoder: DenseSparseEncoder) -> None:
        self.encoder = encoder

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [_to_sparse_vector(e.sparse) for e in self.encoder.encode(list(texts))]

    def embed_query(self, text: str) -> SparseVector:
        return _to_sparse_vector(self.encoder.encode([text])[0].sparse)


def _to_sparse_vector(weights: dict[int, float]) -> SparseVector:
    indices = sorted(weights)
    return SparseVector(indices=indices, values=[weights[i] for i in indices])
