"""Cross-encoder reranking (``reranker.model``).

A cross-encoder reads the question and a chunk *together*, so it judges relevance far better
than comparing two separately computed vectors, but it's too slow to run on every chunk. So
hybrid search finds ``fused_limit`` candidates and the reranker keeps the best ``top_k``.

- ``LocalCrossEncoder``: LangChain's ``BaseCrossEncoder`` over any ``sentence-transformers``
  cross-encoder, loaded on first use. The default is ``ms-marco-MiniLM-L-6-v2`` (90 MB, about
  4 s per query on a laptop CPU); ``BAAI/bge-reranker-v2-m3`` (2.3 GB, multilingual) is the
  stronger choice on a GPU, but takes about a minute per query on that CPU.
- ``ScoredReranker``: LangChain's ``CrossEncoderReranker`` that also records each chunk's
  ``rerank_score`` and ``rerank_rank``, and traces the scores (chunk IDs and numbers only,
  never text) as a ``rerank`` run.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import Callbacks
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import ConfigDict

from hybrid_rag.config import RerankerSettings
from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)


class LocalCrossEncoder(BaseCrossEncoder):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker %s on %s", self.model_name, self.device)
                self._model = CrossEncoder(
                    self.model_name, device=self.device, max_length=self.max_length
                )
        return self._model

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        """Relevance of each (query, text) pair: higher is more relevant. The scale depends on
        the model (MiniLM gives raw logits, bge-reranker 0-1), so compare scores only within
        one query."""
        if not text_pairs:
            return []
        scores = self._load().predict(
            list(text_pairs), batch_size=self.batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]


@lru_cache(maxsize=2)
def _cached_reranker(
    model: str, device: str, batch_size: int, max_length: int
) -> LocalCrossEncoder:
    return LocalCrossEncoder(model, device=device, batch_size=batch_size, max_length=max_length)


def get_reranker(settings: RerankerSettings) -> LocalCrossEncoder:
    """One shared reranker per configuration, so the model is loaded once per process."""
    return _cached_reranker(
        settings.model, settings.device, settings.batch_size, settings.max_length
    )


def _rerank_summary(docs: list[Document] | None) -> dict[str, Any]:
    if docs is None:
        return {}
    return {
        "kept": [
            {
                "chunk_id": d.metadata.get("chunk_id"),
                "source": d.metadata.get("source"),
                "section": d.metadata.get("section_number") or d.metadata.get("section_title"),
                "retrieval_rank": d.metadata.get("retrieval_rank"),
                "rerank_rank": d.metadata.get("rerank_rank"),
                "rerank_score": round(d.metadata["rerank_score"], 4),
            }
            for d in docs
        ]
    }


@traceable(
    name="rerank",
    run_type="chain",
    process_inputs=lambda i: {"candidates": len(i["documents"]), "top_n": i["top_n"]},
    process_outputs=_rerank_summary,
)
def rerank(
    model: BaseCrossEncoder, query: str, documents: Sequence[Document], top_n: int
) -> list[Document]:
    """The ``top_n`` documents by cross-encoder score, with ``rerank_score``/``rerank_rank``."""
    scores = model.score([(query, d.page_content) for d in documents])
    ranked = sorted(zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [
        Document(
            page_content=d.page_content,
            metadata={**d.metadata, "rerank_score": score, "rerank_rank": rank},
            id=d.id,
        )
        for rank, (d, score) in enumerate(ranked[:top_n], start=1)
    ]


class ScoredReranker(BaseDocumentCompressor):
    """``CrossEncoderReranker`` that keeps the scores; use with
    ``ContextualCompressionRetriever``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: BaseCrossEncoder
    top_n: int = 6

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        return rerank(self.model, query, documents, self.top_n)
