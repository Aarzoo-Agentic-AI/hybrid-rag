"""Retrieval: dense, sparse and hybrid search, reranking and small-to-big. See README.md.

``build_retriever`` returns a LangChain retriever for one caller:

| ``retrieval.mode`` | Pipeline |
|---|---|
| ``dense`` / ``sparse`` / ``hybrid`` | ``PolicyRetriever`` returns ``top_k`` chunks |
| ``hybrid_rerank`` | hybrid (``fused_limit`` candidates), then ``ScoredReranker`` (``top_k``) |

then, with ``expand_sections``, ``SectionExpandingRetriever`` swaps hits on split sections for
the whole section. Every stage runs with the caller's security filter.
"""

from __future__ import annotations

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.retrievers import BaseRetriever

from hybrid_rag.access import Principal
from hybrid_rag.config import RetrievalMode, Settings
from hybrid_rag.ingestion.tokens import TokenCounter, get_token_counter
from hybrid_rag.retrieval.expansion import SectionExpandingRetriever, expand_sections
from hybrid_rag.retrieval.filters import RelevanceFilters
from hybrid_rag.retrieval.reranker import LocalCrossEncoder, ScoredReranker, get_reranker, rerank
from hybrid_rag.retrieval.retriever import PolicyRetriever
from hybrid_rag.store import DocumentIndex

MODES: tuple[RetrievalMode, ...] = ("dense", "sparse", "hybrid", "hybrid_rerank")


def build_retriever(
    settings: Settings,
    index: DocumentIndex,
    principal: Principal,
    *,
    mode: RetrievalMode | None = None,
    filters: RelevanceFilters | None = None,
    cross_encoder: BaseCrossEncoder | None = None,
    tokens: TokenCounter | None = None,
    expand: bool | None = None,
) -> BaseRetriever:
    """The retriever for ``principal``. Arguments override ``settings.retrieval``.

    ``cross_encoder`` defaults to the shared ``reranker.model``; tests pass a fake.
    """
    r = settings.retrieval
    mode = mode or r.mode
    filters = filters or RelevanceFilters()
    common = {
        "index": index,
        "principal": principal,
        "filters": filters,
        "dense_limit": r.dense_limit,
        "sparse_limit": r.sparse_limit,
    }
    retriever: BaseRetriever
    if mode == "hybrid_rerank":
        if cross_encoder is None:
            if not settings.reranker.enabled:
                raise ValueError(
                    "retrieval.mode is hybrid_rerank but reranker.enabled is false; "
                    "enable the reranker or choose another mode"
                )
            cross_encoder = get_reranker(settings.reranker)
        retriever = ContextualCompressionRetriever(
            base_compressor=ScoredReranker(model=cross_encoder, top_n=r.top_k),
            base_retriever=PolicyRetriever(mode="hybrid", k=r.fused_limit, **common),
        )
    elif mode in ("dense", "sparse", "hybrid"):
        retriever = PolicyRetriever(mode=mode, k=r.top_k, **common)
    else:
        raise ValueError(f"Unknown retrieval mode {mode!r}; use one of {', '.join(MODES)}")

    if r.expand_sections if expand is None else expand:
        retriever = SectionExpandingRetriever(
            base=retriever,
            index=index,
            principal=principal,
            tokens=tokens or get_token_counter(settings.chunking.tokenizer),
            max_tokens=r.max_section_tokens,
        )
    return retriever


__all__ = [
    "MODES",
    "LocalCrossEncoder",
    "PolicyRetriever",
    "RelevanceFilters",
    "ScoredReranker",
    "SectionExpandingRetriever",
    "build_retriever",
    "expand_sections",
    "get_reranker",
    "rerank",
]
