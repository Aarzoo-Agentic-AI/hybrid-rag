"""``PolicyRetriever``: a LangChain retriever over the chunks one caller may read.

It calls ``DocumentIndex.search`` rather than ``QdrantVectorStore``, because LangChain's
hybrid mode fetches as many candidates from each search as it returns, while we want, for
example, 50 dense + 50 sparse candidates fused down to 20 for the reranker.
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from hybrid_rag.access import Principal
from hybrid_rag.retrieval.filters import RelevanceFilters
from hybrid_rag.store import DocumentIndex, SearchMode


class PolicyRetriever(BaseRetriever):
    """Dense, sparse or hybrid search, always with ``principal``'s security filter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: DocumentIndex
    principal: Principal
    mode: SearchMode = "hybrid"
    k: int = Field(default=6, gt=0)
    dense_limit: int | None = None  # hybrid prefetch sizes (default: k)
    sparse_limit: int | None = None
    filters: RelevanceFilters = Field(default_factory=RelevanceFilters)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.index.search(
            query,
            self.principal,
            k=self.k,
            mode=self.mode,
            conditions=self.filters.conditions(),
            dense_limit=self.dense_limit,
            sparse_limit=self.sparse_limit,
        )
