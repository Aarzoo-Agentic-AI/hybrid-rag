"""Turning retrieved chunks into the numbered sources of the prompt.

- **Duplicates** (the same chunk, or the same text twice) are dropped.
- **Order** is kept: best first, as the retriever (and reranker) ranked them.
- **Budget:** sources are added while they fit in ``max_tokens``; one that doesn't fit is
  skipped and smaller ones after it can still go in. The best source always goes in.
- **Labels** carry what a citation needs: policy, section, pages and effective date. The
  contextual header is removed from the text, because the label already says the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from hybrid_rag.ingestion import citation
from hybrid_rag.ingestion.chunking import chunk_body
from hybrid_rag.ingestion.tokens import TokenCounter


@dataclass(frozen=True)
class Source:
    number: int
    doc: Document

    @property
    def label(self) -> str:
        m = self.doc.metadata
        effective = f"; effective {m['effective_date']}" if m.get("effective_date") else ""
        return f"{citation(m)}{effective}"

    @property
    def body(self) -> str:
        return chunk_body(self.doc.page_content, self.doc.metadata).strip()

    def render(self) -> str:
        return f"[{self.number}] {self.label}\n{self.body}"


def select_sources(docs: Sequence[Document], tokens: TokenCounter, max_tokens: int) -> list[Source]:
    sources: list[Source] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    used = 0
    for doc in docs:
        key = str(doc.metadata.get("chunk_id") or doc.id)
        candidate = Source(len(sources) + 1, doc)
        body = candidate.body
        if key in seen_ids or body in seen_text:
            continue
        cost = tokens.count(candidate.render())
        if sources and used + cost > max_tokens:
            continue
        seen_ids.add(key)
        seen_text.add(body)
        sources.append(candidate)
        used += cost
    return sources


def format_sources(sources: Sequence[Source]) -> str:
    return "\n\n".join(s.render() for s in sources)
