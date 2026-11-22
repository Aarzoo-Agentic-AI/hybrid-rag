"""Small-to-big: search on chunks, answer from whole sections.

Small chunks match questions precisely; whole sections give the LLM the full rule, with its
exceptions and table rows. After retrieval (and reranking), a hit on one part of a split
section is replaced by the whole section:

- All parts are fetched with the caller's security filter (``DocumentIndex.section_chunks``).
- They are joined in order, with the contextual header once and the overlap between
  consecutive parts removed.
- Two hits in the same section become one result, at the better hit's position.
- A section longer than ``max_tokens`` is not expanded; the hit is returned as it is.
- Definition chunks (one glossary term each) are never expanded into the whole glossary.

The expanded result keeps the hit's metadata and scores, with ``section_expanded: true``,
``section_chunk_ids`` and the section's full page range.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from hybrid_rag.access import Principal
from hybrid_rag.ingestion.chunking import contextual_header
from hybrid_rag.ingestion.tokens import TokenCounter
from hybrid_rag.store import DocumentIndex
from hybrid_rag.tracing import traceable


def _body(doc: Document) -> str:
    header = contextual_header(doc.metadata) + "\n\n"
    text = doc.page_content
    return text[len(header) :] if text.startswith(header) else text


def _join(previous: str, following: str) -> str:
    """Append ``following``, dropping the lines it repeats from the end of ``previous``."""
    prev_lines, next_lines = previous.split("\n"), following.split("\n")
    for size in range(min(len(prev_lines), len(next_lines)), 0, -1):
        if prev_lines[-size:] == next_lines[:size]:
            next_lines = next_lines[size:]
            break
    rest = "\n".join(next_lines).strip("\n")
    return f"{previous}\n{rest}" if rest else previous


def merge_section(parts: Sequence[Document], hit: Document) -> Document:
    """One document for a whole section, from its parts in order."""
    body = _body(parts[0])
    for part in parts[1:]:
        body = _join(body, _body(part))
    has_header = hit.page_content.startswith(contextual_header(hit.metadata))
    text = f"{contextual_header(hit.metadata)}\n\n{body}" if has_header else body
    metadata = {
        **hit.metadata,
        "section_expanded": True,
        "section_chunk_ids": [p.metadata.get("chunk_id", p.id) for p in parts],
        "page": min(p.metadata.get("page", 1) for p in parts),
        "page_end": max(p.metadata.get("page_end", p.metadata.get("page", 1)) for p in parts),
    }
    return Document(page_content=text, metadata=metadata, id=hit.id)


def _expansion_summary(docs: list[Document] | None) -> dict[str, Any]:
    if docs is None:
        return {}
    return {
        "results": len(docs),
        "expanded": sum(bool(d.metadata.get("section_expanded")) for d in docs),
    }


@traceable(
    name="expand_sections",
    run_type="chain",
    process_inputs=lambda i: {"hits": len(i["docs"]), "max_tokens": i["max_tokens"]},
    process_outputs=_expansion_summary,
)
def expand_sections(
    docs: Sequence[Document],
    index: DocumentIndex,
    principal: Principal,
    *,
    tokens: TokenCounter,
    max_tokens: int,
) -> list[Document]:
    out: list[Document] = []
    seen: set[str] = set()
    for doc in docs:
        m = doc.metadata
        section_id = m.get("section_id")
        if section_id in seen:
            continue  # a better-ranked hit already brought in this section
        if section_id:
            seen.add(section_id)
        if m.get("chunk_type") != "section" or m.get("section_parts", 1) <= 1 or not section_id:
            out.append(doc)
            continue
        parts = index.section_chunks(section_id, principal)
        merged = merge_section(parts, doc) if len(parts) > 1 else doc
        out.append(merged if tokens.count(merged.page_content) <= max_tokens else doc)
    return out


class SectionExpandingRetriever(BaseRetriever):
    """Wraps a retriever and expands its hits to whole sections."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base: BaseRetriever
    index: DocumentIndex
    principal: Principal
    tokens: Any  # a TokenCounter (a Protocol, so pydantic can't check it)
    max_tokens: int = Field(default=1500, gt=0)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        docs = self.base.invoke(query, config={"callbacks": run_manager.get_child()})
        return expand_sections(
            docs, self.index, self.principal, tokens=self.tokens, max_tokens=self.max_tokens
        )
