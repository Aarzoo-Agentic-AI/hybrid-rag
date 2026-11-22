"""What an answer is made of: the text, its citations, and what was used to write it."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# answered:       the LLM answered from the sources (see ``grounded`` for whether it cited them)
# not_found:      the LLM found the sources don't answer the question
# low_confidence: the best source scored below answer.min_rerank_score; no LLM call
# no_sources:     nothing the caller may read matched; no LLM call
AnswerStatus = Literal["answered", "not_found", "low_confidence", "no_sources"]


@dataclass(frozen=True)
class Citation:
    """One source the answer cites, by its number in the prompt."""

    number: int
    label: str  # "Travel Policy §5 Air travel (Travel_Policy.pdf, page 3)"
    title: str
    source: str
    section: str  # section path
    page: int | None
    page_end: int | None
    effective_date: str | None
    policy_id: str | None
    doc_id: str
    chunk_id: str


@dataclass(frozen=True)
class UsedSource:
    """A source that was put in the prompt (cited or not), for the audit log and debugging."""

    number: int
    chunk_id: str
    doc_id: str
    source: str
    access_version: int | None
    retrieval_rank: int | None
    rerank_score: float | None
    cited: bool


@dataclass
class Answer:
    question: str  # as used for retrieval and the prompt: redacted, rewritten if a follow-up
    text: str
    status: AnswerStatus
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = False  # answered, with at least one valid citation
    sources: list[UsedSource] = field(default_factory=list)
    provider: dict[str, Any] | None = None  # FailoverChatModel metadata: who answered, attempts
    redactions: dict[str, int] = field(default_factory=dict)  # removed from question / history
    output_redactions: dict[str, int] = field(default_factory=dict)  # removed from the answer
    rewritten: bool = False  # the question was rewritten from chat history

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
