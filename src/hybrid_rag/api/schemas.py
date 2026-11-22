"""Request and response models (they also make the Swagger UI at ``/docs``)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.config import AnswerStrategy, RetrievalMode
from hybrid_rag.generation import Answer, AnswerStatus


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class Filters(BaseModel):
    """Which policies apply. These narrow the search; they never widen what you may read."""

    model_config = ConfigDict(extra="forbid")

    current_only: bool = True
    policy_categories: list[str] | None = Field(default=None, min_length=1)
    jurisdictions: list[str] | None = Field(default=None, min_length=1)
    applies_to: list[str] | None = Field(default=None, min_length=1)
    doc_ids: list[str] | None = Field(default=None, min_length=1)


class QueryRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"question": "How many PTO days do I get after 4 years?"}]},
    )

    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    mode: RetrievalMode | None = None  # default: retrieval.mode
    # single: one search. agentic: search again if the sources aren't enough (slower).
    strategy: AnswerStrategy | None = None  # default: answer.strategy
    filters: Filters | None = None
    debug: bool = False  # include the sources, scores, provider and rewritten question
    stream: bool = False  # server-sent events: token events, then one answer event


class CitationOut(BaseModel):
    number: int
    label: str
    title: str
    source: str
    section: str
    page: int | None
    page_end: int | None
    effective_date: str | None
    policy_id: str | None


class SourceOut(BaseModel):
    number: int
    chunk_id: str
    doc_id: str
    source: str
    access_version: int | None
    retrieval_rank: int | None
    rerank_score: float | None
    cited: bool


class DebugOut(BaseModel):
    question_used: str  # redacted, and rewritten if it was a follow-up
    rewritten: bool
    redactions: dict[str, int]
    output_redactions: dict[str, int]
    provider: dict[str, Any] | None
    strategy: str
    queries: list[str]  # every search run (the agentic strategy can add some)
    sources: list[SourceOut]


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    status: AnswerStatus
    grounded: bool
    citations: list[CitationOut]
    debug: DebugOut | None = None

    @classmethod
    def from_answer(cls, answer: Answer, request_id: str, *, debug: bool) -> QueryResponse:
        return cls(
            request_id=request_id,
            answer=answer.text,
            status=answer.status,
            grounded=answer.grounded,
            citations=[
                CitationOut(**{k: getattr(c, k) for k in CitationOut.model_fields})
                for c in answer.citations
            ],
            debug=DebugOut(
                question_used=answer.question,
                rewritten=answer.rewritten,
                redactions=answer.redactions,
                output_redactions=answer.output_redactions,
                provider=answer.provider,
                strategy=answer.strategy,
                queries=answer.queries,
                sources=[SourceOut(**vars(s)) for s in answer.sources],
            )
            if debug
            else None,
        )


class FeedbackRequest(BaseModel):
    """A rating of one answer, by the caller who asked it."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128, description="From the /query response")
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    request_id: str
    rating: Literal["up", "down"]
    recorded: bool = True


class DocumentOut(BaseModel):
    doc_id: str
    source: str
    title: str
    chunk_count: int
    classification: str
    visibility: str
    access_version: int
    ingested_at: str
    pii_status: str
    # Admins only: who may read it.
    allowed_groups: list[str] | None = None
    allowed_roles: list[str] | None = None
    allowed_users: list[str] | None = None


class AccessOut(BaseModel):
    doc_id: str
    source: str
    access_version: int
    classification: str
    visibility: str
    allowed_groups: list[str]
    allowed_roles: list[str]
    allowed_users: list[str]
    note: str


class FileResultOut(BaseModel):
    source: str
    status: str
    chunks: int
    error: str | None
    warnings: list[str]


class JobOut(BaseModel):
    job_id: str
    status: str
    files: list[FileResultOut]
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None


class ErrorOut(BaseModel):
    error: ErrorBody
