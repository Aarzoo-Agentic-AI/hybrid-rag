"""Application settings.

Values are loaded in this order (later sources win):
1. Defaults defined below
2. ``config.yaml`` (or the file named by ``HYBRID_RAG_CONFIG``)
3. ``.env`` file
4. Environment variables (``HYBRID_RAG_`` prefix, ``__`` between nested levels)

API keys are read from their conventional names (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
``QDRANT_API_KEY``) and never from ``config.yaml``.

LangSmith settings (``LANGSMITH_*``) are read by LangChain directly from the process
environment, so ``get_settings()`` copies them from ``.env`` into ``os.environ``.
"""

from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from hybrid_rag.access import validate_tenant_id

CONFIG_ENV_VAR = "HYBRID_RAG_CONFIG"
DOTENV_PATH = Path(".env")
# Variables read straight from os.environ by LangChain/LangSmith rather than by Settings.
TRACING_ENV_PREFIXES = ("LANGSMITH_", "LANGCHAIN_")
DEFAULT_CONFIG_PATH = Path("config.yaml")

ProviderName = Literal["anthropic", "openai", "fake"]


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "documents"


class EmbeddingSettings(BaseModel):
    # "bge-m3": the real model (about 2.3 GB, downloaded on first use).
    # "hashing": a tiny deterministic stand-in for tests and offline demos; no download,
    # but no real understanding of meaning either.
    provider: Literal["bge-m3", "hashing"] = "bge-m3"
    model: str = "BAAI/bge-m3"
    batch_size: int = Field(default=16, gt=0)
    max_length: int = Field(default=1024, gt=0)  # tokens per chunk; longer text is truncated
    device: str = "cpu"
    cache_size: int = Field(default=4096, ge=0)  # encoded texts kept in memory


class RerankerSettings(BaseModel):
    enabled: bool = True
    # Any sentence-transformers cross-encoder. The default (22M parameters, English) reranks
    # 20 candidates in about 4 s on a laptop CPU. "BAAI/bge-reranker-v2-m3" (568M,
    # multilingual, max_length 1024) is stronger but takes about a minute there; use it on a GPU.
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = Field(default=8, gt=0)  # query-chunk pairs scored at once
    max_length: int = Field(default=512, gt=0)  # tokens per pair; longer pairs are truncated


class DocumentsSettings(BaseModel):
    """Where the policy PDFs are and how they are parsed."""

    path: Path = Path("data/pdfs")  # default folder for scripts/ingest.py
    # The organisation these documents belong to; part of every document ID and access check.
    tenant_id: str = "default"
    registry: Path | None = Path("data/pdfs/registry.yaml")  # policy metadata per file
    # "docling": layout-aware (headings, tables, page headers removed). "pypdf": fast text-only
    # fallback with heading heuristics, used by unit tests.
    parser: Literal["docling", "pypdf"] = "docling"
    cache_dir: Path | None = Path(".cache/parsed")  # parsed documents, keyed by file hash
    # Permission changes made through the API (PUT /documents/{id}/access). Applied on top
    # of the registry by every ingest, so the CLI never undoes them. Git-ignored.
    access_overrides: Path = Path(".data/access_overrides.json")

    @field_validator("tenant_id")
    @classmethod
    def _tenant(cls, value: str) -> str:
        return validate_tenant_id(value)


class ChunkingSettings(BaseModel):
    """Section-aware chunking. Sizes are in tokens of the embedding model's tokenizer."""

    target_tokens: int = Field(default=400, gt=0)  # aim for this when a section is split
    max_tokens: int = Field(default=800, gt=0)  # a section up to this size stays whole
    overlap_tokens: int = Field(default=60, ge=0)  # only between parts of the same section
    # "BAAI/bge-m3" counts with that model's tokenizer; "approx" estimates (4 chars per token).
    tokenizer: str = "BAAI/bge-m3"
    contextual_headers: bool = True  # prepend "Policy: ... / Section: ..." to each chunk
    # Sections whose title matches are not indexed (case-insensitive regular expressions).
    skip_sections: list[str] = Field(
        default_factory=lambda: [r"^contents$", r"^what's inside$", r"acknowledge?ment"]
    )
    # Tables in sections whose title matches become one definition chunk per row.
    definition_sections: list[str] = Field(default_factory=lambda: [r"definitions?", r"glossary"])

    @model_validator(mode="after")
    def _check_sizes(self) -> ChunkingSettings:
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")
        for pattern in [*self.skip_sections, *self.definition_sections]:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from None
        return self


PiiEntity = Literal[
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "EMPLOYEE_ID",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_ITIN",
]


class PrivacySettings(BaseModel):
    """Sensitive-data scanning during ingestion (see hybrid_rag/privacy/README.md)."""

    enabled: bool = True
    # "presidio": patterns + Presidio (spaCy names, validated IDs). "patterns": regex only.
    detector: Literal["presidio", "patterns"] = "presidio"
    spacy_model: str = "en_core_web_sm"
    score_threshold: float = Field(default=0.5, ge=0, le=1)
    # Replaced with a typed placeholder such as <EMAIL>.
    redact: list[PiiEntity] = Field(
        default_factory=lambda: ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "EMPLOYEE_ID"]
    )
    # Never indexed: a document containing any of these is quarantined.
    quarantine: list[PiiEntity] = Field(
        default_factory=lambda: [
            "US_SSN",
            "CREDIT_CARD",
            "IBAN_CODE",
            "US_BANK_NUMBER",
            "US_PASSPORT",
            "US_ITIN",
        ]
    )
    employee_id_pattern: str = r"\bEMP-?\d{5,7}\b"
    # Shared business contacts that answers should keep (exact match, case-insensitive)...
    allowlist: list[str] = Field(default_factory=list)
    # ...or regular expressions, e.g. role mailboxes like "^(hr|payroll)@example\.com$".
    allowlist_patterns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> PrivacySettings:
        overlap = set(self.redact) & set(self.quarantine)
        if overlap:
            raise ValueError(f"entities can't be both redacted and quarantined: {sorted(overlap)}")
        for pattern in [self.employee_id_pattern, *self.allowlist_patterns]:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from None
        return self


RetrievalMode = Literal["dense", "sparse", "hybrid", "hybrid_rerank"]
AnswerStrategy = Literal["single", "agentic"]


class RetrievalSettings(BaseModel):
    # dense / sparse / hybrid return top_k directly; hybrid_rerank fuses fused_limit
    # candidates and the cross-encoder keeps the best top_k.
    mode: RetrievalMode = "hybrid_rerank"
    dense_limit: int = Field(default=50, gt=0)  # hybrid: candidates from dense search
    sparse_limit: int = Field(default=50, gt=0)  # hybrid: candidates from sparse search
    fused_limit: int = Field(default=20, gt=0)  # hybrid_rerank: candidates for the reranker
    top_k: int = Field(default=8, gt=0)  # chunks returned (sent to the LLM)
    # Small-to-big: a hit on one part of a split section is replaced by the whole section,
    # if the section is at most max_section_tokens long.
    expand_sections: bool = True
    max_section_tokens: int = Field(default=1500, gt=0)

    @model_validator(mode="after")
    def _top_k_within_fused(self) -> RetrievalSettings:
        if self.top_k > self.fused_limit:
            raise ValueError("top_k must not exceed fused_limit")
        return self


class ModelTarget(BaseModel):
    provider: ProviderName
    model: str
    # OpenAI reasoning models (gpt-5 family) only: how long the model thinks before answering.
    # Lower is faster and cheaper; answering from given sources needs little reasoning.
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None


class AnswerSettings(BaseModel):
    """Answer generation (step 5)."""

    context_max_tokens: int = Field(default=4000, gt=0)  # numbered sources in the prompt
    # Below this top rerank score, the question is answered "not found" without an LLM call:
    # a cost floor, not a quality filter. The scale belongs to reranker.model: -12 is for
    # ms-marco-MiniLM-L-6-v2, from the golden-set evaluation (eval/). null turns it off.
    min_rerank_score: float | None = -12.0
    max_history_turns: int = Field(default=6, ge=0)  # earlier messages used for query rewriting
    # Shown when the policies don't answer a question.
    contact: str = "the HR Service Center"
    # single: one retrieval pass. agentic: a LangGraph loop that judges whether the sources
    # are enough (llm.agent) and searches again if not (generation/agent.py).
    strategy: AnswerStrategy = "single"
    agent_max_rounds: int = Field(default=2, ge=0)  # extra searches after the first
    agent_max_queries: int = Field(default=2, ge=1)  # new queries per extra search


class ApiSettings(BaseModel):
    """The FastAPI service (step 6)."""

    keys_file: Path = Path("secrets/api_keys.yaml")  # key hashes -> principals; git-ignored
    audit_log: Path = Path("logs/audit.jsonl")  # one JSON line per request; git-ignored
    # Store the (redacted) question text in the audit log. Off: only its hash and length,
    # because HR questions can be personal even after redaction.
    audit_questions: bool = False
    max_upload_mb: int = Field(default=25, gt=0)
    max_files_per_upload: int = Field(default=10, gt=0)
    warmup: bool = True  # load the models at startup, not on the first question
    json_logs: bool = True  # structured logs with request_id
    # User feedback on answers (POST /feedback): one JSON line per rating; git-ignored.
    feedback_log: Path = Path("logs/feedback.jsonl")
    # Feedback is accepted only from the caller who asked, for a recent request.
    feedback_window_hours: float = Field(default=168, gt=0)
    recent_requests: int = Field(default=10_000, gt=0)  # remembered for feedback, in memory


class RuntimeEvalSettings(BaseModel):
    """Runtime evaluation: a faithfulness check on a sample of live answers."""

    # Off by default: each checked answer costs one llm.judge call, in a background thread.
    enabled: bool = False
    sample_rate: float = Field(default=0.1, ge=0, le=1)  # share of answered questions checked
    max_pending: int = Field(default=50, gt=0)  # checks waiting; more are skipped, not queued


class RetrySettings(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    base_delay_s: float = Field(default=0.5, ge=0)
    max_delay_s: float = Field(default=8.0, ge=0)


class CircuitBreakerSettings(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    cooldown_s: float = Field(default=30.0, ge=0)


class LLMSettings(BaseModel):
    answer: list[ModelTarget] = Field(
        default_factory=lambda: [ModelTarget(provider="openai", model="gpt-5")],
        min_length=1,
    )
    query_rewrite: list[ModelTarget] = Field(
        default_factory=lambda: [ModelTarget(provider="openai", model="gpt-5-mini")],
        min_length=1,
    )
    # Agentic retrieval: judges whether the sources are enough and writes new queries.
    agent: list[ModelTarget] = Field(
        default_factory=lambda: [ModelTarget(provider="openai", model="gpt-5-mini")],
        min_length=1,
    )
    # Runtime evaluation's faithfulness check: a small model is enough.
    judge: list[ModelTarget] = Field(
        default_factory=lambda: [ModelTarget(provider="openai", model="gpt-5-mini")],
        min_length=1,
    )
    request_timeout_s: float = Field(default=60.0, gt=0)
    overall_deadline_s: float = Field(default=120.0, gt=0)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HYBRID_RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    documents: DocumentsSettings = Field(default_factory=DocumentsSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    privacy: PrivacySettings = Field(default_factory=PrivacySettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    answer: AnswerSettings = Field(default_factory=AnswerSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    runtime_eval: RuntimeEvalSettings = Field(default_factory=RuntimeEvalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_API_KEY")
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)
        # Earlier sources take priority.
        return (init_settings, env_settings, dotenv_settings, yaml_settings, file_secret_settings)


def export_tracing_env(
    dotenv_path: Path = DOTENV_PATH, environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Copy ``LANGSMITH_*`` / ``LANGCHAIN_*`` values from ``.env`` into ``os.environ``.

    Variables already set in the environment win. Returns the names that were exported.
    ``environ`` exists for tests: LangSmith caches these variables, so tests must not
    change the real environment.
    """
    environ = os.environ if environ is None else environ
    if not dotenv_path.is_file():
        return []
    exported = []
    for name, value in dotenv_values(dotenv_path).items():
        if name.startswith(TRACING_ENV_PREFIXES) and value and name not in environ:
            environ[name] = value
            exported.append(name)
    return exported


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached). Also exports LangSmith settings from .env."""
    export_tracing_env()
    return Settings()
