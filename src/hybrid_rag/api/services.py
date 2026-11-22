"""Everything the routes need, built once at startup (``build_services``)."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybrid_rag.access import AccessOverrides
from hybrid_rag.api.audit import AuditLog
from hybrid_rag.api.auth import KeyStore
from hybrid_rag.api.jobs import IngestJobs
from hybrid_rag.config import Settings
from hybrid_rag.generation import AnswerService
from hybrid_rag.ingestion import IngestionPipeline, PolicyRegistry
from hybrid_rag.ingestion.parsers import DoclingParser, Parser, PypdfParser
from hybrid_rag.llm import LLMFactory
from hybrid_rag.privacy import Sanitizer
from hybrid_rag.store import DocumentIndex, build_document_index
from hybrid_rag.tracing import anonymizing_client, tracing_is_enabled

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(".data/uploads")


@dataclass
class Services:
    settings: Settings
    index: DocumentIndex
    answers: AnswerService
    keys: KeyStore
    audit: AuditLog
    jobs: IngestJobs
    overrides: AccessOverrides
    sanitizer: Sanitizer
    parser: Parser
    llm_factory: LLMFactory | None = None
    trace_client: Any = None  # a LangSmith Client with the anonymizer, when tracing is on
    _pipelines: dict[str, IngestionPipeline] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def pipeline(self, tenant_id: str) -> IngestionPipeline:
        """One ingestion pipeline per tenant. The policy registry belongs to
        ``documents.tenant_id``; other tenants get no registry (deny by default) until
        permissions are set with ``PUT /documents/{id}/access``."""
        with self._lock:
            if tenant_id not in self._pipelines:
                own = tenant_id == self.settings.documents.tenant_id
                self._pipelines[tenant_id] = IngestionPipeline.from_settings(
                    self.settings,
                    tenant_id=tenant_id,
                    registry=None if own else PolicyRegistry.empty(),
                    parser=self.parser,
                    sanitizer=self.sanitizer,
                )
                self._pipelines[tenant_id].access_overrides = self.overrides
            return self._pipelines[tenant_id]

    def warm_up(self) -> None:
        """Load the embedding model, the reranker and Presidio now, not on the first question."""
        logger.info("Loading models")
        self.index.embeddings.encoder.encode(["warm up"])
        self.sanitizer.redact_text("Warm up for Jane Doe")
        if self.answers.cross_encoder is not None:
            self.answers.cross_encoder.score([("warm up", "warm up")])
        elif self.settings.reranker.enabled:
            from hybrid_rag.retrieval import get_reranker

            get_reranker(self.settings.reranker).score([("warm up", "warm up")])
        logger.info("Models loaded")


def build_services(settings: Settings) -> Services:
    from hybrid_rag.privacy import build_sanitizer

    index = build_document_index(settings)
    factory = LLMFactory(settings)
    sanitizer = build_sanitizer(settings.privacy)
    answers = AnswerService(
        settings,
        index,
        factory.for_task("answer"),
        rewrite_llm=factory.for_task("query_rewrite"),
        sanitizer=sanitizer,
    )
    keys = KeyStore.load(settings.api.keys_file)
    if not len(keys):
        logger.warning(
            "No API keys in %s: every request will be refused. Create one with "
            "`docker compose run --rm keys --demo` or scripts/make_api_key.py",
            settings.api.keys_file,
        )
    trace_client = None
    if tracing_is_enabled():
        trace_client = anonymizing_client(lambda text: sanitizer.check_output(text)[0])
    return Services(
        settings=settings,
        index=index,
        answers=answers,
        keys=keys,
        audit=AuditLog(settings.api.audit_log, include_questions=settings.api.audit_questions),
        jobs=IngestJobs(UPLOAD_DIR),
        overrides=AccessOverrides(settings.documents.access_overrides),
        sanitizer=sanitizer,
        parser=DoclingParser() if settings.documents.parser == "docling" else PypdfParser(),
        llm_factory=factory,
        trace_client=trace_client,
    )
