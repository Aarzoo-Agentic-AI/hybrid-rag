"""The audit log: one JSON line per request (``api.audit_log``, git-ignored).

It answers "who asked what, and which policy text did they get". For a question it records the
caller, the answer's status and provider, every chunk put in the prompt with its
``access_version`` (so you can tell which permissions it was retrieved under) and whether it
was cited, and the redaction counts. The question itself is stored only as a hash and a length,
unless ``api.audit_questions`` is on (then its redacted text too).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hybrid_rag.generation import Answer

logger = logging.getLogger(__name__)


class AuditLog:
    def __init__(self, path: Path | None, *, include_questions: bool = False) -> None:
        self.path = path
        self.include_questions = include_questions
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(
            {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **record},
            sort_keys=True,
            default=str,
        )
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:
            logger.exception("Couldn't write the audit log")  # never fail the request on it

    def question_fields(self, question: str) -> dict[str, Any]:
        """``question`` is the redacted (and rewritten) question from the ``Answer``."""
        fields: dict[str, Any] = {
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "question_chars": len(question),
        }
        if self.include_questions:
            fields["question"] = question
        return fields

    def answer_fields(self, answer: Answer) -> dict[str, Any]:
        provider = answer.provider or {}
        return {
            **self.question_fields(answer.question),
            "answer_status": answer.status,
            "grounded": answer.grounded,
            "rewritten": answer.rewritten,
            "provider": provider.get("provider"),
            "model": provider.get("model"),
            "failed_over": provider.get("failed_over"),
            "redactions": answer.redactions,
            "output_redactions": answer.output_redactions,
            "chunks": [
                {
                    "chunk_id": s.chunk_id,
                    "doc_id": s.doc_id,
                    "source": s.source,
                    "access_version": s.access_version,
                    "cited": s.cited,
                }
                for s in answer.sources
            ],
        }
