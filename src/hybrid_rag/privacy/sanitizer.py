"""Remove sensitive data from documents before they are chunked, embedded or stored.

Two passes:

1. ``sanitize_units()`` runs on whole pages, before chunking, so an entity is never split
   across a chunk boundary and names are recognised in full sentences. ``redact`` entities
   are replaced with typed placeholders (``<EMAIL>``); a ``quarantine`` entity stops the
   whole document with ``SensitiveDataError``.
2. ``verify()`` re-checks every finished chunk with the pattern detector and refuses to
   continue if anything was missed. It is a safety net; it should never fire.

Found values are never logged, traced, stored or put in error messages: only entity types,
counts and page numbers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

from langchain_core.documents import Document

from hybrid_rag.config import PrivacySettings
from hybrid_rag.ingestion.types import IngestionError
from hybrid_rag.privacy.detectors import (
    Detector,
    Finding,
    PatternDetector,
    PresidioDetector,
    lowercase_vocabulary,
    plausible_name,
)
from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)

# Bump when detection logic changes, so documents are re-scanned on the next ingest.
SCANNER_VERSION = 2

PLACEHOLDERS = {
    "PERSON": "<PERSON>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
    "EMPLOYEE_ID": "<EMPLOYEE_ID>",
}


class SensitiveDataError(IngestionError):
    """The document contains data that must never be indexed. It is quarantined."""

    def __init__(self, source: str, found: Counter[str], pages: list[int], stage: str) -> None:
        self.source, self.found, self.pages, self.stage = source, dict(found), pages, stage
        summary = ", ".join(f"{entity} x{n}" for entity, n in sorted(found.items()))
        where = f" on page(s) {', '.join(map(str, pages))}" if pages else ""
        reason = (
            "high-risk data found"
            if stage == "scan"
            else "sensitive data survived sanitization (please report this as a bug)"
        )
        super().__init__(
            f"{source} quarantined, not indexed: {reason} ({summary}){where}. "
            "Review the document with HR before ingesting it."
        )


@dataclass(frozen=True)
class ScanReport:
    status: Literal["clean", "redacted", "not_scanned"]
    detector: str
    policy: str  # changes whenever the privacy settings or scanner change
    counts: dict[str, int] = field(default_factory=dict)  # redactions by entity type

    def as_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detector": self.detector,
            "policy": self.policy,
            "redactions": dict(self.counts),
        }


class Sanitizer:
    def __init__(
        self,
        settings: PrivacySettings,
        detectors: list[Detector],
        verifier: PatternDetector | None,
    ) -> None:
        self.settings = settings
        self.enabled = settings.enabled
        self.detectors = detectors
        self.verifier = verifier
        self.redact = frozenset(settings.redact)
        self.quarantine = frozenset(settings.quarantine)
        self._allow_exact = {a.strip().lower() for a in settings.allowlist}
        self._allow_patterns = [re.compile(p, re.IGNORECASE) for p in settings.allowlist_patterns]
        self.name = "+".join(d.name for d in detectors) if self.enabled else "disabled"
        self.policy = _policy_hash(settings) if self.enabled else "disabled"

    # --------------------------------------------------------------- pass 1

    @traceable(
        name="sanitize",
        process_inputs=lambda i: {"source": i["source"], "pages": len(i["units"])},
        process_outputs=lambda o: o[2].as_metadata() if o else {},
    )
    def sanitize_units(
        self, units: list[Document], *, title: str, source: str
    ) -> tuple[list[Document], str, ScanReport]:
        """Redact each unit (page) and the title. Raises ``SensitiveDataError`` to quarantine."""
        if not self.enabled:
            return units, title, ScanReport("not_scanned", self.name, self.policy)

        vocabulary = lowercase_vocabulary(u.page_content for u in units)
        redactions: Counter[str] = Counter()
        high_risk: Counter[str] = Counter()
        pages: set[int] = set()

        clean_units = []
        for unit in units:
            text, redacted, risky = self._scan(unit.page_content, vocabulary)
            redactions += redacted
            if risky:
                high_risk += risky
                pages.add(int(unit.metadata.get("page", 0)))
            clean_units.append(Document(page_content=text, metadata=dict(unit.metadata)))
        clean_title, redacted, risky = self._scan(title, vocabulary)
        redactions += redacted
        high_risk += risky

        if high_risk:
            logger.warning("%s quarantined: %s", source, dict(high_risk))
            raise SensitiveDataError(source, high_risk, sorted(p for p in pages if p), "scan")
        status = "redacted" if redactions else "clean"
        if redactions:
            logger.info("%s: redacted %s", source, dict(redactions))
        report = ScanReport(status, self.name, self.policy, dict(sorted(redactions.items())))
        return clean_units, clean_title, report

    def _scan(self, text: str, vocabulary: frozenset[str]) -> tuple[str, Counter, Counter]:
        findings = [
            f
            for f in self._findings(text)
            if not self._allowed(text[f.start : f.end])
            and (f.entity != "PERSON" or plausible_name(text[f.start : f.end], vocabulary))
        ]
        high_risk = Counter(f.entity for f in findings if f.entity in self.quarantine)
        to_redact = _non_overlapping([f for f in findings if f.entity in self.redact])
        redacted = Counter(f.entity for f in to_redact)
        for f in reversed(to_redact):  # right to left keeps earlier offsets valid
            text = text[: f.start] + PLACEHOLDERS[f.entity] + text[f.end :]
        return text, redacted, high_risk

    def _findings(self, text: str) -> list[Finding]:
        """Findings from every detector; one per entity where detectors overlap."""
        wanted = self.redact | self.quarantine
        by_entity: dict[str, list[Finding]] = {}
        for d in self.detectors:
            for f in d.detect(text):
                if f.entity in wanted:
                    by_entity.setdefault(f.entity, []).append(f)
        return [f for found in by_entity.values() for f in _non_overlapping(found)]

    def _allowed(self, value: str) -> bool:
        value = value.strip().lower()
        return value in self._allow_exact or any(p.search(value) for p in self._allow_patterns)

    # --------------------------------------------------------------- pass 2

    def verify(self, chunks: list[Document], *, source: str) -> None:
        """Re-check finished chunks (header included). Raises if anything slipped through."""
        if not self.enabled or self.verifier is None:
            return
        leaked: Counter[str] = Counter()
        pages: set[int] = set()
        for chunk in chunks:
            text = chunk.page_content
            for f in _non_overlapping(self.verifier.detect(text)):
                if not self._allowed(text[f.start : f.end]):
                    leaked[f.entity] += 1
                    pages.add(int(chunk.metadata.get("page", 0)))
        if leaked:
            logger.error("%s: sensitive data survived sanitization: %s", source, dict(leaked))
            raise SensitiveDataError(source, leaked, sorted(p for p in pages if p), "verify")


def _non_overlapping(findings: list[Finding]) -> list[Finding]:
    """Earliest first; at the same start, the longest wins."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda f: (f.start, -(f.end - f.start))):
        if not kept or f.start >= kept[-1].end:
            kept.append(f)
    return kept


def _policy_hash(settings: PrivacySettings) -> str:
    payload = {"scanner": SCANNER_VERSION, **settings.model_dump(exclude={"enabled"})}
    data = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()[:12]


def build_sanitizer(settings: PrivacySettings) -> Sanitizer:
    """Build the sanitizer described by ``settings.privacy`` (Presidio loads on first use)."""
    wanted = set(settings.redact) | set(settings.quarantine)
    patterns = PatternDetector(wanted, settings.employee_id_pattern)
    detectors: list[Detector] = [patterns]
    if settings.detector == "presidio":
        detectors.append(
            PresidioDetector(
                wanted - {"EMPLOYEE_ID"}, settings.spacy_model, settings.score_threshold
            )
        )
    elif "PERSON" in wanted:
        logger.info("privacy.detector is 'patterns', so person names are not detected")
    # The after-chunking check uses patterns only (no PERSON): fast and deterministic.
    verifier = PatternDetector(wanted, settings.employee_id_pattern)
    return Sanitizer(settings, detectors, verifier)


@lru_cache(maxsize=1)
def default_sanitizer() -> Sanitizer:
    """Used when no sanitizer is passed: patterns only, so it's fast and needs no model.

    Production code (``scripts/ingest.py``, the API) builds one from ``settings.privacy``.
    """
    return build_sanitizer(PrivacySettings(detector="patterns"))


def disabled_sanitizer() -> Sanitizer:
    """Skips scanning entirely; chunks are marked ``not_scanned``."""
    return Sanitizer(PrivacySettings(enabled=False), [], None)
