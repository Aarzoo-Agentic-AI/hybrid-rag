"""Load files and store them in the index: the glue between ingestion and the vector store."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from hybrid_rag.ingestion import (
    IngestionError,
    IngestionPipeline,
    find_supported_files,
    make_doc_id,
)
from hybrid_rag.ingestion.text import sha256_bytes
from hybrid_rag.privacy import SensitiveDataError
from hybrid_rag.store import DocumentIndex, IndexResult
from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)


@dataclass
class IngestOutcome:
    path: Path
    result: IndexResult | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    quarantined: bool = False  # sensitive data found; the file was not indexed

    @property
    def ok(self) -> bool:
        return self.error is None


def _outcome_summary(o: IngestOutcome | None) -> dict[str, object]:
    if o is None:  # the call raised; LangSmith records the error
        return {}
    r = o.result
    return {
        "source": r.source if r else o.path.name,
        "status": r.status if r else ("quarantined" if o.quarantined else "failed"),
        "chunks": r.chunks if r else 0,
        "removed": r.removed if r else 0,
        "error": o.error,
        "warnings": o.warnings,
    }


@traceable(
    name="ingest_file",
    process_inputs=lambda i: {"source": i["source"] or Path(i["path"]).name, "force": i["force"]},
    process_outputs=_outcome_summary,
)
def ingest_file(
    path: str | Path,
    index: DocumentIndex,
    pipeline: IngestionPipeline,
    *,
    source: str | None = None,
    force: bool = False,
) -> IngestOutcome:
    """Load, sanitize, chunk and index one file.

    File problems, including quarantined files, are reported in the outcome, not raised.
    """
    path = Path(path)
    if not force and (unchanged := _unchanged(path, source, index, pipeline)):
        return IngestOutcome(path, result=unchanged)
    try:
        document = pipeline.load(path, source=source)
    except IngestionError as exc:
        logger.warning("Skipping %s: %s", path.name, exc)
        return IngestOutcome(path, error=str(exc), quarantined=isinstance(exc, SensitiveDataError))
    result = index.index(document, force=force)
    return IngestOutcome(path, result=result, warnings=document.warnings)


def _unchanged(
    path: Path,
    source: str | None,
    index: DocumentIndex,
    pipeline: IngestionPipeline,
) -> IndexResult | None:
    """Skip a file that is already indexed exactly as it would be now, without loading it."""
    try:
        doc_hash = sha256_bytes(path.read_bytes())
    except OSError:
        return None  # let load_document report the problem
    source = source or path.name
    doc_id = make_doc_id(source)
    existing = index.current_version(doc_id, doc_hash, pipeline.fingerprint(source))
    if existing is None:
        return None
    logger.info("%s unchanged, skipped", source)
    return IndexResult(doc_id, source, "unchanged", existing.chunk_count)


def expand_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Files are kept as given; directories are replaced by the supported files inside them."""
    expanded: list[Path] = []
    for p in map(Path, paths):
        expanded.extend(find_supported_files(p) if p.is_dir() else [p])
    return expanded


def _ingest_summary(outcomes: list[IngestOutcome] | None) -> dict[str, object]:
    if outcomes is None:
        return {}
    statuses = [_outcome_summary(o)["status"] for o in outcomes]
    return {"files": len(outcomes), **{s: statuses.count(s) for s in sorted(set(statuses))}}


@traceable(
    name="ingest",
    tags=["ingestion"],
    process_inputs=lambda i: {
        # File names only (never local paths), and never consume an iterator here.
        "paths": [Path(p).name for p in i["paths"]]
        if isinstance(i["paths"], list | tuple)
        else "(iterable)",
        "force": i["force"],
    },
    process_outputs=_ingest_summary,
)
def ingest_paths(
    paths: Iterable[str | Path],
    index: DocumentIndex,
    pipeline: IngestionPipeline,
    *,
    force: bool = False,
) -> list[IngestOutcome]:
    """Ingest files and folders. Traced as one ``ingest`` run with a child run per file."""
    return [ingest_file(p, index, pipeline, force=force) for p in expand_paths(paths)]
