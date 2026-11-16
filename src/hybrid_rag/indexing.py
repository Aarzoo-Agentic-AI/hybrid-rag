"""Load files and store them in the index: the glue between ingestion and the vector store."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion import IngestionError, find_supported_files, load_document
from hybrid_rag.store import DocumentIndex, IndexResult
from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)


@dataclass
class IngestOutcome:
    path: Path
    result: IndexResult | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def _outcome_summary(o: IngestOutcome) -> dict[str, object]:
    r = o.result
    return {
        "source": r.source if r else o.path.name,
        "status": r.status if r else "failed",
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
    chunking: ChunkingSettings,
    *,
    source: str | None = None,
    force: bool = False,
) -> IngestOutcome:
    """Load, chunk and index one file. File problems are reported in the outcome, not raised."""
    path = Path(path)
    try:
        document = load_document(path, chunking, source=source)
    except IngestionError as exc:
        logger.warning("Skipping %s: %s", path.name, exc)
        return IngestOutcome(path, error=str(exc))
    result = index.index(document, force=force)
    return IngestOutcome(path, result=result, warnings=document.warnings)


def expand_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Files are kept as given; directories are replaced by the supported files inside them."""
    expanded: list[Path] = []
    for p in map(Path, paths):
        expanded.extend(find_supported_files(p) if p.is_dir() else [p])
    return expanded


def _ingest_summary(outcomes: list[IngestOutcome]) -> dict[str, object]:
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
    chunking: ChunkingSettings,
    *,
    force: bool = False,
) -> list[IngestOutcome]:
    """Ingest files and folders. Traced as one ``ingest`` run with a child run per file."""
    return [ingest_file(p, index, chunking, force=force) for p in expand_paths(paths)]
