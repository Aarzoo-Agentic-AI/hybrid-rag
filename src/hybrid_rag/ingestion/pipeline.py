"""Entry point for ingestion: load a file and turn it into chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion.chunking import finalize_chunks, make_doc_id, split_units
from hybrid_rag.ingestion.loaders import load_pdf
from hybrid_rag.ingestion.text import sha256_bytes, sha256_text
from hybrid_rag.ingestion.types import FileType, IngestedDocument
from hybrid_rag.tracing import traceable

if TYPE_CHECKING:
    from hybrid_rag.privacy import Sanitizer

logger = logging.getLogger(__name__)

# Bump when loading or chunking logic changes, so every document is re-processed.
PIPELINE_VERSION = 1


def pipeline_hash(settings: ChunkingSettings, sanitizer: Sanitizer) -> str:
    """Fingerprint of everything that shapes a document's chunks besides the file itself:
    the chunking settings, the privacy policy and the pipeline version. Stored on every chunk;
    when it changes, unchanged files are processed again."""
    data = {"v": PIPELINE_VERSION, "chunking": settings.model_dump(), "privacy": sanitizer.policy}
    return sha256_text(json.dumps(data, sort_keys=True))[:12]


@traceable(
    name="load_document",
    run_type="parser",
    process_inputs=lambda i: {"source": i["source"] or Path(i["path"]).name},
    process_outputs=lambda d: (
        {
            "title": d.title,
            **d.stats,  # pages, pages_with_text, chunks
            "warnings": d.warnings,
            "pii": d.pii_scan["status"],
            "redactions": d.pii_scan["redactions"],
        }
        if d
        else {}  # the call raised; LangSmith records the error
    ),
)
def load_document(
    path: str | Path,
    settings: ChunkingSettings,
    *,
    source: str | None = None,
    sanitizer: Sanitizer | None = None,
) -> IngestedDocument:
    """Load a PDF file, remove sensitive data and split it into chunks.

    ``source`` is the name used in citations and for the document ID; it defaults to the
    file name. Pass the original upload name when ``path`` is a temporary file.

    ``sanitizer`` defaults to the pattern-only sanitizer, so nothing is ever chunked unscanned
    by accident; build one from ``settings.privacy`` with ``build_sanitizer()`` for names too.
    Raises ``SensitiveDataError`` (an ``IngestionError``) if the document is quarantined.
    """
    from hybrid_rag.privacy import default_sanitizer

    sanitizer = sanitizer or default_sanitizer()
    path = Path(path)
    file_type = FileType.from_path(source or path)
    source = source or path.name
    doc_hash = sha256_bytes(path.read_bytes())

    loaded = load_pdf(path)
    # Pass 1 on whole pages, before chunking; pass 2 on the finished chunks.
    units, title, report = sanitizer.sanitize_units(loaded.units, title=loaded.title, source=source)
    chunks = split_units(units, settings)

    doc_id = make_doc_id(source)
    fingerprint = pipeline_hash(settings, sanitizer)
    final = finalize_chunks(
        chunks,
        doc_id=doc_id,
        doc_hash=doc_hash,
        source=source,
        title=title,
        file_type=file_type,
        contextual_headers=settings.contextual_headers,
        pii_scan=report.as_metadata(),
        pipeline_hash=fingerprint,
    )
    sanitizer.verify(final, source=source)
    for warning in loaded.warnings:
        logger.warning("%s: %s", source, warning)
    logger.info("Loaded %s: %d chunk(s)", source, len(final))

    return IngestedDocument(
        doc_id=doc_id,
        source=source,
        file_type=file_type,
        doc_hash=doc_hash,
        title=title,
        chunks=final,
        warnings=loaded.warnings,
        stats={**loaded.stats, "chunks": len(final)},
        pii_scan=report.as_metadata(),
        pipeline_hash=fingerprint,
    )


def find_supported_files(directory: str | Path) -> list[Path]:
    """All supported (PDF) files under ``directory`` (recursively), sorted by path."""
    suffixes = {f".{t.value}" for t in FileType}
    return sorted(
        p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )
