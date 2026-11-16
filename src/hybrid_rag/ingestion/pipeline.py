"""Entry point for ingestion: load a file and turn it into chunks."""

from __future__ import annotations

import logging
from pathlib import Path

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion.chunking import finalize_chunks, make_doc_id, split_units
from hybrid_rag.ingestion.loaders import load_pdf
from hybrid_rag.ingestion.text import sha256_bytes
from hybrid_rag.ingestion.types import FileType, IngestedDocument

logger = logging.getLogger(__name__)


def load_document(
    path: str | Path, settings: ChunkingSettings, *, source: str | None = None
) -> IngestedDocument:
    """Load a PDF file and split it into chunks.

    ``source`` is the name used in citations and for the document ID; it defaults to the
    file name. Pass the original upload name when ``path`` is a temporary file.
    """
    path = Path(path)
    file_type = FileType.from_path(source or path)
    source = source or path.name
    doc_hash = sha256_bytes(path.read_bytes())

    loaded = load_pdf(path)
    chunks = split_units(loaded.units, settings)

    doc_id = make_doc_id(source)
    final = finalize_chunks(
        chunks,
        doc_id=doc_id,
        doc_hash=doc_hash,
        source=source,
        title=loaded.title,
        file_type=file_type,
        contextual_headers=settings.contextual_headers,
    )
    for warning in loaded.warnings:
        logger.warning("%s: %s", source, warning)
    logger.info("Loaded %s: %d chunk(s)", source, len(final))

    return IngestedDocument(
        doc_id=doc_id,
        source=source,
        file_type=file_type,
        doc_hash=doc_hash,
        title=loaded.title,
        chunks=final,
        warnings=loaded.warnings,
        stats={**loaded.stats, "chunks": len(final)},
    )


def find_supported_files(directory: str | Path) -> list[Path]:
    """All supported (PDF) files under ``directory`` (recursively), sorted by path."""
    suffixes = {f".{t.value}" for t in FileType}
    return sorted(
        p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )
