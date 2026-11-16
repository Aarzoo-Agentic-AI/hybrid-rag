"""Shared types for ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from langchain_core.documents import Document


class FileType(StrEnum):
    """Supported file types. PDF is the only data source."""

    PDF = "pdf"

    @classmethod
    def from_path(cls, path: str | Path) -> FileType:
        suffix = Path(path).suffix.lower().lstrip(".")
        try:
            return cls(suffix)
        except ValueError:
            supported = ", ".join(f".{t.value}" for t in cls)
            raise UnsupportedFileTypeError(
                f"Unsupported file type {Path(path).suffix or '(none)'!r} for {Path(path).name}. "
                f"Supported: {supported}"
            ) from None


class IngestionError(Exception):
    """Base class for ingestion errors."""


class UnsupportedFileTypeError(IngestionError):
    """The file extension isn't one of the supported types."""


class EmptyDocumentError(IngestionError):
    """The file contains no text that can be indexed."""


@dataclass
class IngestedDocument:
    """A loaded, chunked file, ready to be embedded and stored."""

    doc_id: str  # stable for a given source name, so re-ingesting a file replaces it
    source: str  # file name, used in citations
    file_type: FileType
    doc_hash: str  # SHA-256 of the file bytes, used to skip unchanged files
    title: str
    chunks: list[Document]
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)  # pages, pages_with_text, chunks
    # Result of the sensitive-data scan: status, detector, policy and redaction counts.
    pii_scan: dict = field(default_factory=dict)
    pipeline_hash: str = ""  # chunking settings + privacy policy; see pipeline_hash()
