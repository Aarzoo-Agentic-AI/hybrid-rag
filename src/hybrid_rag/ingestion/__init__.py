"""PDF loading, sectioning and chunking for HR policies. See README.md."""

from hybrid_rag.ingestion.chunking import citation, location_label, make_doc_id, section_label
from hybrid_rag.ingestion.parsers import DoclingParser, PypdfParser
from hybrid_rag.ingestion.pipeline import IngestionPipeline, find_supported_files, load_document
from hybrid_rag.ingestion.registry import PolicyInfo, PolicyRegistry, RegistryError
from hybrid_rag.ingestion.types import (
    EmptyDocumentError,
    FileType,
    IngestedDocument,
    IngestionError,
    UnsupportedFileTypeError,
)

__all__ = [
    "DoclingParser",
    "EmptyDocumentError",
    "FileType",
    "IngestedDocument",
    "IngestionError",
    "IngestionPipeline",
    "PolicyInfo",
    "PolicyRegistry",
    "PypdfParser",
    "RegistryError",
    "UnsupportedFileTypeError",
    "citation",
    "find_supported_files",
    "load_document",
    "location_label",
    "make_doc_id",
    "section_label",
]
