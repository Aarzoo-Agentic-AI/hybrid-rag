"""Split loaded units into chunks and attach the chunk metadata schema.

Chunk metadata (flat and JSON-friendly, so it can be stored as a Qdrant payload):

| Key | Example | Notes |
|---|---|---|
| ``doc_id`` | ``"5c1f..."`` | Same for every chunk of a file; stable for a given source name |
| ``doc_hash`` | ``"9a0b..."`` | SHA-256 of the file bytes |
| ``source`` | ``"handbook.pdf"`` | File name only, never the local path |
| ``title`` | ``"Employee Handbook"`` | PDF title, otherwise the file name without extension |
| ``file_type`` | ``"pdf"`` | ``pdf``, ``txt`` or ``csv`` |
| ``chunk_id`` | ``"0e7d..."`` | UUID, deterministic from ``doc_id`` and ``chunk_index`` |
| ``chunk_index`` / ``chunk_count`` | ``0`` / ``12`` | Position within the file |
| ``content_hash`` | ``"77c2..."`` | SHA-256 of ``page_content`` |
| ``page`` (+ ``page_label``) | ``3`` | PDF only, 1-based |
| ``line_start`` / ``line_end`` | ``40`` / ``58`` | TXT only, 1-based |
| ``row_start`` / ``row_end``, ``columns`` | ``5`` / ``5`` | CSV only, 1-based data rows |
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion.text import sha256_text
from hybrid_rag.ingestion.types import FileType

# Fixed namespace so IDs are the same on every machine and every run.
ID_NAMESPACE = uuid.UUID("6f1d3c2e-8a4b-4f6e-9c1d-2b7a5e8f0c31")

# Paragraphs, then lines, then sentences, then words.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def make_doc_id(source: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"doc:{source}"))


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"chunk:{doc_id}:{chunk_index}"))


def location_label(metadata: Mapping[str, Any]) -> str:
    """Human-readable location of a chunk, for headers and citations."""
    if "page" in metadata:
        label = metadata.get("page_label")
        return f"page {metadata['page']}" + (f" (labelled {label})" if label else "")
    if "row_start" in metadata:
        start, end = metadata["row_start"], metadata["row_end"]
        return f"row {start}" if start == end else f"rows {start}-{end}"
    if "line_start" in metadata:
        start, end = metadata["line_start"], metadata["line_end"]
        return f"line {start}" if start == end else f"lines {start}-{end}"
    return ""


def citation(metadata: Mapping[str, Any]) -> str:
    """e.g. 'handbook.pdf, page 3'"""
    location = location_label(metadata)
    return f"{metadata['source']}, {location}" if location else str(metadata["source"])


def contextual_header(source: str, title: str, metadata: Mapping[str, Any]) -> str:
    """A short header prepended to each chunk, so the embedding and the LLM know where it's from."""
    stem = source.rsplit(".", 1)[0]
    document = source if title == stem else f"{title} ({source})"
    location = location_label(metadata)
    return f"Document: {document}" + (f"\nLocation: {location}" if location else "")


def split_units(units: list[Document], settings: ChunkingSettings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
        keep_separator="end",  # keep "." at the end of a sentence, not the start of the next
        add_start_index=True,
        strip_whitespace=True,
    )
    return splitter.split_documents(units)


def add_line_numbers(chunks: list[Document], full_text: str) -> None:
    """Turn the splitter's character offsets into 1-based line numbers (TXT files)."""
    for chunk in chunks:
        start = chunk.metadata.pop("start_index")
        line_start = full_text.count("\n", 0, start) + 1
        chunk.metadata["line_start"] = line_start
        chunk.metadata["line_end"] = line_start + chunk.page_content.count("\n")


def finalize_chunks(
    chunks: list[Document],
    *,
    doc_id: str,
    doc_hash: str,
    source: str,
    title: str,
    file_type: FileType,
    contextual_headers: bool,
) -> list[Document]:
    """Add headers, IDs and hashes; returns new Documents in the final schema."""
    final: list[Document] = []
    for index, chunk in enumerate(chunks):
        location = {k: v for k, v in chunk.metadata.items() if k != "start_index"}
        body = chunk.page_content
        if contextual_headers:
            body = f"{contextual_header(source, title, location)}\n\n{body}"
        metadata = {
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "source": source,
            "title": title,
            "file_type": file_type.value,
            "chunk_id": make_chunk_id(doc_id, index),
            "chunk_index": index,
            "chunk_count": len(chunks),
            "content_hash": sha256_text(body),
            **location,
        }
        final.append(Document(page_content=body, metadata=metadata, id=metadata["chunk_id"]))
    return final
