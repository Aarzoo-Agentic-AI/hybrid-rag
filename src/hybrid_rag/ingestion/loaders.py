"""PDF loading.

Two layers:

1. **``PdfPageLoader``**, a LangChain loader: it subclasses
   ``langchain_core.document_loaders.BaseLoader`` and yields one raw ``Document`` per page.
   It is written here, on top of ``pypdf``, because ``langchain-community`` (which had
   ``PyPDFLoader``) was sunset in May 2026.
2. **``load_pdf``** cleans the text and turns those pages into *units* tagged with their page
   number, plus warnings and stats. The chunker then splits units into chunks; see
   ``chunking.py``.

The loader never stores the local file path in metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader

from hybrid_rag.ingestion.text import clean_text
from hybrid_rag.ingestion.types import EmptyDocumentError, IngestionError


class PdfPageLoader(BaseLoader):
    """One ``Document`` per page, with ``page`` (1-based), ``page_label`` and ``title``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def lazy_load(self) -> Iterator[Document]:
        try:
            reader = PdfReader(self.path)
            if reader.is_encrypted and not reader.decrypt(""):
                raise IngestionError(f"{self.path.name} is password-protected.")
            title = str((reader.metadata.title if reader.metadata else None) or "").strip()
            labels = reader.page_labels
            for index, page in enumerate(reader.pages):
                yield Document(
                    page_content=page.extract_text() or "",
                    metadata={
                        "page": index + 1,
                        "page_label": labels[index] if index < len(labels) else str(index + 1),
                        "title": title,
                    },
                )
        except IngestionError:
            raise
        except Exception as exc:  # pypdf raises many types for damaged files
            raise IngestionError(f"Could not read PDF {self.path.name}: {exc}") from exc


@dataclass
class LoadedFile:
    units: list[Document]
    title: str
    stats: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def load_pdf(path: Path) -> LoadedFile:
    """One unit per page that has text. Page numbers are 1-based."""
    pages = PdfPageLoader(path).load()

    units: list[Document] = []
    empty_pages: list[int] = []
    for page in pages:
        number = page.metadata["page"]
        text = clean_text(page.page_content, join_hyphenated=True)
        if not text:
            empty_pages.append(number)
            continue
        metadata: dict[str, object] = {"page": number}
        if page.metadata["page_label"] != str(number):
            metadata["page_label"] = page.metadata["page_label"]  # e.g. "iv" in a preface
        units.append(Document(page_content=text, metadata=metadata))

    if not units:
        raise EmptyDocumentError(
            f"{path.name} has no extractable text. Scanned PDFs need OCR, which isn't supported."
        )

    warnings = []
    if empty_pages:
        warnings.append(
            f"{len(empty_pages)} of {len(pages)} pages had no extractable text and were skipped "
            f"(scanned pages need OCR): {_compact_ranges(empty_pages)}"
        )

    title = pages[0].metadata["title"] if pages else ""
    return LoadedFile(
        units=units,
        title=title or path.stem,
        stats={"pages": len(pages), "pages_with_text": len(units)},
        warnings=warnings,
    )


def _compact_ranges(numbers: list[int]) -> str:
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10'"""
    parts: list[str] = []
    start = prev = numbers[0]
    for n in [*numbers[1:], None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)
