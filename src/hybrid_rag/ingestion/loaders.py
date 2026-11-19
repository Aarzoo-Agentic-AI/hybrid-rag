"""``PdfPageLoader``: a LangChain loader yielding one raw ``Document`` per PDF page.

It subclasses ``langchain_core.document_loaders.BaseLoader`` and is built on ``pypdf``, because
``langchain-community`` (which had ``PyPDFLoader``) was sunset in May 2026. The pypdf parser
(``parsers.PypdfParser``) uses it; the default Docling parser reads the PDF itself.

The loader never stores the local file path in metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader

from hybrid_rag.ingestion.types import IngestionError


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


def compact_ranges(numbers: list[int]) -> str:
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
