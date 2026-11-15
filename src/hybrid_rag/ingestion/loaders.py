"""File loaders for PDF, TXT and CSV.

Two layers:

1. **LangChain loaders** (``PdfPageLoader``, ``TextFileLoader``, ``CsvRowLoader``) subclass
   ``langchain_core.document_loaders.BaseLoader`` and yield raw LangChain ``Document``s: one
   per PDF page, one per TXT file, one per CSV row. They are written here, on top of
   ``pypdf`` and Python's ``csv`` module, because ``langchain-community`` (which had
   ``PyPDFLoader``, ``TextLoader`` and ``CSVLoader``) was sunset in May 2026.
2. **``load_pdf`` / ``load_txt`` / ``load_csv``** clean the text and turn those documents into
   *units* tagged with their location (a page, the whole file, or a group of rows), plus
   warnings and stats. The chunker then splits units into chunks; see ``chunking.py``.

No loader stores the local file path in metadata.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pypdf import PdfReader

from hybrid_rag.ingestion.text import clean_text, detect_encoding
from hybrid_rag.ingestion.types import EmptyDocumentError, IngestionError

# ------------------------------------------------------- LangChain loaders


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


class TextFileLoader(BaseLoader):
    """The whole file as one ``Document``. The encoding is detected if not given."""

    def __init__(self, path: str | Path, encoding: str | None = None) -> None:
        self.path = Path(path)
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        encoding = self.encoding or detect_encoding(self.path)
        text = self.path.read_text(encoding=encoding)
        yield Document(page_content=text, metadata={"encoding": encoding})


class CsvRowLoader(BaseLoader):
    """One ``Document`` per data row, written as ``column: value`` lines.

    Metadata: ``row`` (1-based data row; the header isn't counted) and ``columns``.
    Empty cells are left out. Values in extra, unnamed columns are kept under ``column N``.
    """

    def __init__(self, path: str | Path, encoding: str | None = None) -> None:
        self.path = Path(path)
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        encoding = self.encoding or detect_encoding(self.path)
        try:
            with self.path.open(encoding=encoding, newline="") as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader, [])]
                for row_number, row in enumerate(reader, start=1):
                    lines = []
                    for position, value in enumerate(row):
                        value = clean_text(value)
                        if not value:
                            continue
                        name = header[position] if position < len(header) else ""
                        lines.append(f"{name or f'column {position + 1}'}: {value}")
                    yield Document(
                        page_content="\n".join(lines),
                        metadata={"row": row_number, "columns": [h for h in header if h]},
                    )
        except csv.Error as exc:
            raise IngestionError(f"Could not read CSV {self.path.name}: {exc}") from exc


# ------------------------------------------------- units for the chunker


@dataclass
class LoadedFile:
    units: list[Document]
    title: str
    split_units: bool  # False for CSV: each row group is already one chunk
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
        split_units=True,
        stats={"pages": len(pages), "pages_with_text": len(units)},
        warnings=warnings,
    )


def load_txt(path: Path) -> LoadedFile:
    """The whole file as one unit; the chunker records line numbers."""
    (document,) = TextFileLoader(path).load()
    text = clean_text(document.page_content, collapse_blank_lines=False)
    if not text.strip():
        raise EmptyDocumentError(f"{path.name} is empty.")
    return LoadedFile(
        units=[Document(page_content=text, metadata={})],
        title=path.stem,
        split_units=True,
        stats={"lines": text.count("\n") + 1},
    )


def load_csv(path: Path, rows_per_chunk: int = 1) -> LoadedFile:
    """Rows grouped ``rows_per_chunk`` to a unit. Rows with no values at all are skipped."""
    rows = CsvRowLoader(path).load()
    columns = rows[0].metadata["columns"] if rows else _read_header(path)
    if not columns:
        raise EmptyDocumentError(f"{path.name} has no header row.")

    parsed = [(r.metadata["row"], r.page_content) for r in rows if r.page_content]
    if not parsed:
        raise EmptyDocumentError(f"{path.name} has no data rows.")

    units = []
    for start in range(0, len(parsed), rows_per_chunk):
        group = parsed[start : start + rows_per_chunk]
        if len(group) == 1:
            content = group[0][1]
        else:
            content = "\n\n".join(f"Row {number}\n{text}" for number, text in group)
        units.append(
            Document(
                page_content=content,
                metadata={"row_start": group[0][0], "row_end": group[-1][0], "columns": columns},
            )
        )

    skipped = len(rows) - len(parsed)
    warnings = [f"{skipped} empty row(s) skipped"] if skipped else []
    return LoadedFile(
        units=units,
        title=path.stem,
        split_units=False,
        stats={"rows": len(parsed), "columns": len(columns)},
        warnings=warnings,
    )


def _read_header(path: Path) -> list[str]:
    with path.open(encoding=detect_encoding(path), newline="") as f:
        return [h.strip() for h in next(csv.reader(f), []) if h.strip()]


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
