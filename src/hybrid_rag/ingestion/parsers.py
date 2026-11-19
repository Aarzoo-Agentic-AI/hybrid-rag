"""PDF parsers. Each turns a PDF into a ``ParsedDocument`` (blocks with page numbers).

- ``DoclingParser`` (default): layout-aware. It finds headings, paragraphs, list items and
  tables (with their rows and column headers), and leaves out running page headers and
  footers. About 4 s per page on CPU, so results are cached by file hash.
- ``PypdfParser``: fast, text only. Headings are guessed from numbering ("5. Air travel"),
  repeated page headers and "Page N" lines are removed, and tables come out as plain lines.
  Used by unit tests and as a no-model fallback.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from pypdf import PdfReader

from hybrid_rag.ingestion.blocks import Block, ParsedDocument
from hybrid_rag.ingestion.loaders import PdfPageLoader, compact_ranges
from hybrid_rag.ingestion.text import clean_text
from hybrid_rag.ingestion.types import EmptyDocumentError, IngestionError

logger = logging.getLogger(__name__)

# Bullet markers: hyphen, bullet, middle dot, small square, white bullet, asterisk, en dash.
_BULLET = re.compile("^\\s*[-•·▪◦*–]\\s+")  # noqa: RUF001 (the en dash is a bullet)
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_PAGE_LINE = re.compile(r"^page\s+\d+(?:\s+of\s+\d+)?$", re.IGNORECASE)


class Parser(Protocol):
    name: str  # includes a version; part of the cache key and the pipeline hash

    def parse(self, path: Path) -> ParsedDocument: ...


def pdf_info(path: Path) -> tuple[str, int]:
    """Title (from PDF metadata) and page count. Raises ``IngestionError`` for bad files."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestionError(f"{path.name} is password-protected.")
        title = str((reader.metadata.title if reader.metadata else None) or "").strip()
        return title, len(reader.pages)
    except IngestionError:
        raise
    except Exception as exc:  # pypdf raises many types for damaged files
        raise IngestionError(f"Could not read PDF {path.name}: {exc}") from exc


def _finish(path: Path, title: str, pages: int, blocks: list[Block]) -> ParsedDocument:
    if not blocks:
        raise EmptyDocumentError(
            f"{path.name} has no extractable text. Scanned PDFs need OCR, which isn't supported."
        )
    with_text = {b.page for b in blocks}
    empty = [p for p in range(1, pages + 1) if p not in with_text]
    warnings = []
    if empty:
        warnings.append(
            f"{len(empty)} of {pages} pages had no extractable text and were skipped "
            f"(scanned pages need OCR): {compact_ranges(empty)}"
        )
    return ParsedDocument(title=title, blocks=blocks, pages=pages, warnings=warnings)


def _text_block(text: str, page: int, kind: str = "paragraph") -> Block | None:
    text = clean_text(text, join_hyphenated=True)
    if not text:
        return None
    if kind != "heading" and _BULLET.match(text):
        return Block("list_item", _BULLET.sub("", text, count=1), page)
    return Block(kind, text, page)  # type: ignore[arg-type]


# ------------------------------------------------------------------- Docling


class DoclingParser:
    """Docling without OCR (the policies are born-digital), with accurate table structure."""

    VERSION = 1
    name = f"docling-v{VERSION}"

    def __init__(self) -> None:
        self._converter: Any = None
        self._lock = threading.Lock()

    def _get_converter(self) -> Any:
        with self._lock:
            if self._converter is None:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    PdfPipelineOptions,
                    TableFormerMode,
                )
                from docling.document_converter import DocumentConverter, PdfFormatOption

                options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
                options.table_structure_options.mode = TableFormerMode.ACCURATE
                self._converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
                )
                logger.info("Docling converter ready (first use downloads its layout models)")
            return self._converter

    def parse(self, path: Path) -> ParsedDocument:
        from docling_core.types.doc import DocItemLabel, TableItem

        title, pages = pdf_info(path)
        try:
            document = self._get_converter().convert(str(path)).document
        except Exception as exc:
            raise IngestionError(f"Could not parse PDF {path.name}: {exc}") from exc

        text_labels = {
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.CAPTION,
            DocItemLabel.CODE,
            DocItemLabel.FORMULA,
        }
        heading_labels = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
        blocks: list[Block] = []
        page = 1
        # Body items only, in reading order; page headers and footers are not in the body.
        for item, _level in document.iterate_items(with_groups=False):
            page = item.prov[0].page_no if getattr(item, "prov", None) else page
            if isinstance(item, TableItem):
                block = _docling_table(item, page)
            elif item.label in heading_labels:
                block = _text_block(item.text, page, "heading")
            elif item.label == DocItemLabel.LIST_ITEM:
                text = clean_text(item.text, join_hyphenated=True)
                block = Block("list_item", _BULLET.sub("", text), page) if text else None
            elif item.label in text_labels:
                block = _text_block(item.text, page)
            else:
                block = None  # pictures, checkboxes, ...
            if block is not None:
                blocks.append(block)
        return _finish(path, title, pages, blocks)


def _docling_table(item: Any, page: int) -> Block | None:
    grid = [[clean_text(cell.text) for cell in row] for row in item.data.grid]
    flags = [all(cell.column_header for cell in row) for row in item.data.grid]
    grid_rows = [r for r in grid if any(r)]
    if not grid_rows:
        return None
    header_rows = [r for r, is_header in zip(grid, flags, strict=True) if is_header and any(r)]
    data_rows = [r for r, is_header in zip(grid, flags, strict=True) if not is_header and any(r)]
    header = header_rows[0] if header_rows else None
    return Block("table", page=page, header=header, rows=data_rows)


# --------------------------------------------------------------------- pypdf


class PypdfParser:
    """Text-only parsing with heading heuristics. Fast and model-free."""

    VERSION = 1
    name = f"pypdf-v{VERSION}"

    def parse(self, path: Path) -> ParsedDocument:
        title, _ = pdf_info(path)
        pages = PdfPageLoader(path).load()
        page_lines = [
            [ln.strip() for ln in clean_text(p.page_content, join_hyphenated=True).split("\n")]
            for p in pages
        ]
        repeated = _repeated_lines(page_lines)

        blocks: list[Block] = []
        for number, lines in enumerate(page_lines, start=1):
            current: Block | None = None
            for line in lines:
                if not line or line in repeated or _PAGE_LINE.match(line):
                    current = None
                    continue
                if _looks_like_heading(line):
                    blocks.append(Block("heading", line, number))
                    current = None
                elif _BULLET.match(line):
                    current = Block("list_item", _BULLET.sub("", line, count=1), number)
                    blocks.append(current)
                elif current is not None and not current.text.endswith((".", "!", "?", ":")):
                    current.text += " " + line  # a wrapped line continues the block
                else:
                    current = Block("paragraph", line, number)
                    blocks.append(current)
        return _finish(path, title, len(pages), blocks)


def _looks_like_heading(line: str) -> bool:
    return (
        bool(_NUMBERED_HEADING.match(line))
        and len(line) <= 80
        and not line.endswith((".", ",", ";", ":"))
    )


def _repeated_lines(page_lines: list[list[str]]) -> set[str]:
    """Lines on at least half the pages (and at least two): running headers and footers."""
    if len(page_lines) < 2:
        return set()
    counts = Counter(line for lines in page_lines for line in set(lines) if line)
    threshold = max(2, math.ceil(len(page_lines) / 2))
    return {line for line, n in counts.items() if n >= threshold}
