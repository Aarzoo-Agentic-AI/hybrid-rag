"""Section-aware chunking, contextual headers, IDs and the chunk metadata schema.

How a section becomes chunks:

- A section up to ``max_tokens`` stays **whole**: one chunk.
- A longer section is split near ``target_tokens`` at block boundaries (paragraphs, list
  items, table rows), never inside one. Consecutive parts share up to ``overlap_tokens`` of
  trailing blocks; there is no overlap across sections.
- A table that is split keeps its column headers at the top of every part.
- A block starting with "Unless", "Except", "However", "Provided that", "Note" ... stays with
  the block before it (a rule keeps its exception), and a paragraph ending with ":" stays
  with what follows (a list keeps its introduction).
- Only a single block longer than ``max_tokens`` is split inside, on sentences.
- In definition sections (``definition_sections``), each table row becomes its own chunk
  ("Term: meaning") with ``chunk_type: "definition"``.

Chunk metadata (flat and JSON-friendly, stored as the Qdrant payload):

| Key | Example | Notes |
|---|---|---|
| ``doc_id`` / ``doc_hash`` | ``"5c1f..."`` | Per tenant and source name / SHA-256 of the file |
| ``source`` / ``file_type`` | ``"Travel_Policy.pdf"`` / ``"pdf"`` | File name only, never a path |
| ``title`` | ``"Travel Policy"`` | The policy name |
| ``chunk_id`` / ``chunk_index`` / ``chunk_count`` | | Deterministic UUID / position |
| ``chunk_type`` | ``"section"`` / ``"definition"`` | |
| ``section_id`` | ``"9d2e..."`` | Same for all parts of a section (small-to-big retrieval) |
| ``section_number`` / ``section_title`` | ``"6.2"`` / ``"Merit matrix"`` | |
| ``section_path`` | ``"6 Merit Increase Guidelines > 6.2 Merit matrix"`` | |
| ``section_part`` / ``section_parts`` | ``1`` / ``2`` | 1-based part of a split section |
| ``term`` | ``"Compa-ratio"`` | Definition chunks only |
| ``page`` / ``page_end`` | ``3`` / ``4`` | 1-based; a section can cross pages |
| ``policy_*``, ``version``, ``effective_date`` ... | | From the policy registry |
| ``content_hash`` | ``"77c2..."`` | SHA-256 of ``page_content`` (after redaction) |
| ``pii_scan`` / ``pipeline_hash`` | | Privacy scan result / settings fingerprint |
| ``tenant_id``, ``classification``, ``visibility``, ``allowed_*``, ``access_hash`` | | Access
  rules (``hybrid_rag.access``); never part of the text or the header |
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion.blocks import Block, table_header_lines, table_row_line
from hybrid_rag.ingestion.sections import Section
from hybrid_rag.ingestion.text import sha256_text
from hybrid_rag.ingestion.tokens import TokenCounter
from hybrid_rag.ingestion.types import FileType

# Fixed namespace so IDs are the same on every machine and every run.
ID_NAMESPACE = uuid.UUID("6f1d3c2e-8a4b-4f6e-9c1d-2b7a5e8f0c31")

_EXCEPTION_START = re.compile(
    r"^(?:- )?(?:unless|except|exceptions?|however|provided that|this does not apply|note)\b",
    re.IGNORECASE,
)
_SENTENCE_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def make_doc_id(tenant_id: str, source: str) -> str:
    """Tenant-scoped: two tenants can both have a ``handbook.pdf``. Chunk and section IDs
    derive from it, so they are tenant-scoped too."""
    return str(uuid.uuid5(ID_NAMESPACE, f"doc:{tenant_id}:{source}"))


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"chunk:{doc_id}:{chunk_index}"))


def make_section_id(doc_id: str, path: str) -> str:
    return str(uuid.uuid5(ID_NAMESPACE, f"section:{doc_id}:{path}"))


# ------------------------------------------------------------------ labels


def location_label(metadata: Mapping[str, Any]) -> str:
    """'page 3' or 'pages 3-4'."""
    if "page" not in metadata:
        return ""
    start, end = metadata["page"], metadata.get("page_end", metadata["page"])
    return f"page {start}" if start == end else f"pages {start}-{end}"


def section_label(metadata: Mapping[str, Any]) -> str:
    """'§6.2 Merit matrix', '§3 Anti-Harassment > Prohibited conduct includes', 'Overview'."""
    path = metadata.get("section_path", "")
    number = metadata.get("section_number", "")
    if number:
        return f"§{number} {metadata.get('section_title', '')}".strip()
    if " > " in path:  # an unnumbered sub-section: name its numbered parent
        return f"§{path}" if path[:1].isdigit() else path
    return path or metadata.get("section_title", "")


def citation(metadata: Mapping[str, Any]) -> str:
    """e.g. 'Travel Policy §5 Air travel (Travel_Policy.pdf, page 3)'"""
    name = metadata.get("title") or metadata["source"]
    section = section_label(metadata)
    where = ", ".join(x for x in (metadata["source"], location_label(metadata)) if x)
    return f"{name}{' ' + section if section else ''} ({where})"


def contextual_header(metadata: Mapping[str, Any]) -> str:
    """Prepended to each chunk, so the embedding and the LLM know where it's from."""
    details = [
        metadata.get("policy_id"),
        f"v{metadata['version']}" if metadata.get("version") else None,
        f"effective {metadata['effective_date']}" if metadata.get("effective_date") else None,
        metadata.get("jurisdiction"),
    ]
    details_text = ", ".join(d for d in details if d)
    policy = metadata.get("title") or metadata["source"]
    lines = [f"Policy: {policy}" + (f" ({details_text})" if details_text else "")]
    if metadata.get("section_path"):
        lines.append(f"Section: {metadata['section_path']}")
    return "\n".join(lines)


def chunk_body(text: str, metadata: Mapping[str, Any]) -> str:
    """A chunk's text without its contextual header (if it has one)."""
    header = contextual_header(metadata) + "\n\n"
    return text[len(header) :] if text.startswith(header) else text


# --------------------------------------------------------------- chunking


@dataclass
class Draft:
    """A chunk before IDs, headers and document metadata are added."""

    text: str
    section: Section
    pages: tuple[int, int]
    chunk_type: str = "section"
    part: int = 1
    parts: int = 1
    term: str | None = None


@dataclass
class _Unit:
    """Blocks that must stay together, as rendered text with a token count."""

    text: str
    tokens: int
    pages: tuple[int, int]
    table: int | None = None  # which table a row belongs to
    header: str = ""  # that table's header lines, repeated at the top of each part


class SectionChunker:
    def __init__(self, settings: ChunkingSettings, counter: TokenCounter) -> None:
        self.settings = settings
        self.count = counter.count
        self._definitions = [re.compile(p, re.IGNORECASE) for p in settings.definition_sections]

    def chunk(self, sections: list[Section]) -> list[Draft]:
        drafts: list[Draft] = []
        for section in sections:
            if not section.skipped and section.blocks:
                drafts.extend(self._chunk_section(section))
        return drafts

    def _chunk_section(self, section: Section) -> list[Draft]:
        blocks = section.blocks
        definitions: list[Draft] = []
        if any(p.search(section.title) for p in self._definitions):
            glossaries = [b for b in blocks if b.kind == "table" and _has_two_columns(b)]
            definitions = [d for t in glossaries for d in self._definition_drafts(section, t)]
            if definitions:
                blocks = [b for b in blocks if b not in glossaries]

        units = self._units(blocks)
        if not units:
            return definitions
        if self.count(_render(units)) <= self.settings.max_tokens:
            parts = [units]
        else:
            parts = self._pack(units)
        drafts = [
            Draft(_render(p), section, _span(p), part=i, parts=len(parts))
            for i, p in enumerate(parts, start=1)
        ]
        return drafts + definitions

    # ------------------------------------------------------------ units

    def _units(self, blocks: list[Block]) -> list[_Unit]:
        units: list[_Unit] = []
        glue_next = False
        for index, block in enumerate(blocks):
            if block.kind == "table":
                header = table_header_lines(block.header)
                pages = (block.page, block.page)
                new = [
                    _Unit(line, self.count(line), pages, index, header)
                    for line in map(table_row_line, block.rows)
                ]
                if not new:
                    continue
            else:
                text = block.render()
                new = [_Unit(text, self.count(text), (block.page, block.page))]

            first = new[0]
            exception = first.table is None and _EXCEPTION_START.match(first.text)
            glue = units and (glue_next or exception)
            if glue and units[-1].table is None and first.table is None:
                units[-1] = self._merge(units[-1], first)
                new = new[1:]
            units.extend(new)
            last = units[-1]
            glue_next = last.table is None and last.text.rstrip().endswith(":")
        return units

    def _merge(self, a: _Unit, b: _Unit) -> _Unit:
        tight = a.text.rstrip().endswith(":") or (
            a.text.startswith("- ") and b.text.startswith("- ")
        )
        text = f"{a.text}{chr(10) if tight else chr(10) * 2}{b.text}"
        pages = (min(a.pages[0], b.pages[0]), max(a.pages[1], b.pages[1]))
        return _Unit(text, self.count(text), pages)

    # ---------------------------------------------------------- packing

    def _cost(self, current: list[_Unit], unit: _Unit) -> int:
        header = self.count(unit.header) if _starts_table(current, unit) else 0
        return unit.tokens + header

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        target, limit = self.settings.target_tokens, self.settings.max_tokens
        parts: list[list[_Unit]] = []
        current: list[_Unit] = []
        size = 0
        fresh = False  # True while ``current`` holds only the overlap from the previous part
        for unit in units:
            if unit.tokens > limit:
                if current and not fresh:
                    parts.append(current)
                parts.extend([[piece] for piece in self._split_unit(unit)])
                current, size, fresh = [], 0, False
                continue
            if current and not fresh and size + self._cost(current, unit) > target:
                parts.append(current)
                current = _tail(current, self.settings.overlap_tokens)
                size = sum(u.tokens for u in current)
                fresh = bool(current)
            if current and size + self._cost(current, unit) > limit:
                current, size = [], 0  # the overlap would push this part over the limit
            current.append(unit)
            size += self._cost(current[:-1], unit)
            fresh = False
        if current and not fresh:
            parts.append(current)
        return parts

    def _split_unit(self, unit: _Unit) -> list[_Unit]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.target_tokens,
            chunk_overlap=self.settings.overlap_tokens,
            length_function=self.count,
            separators=_SENTENCE_SEPARATORS,
            keep_separator="end",
        )
        return [_Unit(t, self.count(t), unit.pages) for t in splitter.split_text(unit.text)]

    def _definition_drafts(self, section: Section, table: Block) -> list[Draft]:
        drafts = []
        for row in table.rows:
            term = row[0].strip()
            meaning = " ".join(c.strip() for c in row[1:] if c.strip())
            if term and meaning:
                drafts.append(
                    Draft(
                        f"{term}: {meaning}",
                        section,
                        (table.page, table.page),
                        chunk_type="definition",
                        term=term,
                    )
                )
        return drafts


def _has_two_columns(table: Block) -> bool:
    return bool(table.rows) and all(len(r) >= 2 for r in table.rows)


def _starts_table(current: list[_Unit], unit: _Unit) -> bool:
    return unit.table is not None and (not current or current[-1].table != unit.table)


def _tail(units: list[_Unit], budget: int) -> list[_Unit]:
    """Trailing units that fit in the overlap budget (never the whole part)."""
    tail: list[_Unit] = []
    total = 0
    for unit in reversed(units[1:]):
        if total + unit.tokens > budget:
            break
        tail.insert(0, unit)
        total += unit.tokens
    return tail


def _render(units: list[_Unit]) -> str:
    out: list[str] = []
    prev: _Unit | None = None
    for unit in units:
        if unit.table is not None and (prev is None or prev.table != unit.table):
            if prev is not None:
                out.append("\n\n")
            if unit.header:
                out.append(unit.header + "\n")
        elif prev is not None:
            same_table = unit.table is not None and prev.table == unit.table
            both_items = unit.text.startswith("- ") and prev.text.startswith("- ")
            out.append("\n" if same_table or both_items else "\n\n")
        out.append(unit.text)
        prev = unit
    return "".join(out).strip()


def _span(units: list[_Unit]) -> tuple[int, int]:
    return min(u.pages[0] for u in units), max(u.pages[1] for u in units)


# --------------------------------------------------------------- finalize


def finalize_chunks(
    drafts: list[Draft],
    *,
    doc_id: str,
    doc_hash: str,
    source: str,
    title: str,
    file_type: FileType,
    policy: Mapping[str, Any],
    contextual_headers: bool,
    pii_scan: dict | None = None,
    pipeline_hash: str | None = None,
    access: Mapping[str, Any] | None = None,
) -> list[Document]:
    """Add IDs, hashes, policy, section and access metadata, and headers. Returns LangChain
    Documents. Access fields are added after the header and hash, so they never affect them."""
    final: list[Document] = []
    for index, draft in enumerate(drafts):
        section = draft.section
        metadata: dict[str, Any] = {
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "source": source,
            "title": title,
            "file_type": file_type.value,
            "chunk_id": make_chunk_id(doc_id, index),
            "chunk_index": index,
            "chunk_count": len(drafts),
            "chunk_type": draft.chunk_type,
            "section_id": make_section_id(doc_id, section.path_text),
            "section_number": section.number,
            "section_title": section.title,
            "section_path": section.path_text,
            "section_part": draft.part,
            "section_parts": draft.parts,
            "page": draft.pages[0],
            "page_end": draft.pages[1],
            **policy,
        }
        if draft.term:
            metadata["term"] = draft.term
        body = draft.text
        if contextual_headers:
            body = f"{contextual_header(metadata)}\n\n{body}"
        metadata["content_hash"] = sha256_text(body)
        if pii_scan is not None:
            metadata["pii_scan"] = pii_scan
        if pipeline_hash is not None:
            metadata["pipeline_hash"] = pipeline_hash
        if access is not None:
            metadata.update(access)
        final.append(Document(page_content=body, metadata=metadata, id=metadata["chunk_id"]))
    return final
