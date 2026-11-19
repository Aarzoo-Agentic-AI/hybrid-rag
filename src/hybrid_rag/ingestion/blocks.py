"""The parsed form of a PDF: an ordered list of blocks, each tagged with its page.

Both parsers (Docling and pypdf) produce this, so sectioning and chunking don't depend on
which parser was used. It is also what the parse cache stores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BlockKind = Literal["heading", "paragraph", "list_item", "table"]


@dataclass
class Block:
    kind: BlockKind
    text: str = ""  # headings, paragraphs and list items; empty for tables
    page: int = 1
    header: list[str] | None = None  # tables: column headers, if the table has them
    rows: list[list[str]] = field(default_factory=list)  # tables: data rows

    def render(self) -> str:
        if self.kind == "list_item":
            return f"- {self.text}"
        if self.kind == "table":
            return render_table(self.header, self.rows)
        return self.text


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "/")


def table_header_lines(header: list[str] | None) -> str:
    if not header:
        return ""
    return f"| {' | '.join(map(_cell, header))} |\n|{'---|' * len(header)}"


def table_row_line(row: list[str]) -> str:
    return f"| {' | '.join(map(_cell, row))} |"


def render_table(header: list[str] | None, rows: list[list[str]]) -> str:
    lines = [table_header_lines(header)] if header else []
    lines += [table_row_line(r) for r in rows]
    return "\n".join(lines)


@dataclass
class ParsedDocument:
    title: str  # from the PDF's metadata, or "" if it has none
    blocks: list[Block]
    pages: int
    warnings: list[str] = field(default_factory=list)

    @property
    def pages_with_text(self) -> int:
        return len({b.page for b in self.blocks})

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "pages": self.pages,
            "warnings": self.warnings,
            "blocks": [asdict(b) for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedDocument:
        return cls(
            title=data["title"],
            pages=data["pages"],
            warnings=list(data.get("warnings", [])),
            blocks=[Block(**b) for b in data["blocks"]],
        )
