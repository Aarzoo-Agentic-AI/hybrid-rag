"""Group blocks into sections with a heading path.

Example path: "6 Merit Increase Guidelines > 6.2 Merit matrix".

Levels come from the numbering: "6." is level 1 and "6.2" is level 2. An unnumbered heading
("Dental", "Glossary") sits one level below the last numbered heading, or at level 1 before
section 1 (front matter such as "Document Control"). Everything before the first real
heading (the cover: title, company name, policy number, effective date) becomes an
"Overview" section, and headings inside it are kept as text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from hybrid_rag.ingestion.blocks import Block

_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.+)$")
OVERVIEW = "Overview"


@dataclass
class Section:
    number: str  # "6.2", or "" for unnumbered headings and the overview
    title: str  # "Merit matrix"
    level: int
    path: list[str]  # labels from the top: ["6 Merit Increase Guidelines", "6.2 Merit matrix"]
    blocks: list[Block] = field(default_factory=list)
    skipped: bool = False  # matched chunking.skip_sections (or is inside such a section)

    @property
    def label(self) -> str:
        return f"{self.number} {self.title}" if self.number else self.title

    @property
    def path_text(self) -> str:
        return " > ".join(self.path)

    @property
    def pages(self) -> tuple[int, int]:
        pages = [b.page for b in self.blocks]
        return (min(pages), max(pages)) if pages else (0, 0)


def split_heading(text: str) -> tuple[str, str]:
    """'6.2 Merit matrix' -> ('6.2', 'Merit matrix'); 'Glossary' -> ('', 'Glossary')."""
    match = _NUMBER.match(text.strip())
    return (match.group(1), match.group(2).strip()) if match else ("", text.strip())


def build_sections(blocks: Iterable[Block], skip_patterns: Iterable[str] = ()) -> list[Section]:
    skip = [re.compile(p, re.IGNORECASE) for p in skip_patterns]
    sections: list[Section] = []
    overview = Section("", OVERVIEW, 1, [OVERVIEW])
    current = overview
    stack: list[Section] = []  # open headings, outermost first
    last_numbered_depth = 0
    in_front = True

    for block in blocks:
        if block.kind != "heading":
            current.blocks.append(block)
            continue
        number, title = split_heading(block.text)
        if in_front and not number and block.page == 1:
            # Cover headings (title, company name) are part of the overview text.
            current.blocks.append(Block("paragraph", block.text, block.page))
            continue
        in_front = False
        if number:
            level = number.count(".") + 1
            last_numbered_depth = level
        else:
            level = last_numbered_depth + 1 if last_numbered_depth else 1
        while stack and stack[-1].level >= level:
            stack.pop()
        parent_skipped = any(s.skipped for s in stack)
        section = Section(
            number=number,
            title=title,
            level=level,
            path=[*(s.label for s in stack), f"{number} {title}" if number else title],
            skipped=parent_skipped or any(p.search(title) for p in skip),
        )
        stack.append(section)
        sections.append(section)
        current = section

    return ([overview] if overview.blocks else []) + sections
