"""Repairs to the parsed blocks, before sanitizing and sectioning.

Layout parsers occasionally read a page in the wrong order. On the HR Policy Handbook, Docling
reads the second column of the two-column contents page *after* the body text of section 2,
so eleven contents entries land in the middle of a sentence. Three steps repair this:

1. **Corrections** from the policy registry: exact find-and-replace fixes for one-off parser
   errors that no rule can recognise safely. A correction that no longer matches anything is
   reported, so stale fixes don't linger after the PDF or the parser changes.
2. **Contents spill:** a run of at least ``MIN_SPILL`` list items that each repeat the title of
   a numbered section of the same document is a misplaced table of contents, and is dropped.
3. **Continuations:** a paragraph that starts in lower case, following a paragraph that ends
   mid-sentence, is the rest of that paragraph (split by a page, column or spill), and is
   joined back onto it.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.ingestion.blocks import Block
from hybrid_rag.ingestion.sections import split_heading

MIN_SPILL = 3  # a shorter run could be a genuine list that mentions other sections

_SENTENCE_END = re.compile(r"[.:!?)\]\"'”’]\s*$")  # noqa: RUF001 (closing quotes)


class Correction(BaseModel):
    """One find-and-replace fix, applied to every heading, paragraph and list item containing
    it (not to table cells)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    find: str = Field(min_length=3)
    replace: str = ""
    reason: str = ""  # why the fix is needed; for the registry's reader, not used in code


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" .").casefold()


def repair_blocks(
    blocks: Sequence[Block], corrections: Iterable[Correction] = ()
) -> tuple[list[Block], list[str]]:
    """The repaired blocks, and warnings (for corrections that matched nothing)."""
    out = [
        Block(
            b.kind, b.text, b.page, list(b.header) if b.header else b.header, [*map(list, b.rows)]
        )
        for b in blocks
    ]
    warnings = []
    for fix in corrections:
        hits = 0
        for block in out:
            if fix.find in block.text:
                hits += block.text.count(fix.find)
                block.text = block.text.replace(fix.find, fix.replace)
        if not hits:
            warnings.append(f"registry correction not applied, text not found: {fix.find!r}")
    out = _drop_contents_spill(out)
    return _join_continuations(out), warnings


def _drop_contents_spill(blocks: list[Block]) -> list[Block]:
    titles = {
        _norm(title)
        for b in blocks
        if b.kind == "heading"
        for number, title in [split_heading(b.text)]
        if number
    }
    spill = [b.kind == "list_item" and _norm(split_heading(b.text)[1]) in titles for b in blocks]
    keep = [True] * len(blocks)
    i = 0
    while i < len(blocks):
        if not spill[i]:
            i += 1
            continue
        j = i
        while j < len(blocks) and spill[j]:
            j += 1
        if j - i >= MIN_SPILL and not _under_contents_heading(blocks, i):
            keep[i:j] = [False] * (j - i)
        i = j
    return [b for b, k in zip(blocks, keep, strict=True) if k]


def _under_contents_heading(blocks: list[Block], index: int) -> bool:
    """True if the run is the real table of contents (skipped later as its own section)."""
    for b in reversed(blocks[:index]):
        if b.kind == "heading":
            return _norm(b.text) in {"contents", "table of contents", "what's inside"}
    return False


def _join_continuations(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        prev = out[-1] if out else None
        if (
            prev is not None
            and block.kind == prev.kind == "paragraph"
            and block.text[:1].islower()
            and prev.text
            and not _SENTENCE_END.search(prev.text)
            and block.page - prev.page <= 1
        ):
            prev.text = f"{prev.text.rstrip()} {block.text.lstrip()}"
            continue
        out.append(block)
    return out
