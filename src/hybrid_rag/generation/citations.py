"""Inline citations: the model writes ``[1]`` or ``[1, 3]`` after a claim; we link each number
back to its source.

Markers are plain text, not structured output, so they work with every provider behind
``FailoverChatModel`` (it doesn't do tool calling) and while streaming. A marker that names a
source that wasn't in the prompt is removed from the text and reported.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from hybrid_rag.generation.context import Source
from hybrid_rag.generation.types import Citation
from hybrid_rag.ingestion import citation

_MARKER = re.compile(r"\s?\[(\d+(?:\s*,\s*\d+)*)\]")


def _citation(source: Source) -> Citation:
    m = source.doc.metadata
    return Citation(
        number=source.number,
        label=citation(m),
        title=m.get("title", m.get("source", "")),
        source=m.get("source", ""),
        section=m.get("section_path", ""),
        page=m.get("page"),
        page_end=m.get("page_end"),
        effective_date=m.get("effective_date"),
        policy_id=m.get("policy_id"),
        doc_id=m.get("doc_id", ""),
        chunk_id=str(m.get("chunk_id") or source.doc.id),
    )


def parse_citations(text: str, sources: Sequence[Source]) -> tuple[str, list[Citation], list[int]]:
    """The text with invalid markers removed, the cited sources in order of first mention,
    and the invalid numbers."""
    by_number = {s.number: s for s in sources}
    cited: dict[int, Citation] = {}
    invalid: list[int] = []

    def fix(match: re.Match[str]) -> str:
        numbers = [int(n) for n in re.findall(r"\d+", match.group(1))]
        valid = [n for n in numbers if n in by_number]
        invalid.extend(n for n in numbers if n not in by_number)
        for n in valid:
            cited.setdefault(n, _citation(by_number[n]))
        if not valid:
            return ""
        lead = " " if match.group(0).startswith(" ") else ""
        return f"{lead}[{', '.join(map(str, valid))}]"

    clean = _MARKER.sub(fix, text).strip()
    return clean, list(cited.values()), invalid
