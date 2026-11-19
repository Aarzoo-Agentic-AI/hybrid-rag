"""A parser that returns Docling's saved output for the HR policy PDFs (tests/fixtures/parsed).

Regenerate the fixtures with ``uv run python scripts/make_parse_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_rag.ingestion.blocks import ParsedDocument
from hybrid_rag.ingestion.parsers import DoclingParser
from hybrid_rag.ingestion.text import sha256_bytes

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "parsed"


class DoclingFixtureParser:
    """Same name as ``DoclingParser``, so pipeline fingerprints match production."""

    name = DoclingParser.name

    def parse(self, path: Path) -> ParsedDocument:
        data = json.loads((FIXTURES / f"{path.stem}.json").read_text(encoding="utf-8"))
        if data["doc_hash"] != sha256_bytes(path.read_bytes()) or data["parser"] != self.name:
            raise AssertionError(
                f"{path.name} or the Docling parser changed since its fixture was made: "
                "run scripts/make_parse_fixtures.py"
            )
        return ParsedDocument.from_dict(data)
