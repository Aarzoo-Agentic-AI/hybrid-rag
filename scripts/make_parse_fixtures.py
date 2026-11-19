"""Save Docling's parse of the HR policy PDFs as test fixtures (tests/fixtures/parsed/).

Unit tests then exercise real Docling output (headings, tables, page numbers) without running
Docling, which needs its layout models and takes about 4 s per page. Re-run this script when
a PDF in data/pdfs changes or DoclingParser.VERSION is bumped; a unit test fails otherwise.

Usage: uv run python scripts/make_parse_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid_rag.ingestion.parsers import DoclingParser  # noqa: E402
from hybrid_rag.ingestion.text import sha256_bytes  # noqa: E402

PDFS = ROOT / "data" / "pdfs"
OUT = ROOT / "tests" / "fixtures" / "parsed"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parser = DoclingParser()
    for pdf in sorted(PDFS.glob("*.pdf")):
        parsed = parser.parse(pdf)
        data = {
            "source": pdf.name,
            "doc_hash": sha256_bytes(pdf.read_bytes()),
            "parser": parser.name,
            **parsed.to_dict(),
        }
        target = OUT / f"{pdf.stem}.json"
        target.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)} ({len(parsed.blocks)} blocks)")


if __name__ == "__main__":
    main()
