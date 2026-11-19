"""Real Docling and the real bge-m3 tokenizer. Run with: uv run pytest -m integration

The first run downloads Docling's layout models (a few hundred MB). Parsing takes about 4 s
per page on CPU.
"""

from pathlib import Path

import pytest

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion import DoclingParser, IngestionPipeline, PolicyRegistry
from hybrid_rag.ingestion.tokens import get_token_counter
from tests.parse_fixtures import DoclingFixtureParser

pytestmark = pytest.mark.integration

PDFS = Path(__file__).resolve().parents[1] / "data" / "pdfs"


def test_docling_output_matches_the_saved_fixture() -> None:
    """If this fails, Docling's output changed: re-run scripts/make_parse_fixtures.py."""
    pdf = PDFS / "Travel_Policy.pdf"

    live = DoclingParser().parse(pdf)
    saved = DoclingFixtureParser().parse(pdf)

    assert live.to_dict() == saved.to_dict()


def test_chunks_fit_the_limit_in_real_bge_m3_tokens() -> None:
    chunking = ChunkingSettings()  # the production settings, counted with bge-m3's tokenizer
    pipeline = IngestionPipeline(
        chunking,
        parser=DoclingFixtureParser(),
        registry=PolicyRegistry.load(PDFS / "registry.yaml"),
    )
    count = get_token_counter(chunking.tokenizer).count

    for pdf in sorted(PDFS.glob("*.pdf")):
        for chunk in pipeline.load(pdf).chunks:
            body = chunk.page_content.split("\n\n", 1)[1]
            assert count(body) <= chunking.max_tokens, (pdf.name, chunk.metadata["section_path"])
