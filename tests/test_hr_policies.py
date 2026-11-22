"""The full pipeline on the 5 HR policy PDFs, using Docling's saved parse (no models needed)."""

from pathlib import Path

import pytest

from hybrid_rag.config import ChunkingSettings, PrivacySettings
from hybrid_rag.ingestion import IngestionPipeline, PolicyRegistry, citation
from hybrid_rag.ingestion.tokens import ApproxTokenCounter
from hybrid_rag.privacy import build_sanitizer
from tests.parse_fixtures import DoclingFixtureParser

PDFS = Path(__file__).resolve().parents[1] / "data" / "pdfs"
CHUNKING = ChunkingSettings(tokenizer="approx")


@pytest.fixture(scope="module")
def docs():
    pipeline = IngestionPipeline(
        CHUNKING,
        parser=DoclingFixtureParser(),
        sanitizer=build_sanitizer(PrivacySettings(allowlist=["hr@dummymocktest.example"])),
        registry=PolicyRegistry.load(PDFS / "registry.yaml"),
        tenant_id="dummymocktest",
    )
    return {pdf.name: pipeline.load(pdf) for pdf in sorted(PDFS.glob("*.pdf"))}


def chunks(docs, name: str):
    return docs[name].chunks


def section(docs, name: str, path: str):
    return [c for c in chunks(docs, name) if c.metadata["section_path"] == path]


def test_every_policy_loads_clean_with_registry_metadata(docs) -> None:
    assert len(docs) == 5
    for name, doc in docs.items():
        assert not doc.warnings, (name, doc.warnings)
        assert doc.pii_scan["status"] == "clean", (name, doc.pii_scan["redactions"])
        for c in doc.chunks:
            assert c.metadata["policy_category"] in {
                "handbook",
                "benefits",
                "travel",
                "expenses",
                "compensation",
            }
            assert (c.metadata["jurisdiction"], c.metadata["effective_date"]) == (
                "US",
                "2027-01-01",
            )


def test_access_rules_from_the_registry_are_on_every_chunk(docs) -> None:
    for name, doc in docs.items():
        restricted = name == "Annual_Review_Merit_Bonus_Policy_2027.pdf"
        for c in doc.chunks:
            m = c.metadata
            assert m["tenant_id"] == "dummymocktest"
            assert (m["classification"], m["visibility"]) == (
                ("confidential", "restricted") if restricted else ("internal", "tenant")
            )
            assert m["access_hash"] == doc.access.hash
            assert "people_manager" not in c.page_content  # never in the text or header


def test_handbook_contents_spill_is_repaired(docs) -> None:
    (eeo,) = section(docs, "HR_Policy_Handbook_2027.pdf", "2 Equal Employment Opportunity (EEO)")

    assert "veteran or military status, marital status, or any other characteristic" in (
        eeo.page_content
    )
    for entry in ("Travel and expenses", "Corrective action", "Acknowledgment form"):
        assert entry not in eeo.page_content


def test_policy_names_come_from_the_registry(docs) -> None:
    # The Expense Policy's title is an image: no text to read it from.
    assert docs["Expense_Policy.pdf"].title == "Expense Policy"
    assert chunks(docs, "Expense_Policy.pdf")[0].page_content.startswith(
        "Policy: Expense Policy (HR-EXP-004, effective 2027-01-01, US)\nSection: Overview"
    )


def test_tables_keep_their_rows_and_headers(docs) -> None:
    (air,) = section(docs, "Travel_Policy.pdf", "5 Air travel")

    assert (
        "| Flight length | Cabin class | Notes |\n|---|---|---|\n"
        "| 6 hours or less | Economy | Lowest logical fare; avoid unnecessary connections. |"
    ) in air.page_content
    assert citation(air.metadata) == "Travel Policy §5 Air travel (Travel_Policy.pdf, page 3)"


def test_nested_sections_have_full_paths(docs) -> None:
    paths = {c.metadata["section_path"] for d in docs.values() for c in d.chunks}

    assert (
        "7 Annual Performance Bonus Plan > 7.3 Company Performance Factor (bonus pool funding)"
        in paths
    )
    assert "3 Dental and Vision > Dental" in paths
    assert "3 Anti-Harassment and Anti-Discrimination > Prohibited conduct includes" in paths
    assert {"Overview", "Document Control"} <= paths


def test_glossaries_become_definition_chunks(docs) -> None:
    merit = [c for c in chunks(docs, "Annual_Review_Merit_Bonus_Policy_2027.pdf")
             if c.metadata["chunk_type"] == "definition"]  # fmt: skip
    benefits = [c for c in chunks(docs, "Employee_Benefits_Guide_2027.pdf")
                if c.metadata["chunk_type"] == "definition"]  # fmt: skip

    assert len(merit) == 9 and len(benefits) == 8
    ratio = next(c for c in merit if c.metadata["term"] == "Compa-ratio")
    assert ratio.page_content.endswith(
        "Compa-ratio: Base salary divided by the midpoint of the salary range"
    )
    assert ratio.metadata["section_path"] == "12 Definitions"


def test_boilerplate_is_not_indexed(docs) -> None:
    text = "\n".join(c.page_content for d in docs.values() for c in d.chunks)

    assert "Internal use" not in text  # running page header
    assert "I agree to follow it" not in text  # acknowledgement forms
    assert "Signature:" not in text
    paths = {c.metadata["section_path"] for d in docs.values() for c in d.chunks}
    assert not any(p in {"Contents", "What's inside"} for p in paths)


def test_chunks_respect_the_size_limit(docs) -> None:
    count = ApproxTokenCounter().count
    for doc in docs.values():
        for c in doc.chunks:
            body = c.page_content.split("\n\n", 1)[1]
            assert count(body) <= CHUNKING.max_tokens, (doc.source, c.metadata["section_path"])


def test_sections_crossing_pages_record_both_pages(docs) -> None:
    (ground,) = section(docs, "Travel_Policy.pdf", "7 Ground transportation")

    assert (ground.metadata["page"], ground.metadata["page_end"]) == (3, 4)
