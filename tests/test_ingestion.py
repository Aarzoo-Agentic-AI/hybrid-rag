import json
from itertools import pairwise
from pathlib import Path

import pytest

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion import (
    EmptyDocumentError,
    FileType,
    IngestionError,
    IngestionPipeline,
    PolicyRegistry,
    PypdfParser,
    RegistryError,
    UnsupportedFileTypeError,
    citation,
    find_supported_files,
    load_document,
)
from hybrid_rag.ingestion.blocks import Block, ParsedDocument
from hybrid_rag.ingestion.chunking import SectionChunker
from hybrid_rag.ingestion.sections import build_sections
from hybrid_rag.ingestion.text import clean_text
from hybrid_rag.ingestion.tokens import ApproxTokenCounter
from tests.pdf_fixture import write_pdf

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
PDFS = ROOT / "data" / "pdfs"
COUNT = ApproxTokenCounter().count


def settings(**overrides) -> ChunkingSettings:
    base = {"target_tokens": 60, "max_tokens": 120, "overlap_tokens": 12, "tokenizer": "approx"}
    return ChunkingSettings(**{**base, **overrides})


def body(chunk) -> str:
    """Chunk text without the contextual header."""
    return chunk.page_content.split("\n\n", 1)[1]


def para(text: str, page: int = 1) -> Block:
    return Block("paragraph", text, page)


def heading(text: str, page: int = 2) -> Block:
    return Block("heading", text, page)


def item(text: str, page: int = 2) -> Block:
    return Block("list_item", text, page)


def chunk_blocks(blocks: list[Block], **overrides):
    s = settings(**overrides)
    return SectionChunker(s, ApproxTokenCounter()).chunk(build_sections(blocks, s.skip_sections))


# -------------------------------------------------------------- clean_text


def test_clean_text_normalises_whitespace() -> None:
    assert clean_text("a\r\nb\t\tc  d \n\n\n\ne\x00") == "a\nb c d\n\ne"


def test_clean_text_joins_hyphenated_line_breaks_when_asked() -> None:
    assert clean_text("hy-\nbrid", join_hyphenated=True) == "hybrid"
    assert clean_text("hy-\nbrid") == "hy-\nbrid"


# ------------------------------------------------------------ pypdf parser


def test_pypdf_parser_finds_headings_paragraphs_and_list_items(tmp_path: Path) -> None:
    pdf = write_pdf(
        tmp_path / "p.pdf",
        [
            ["Travel Policy", "Effective January 1, 2027."],
            [
                "1. Purpose",
                "This policy sets the standards for travel and",
                "applies to all employees.",
                "2. Air travel",
                "- Economy for flights under 6 hours.",
                "- Premium economy for longer flights.",
            ],
        ],
        title="Travel Policy",
    )

    parsed = PypdfParser().parse(pdf)

    kinds = [(b.kind, b.text, b.page) for b in parsed.blocks]
    assert kinds == [
        ("paragraph", "Travel Policy Effective January 1, 2027.", 1),
        ("heading", "1. Purpose", 2),
        ("paragraph", "This policy sets the standards for travel and applies to all employees.", 2),
        ("heading", "2. Air travel", 2),
        ("list_item", "Economy for flights under 6 hours.", 2),
        ("list_item", "Premium economy for longer flights.", 2),
    ]
    assert parsed.title == "Travel Policy"


def test_pypdf_parser_removes_running_headers_and_page_numbers(tmp_path: Path) -> None:
    running = "Acme Inc. | Travel Policy | Internal use"
    pages = [
        [running, f"Page {n}", f"{n}. Section {n}", f"Text of section {n}."] for n in (1, 2, 3)
    ]
    parsed = PypdfParser().parse(write_pdf(tmp_path / "p.pdf", pages))

    texts = [b.text for b in parsed.blocks]
    assert running not in texts
    assert not any(t.startswith("Page ") for t in texts)
    assert texts == ["1. Section 1", "Text of section 1.", "2. Section 2", "Text of section 2.",
                     "3. Section 3", "Text of section 3."]  # fmt: skip


def test_pages_without_text_are_skipped_with_a_warning(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "scan.pdf", [["Text page."], [], [], ["Another."]])

    doc = load_document(pdf, settings())

    (chunk,) = doc.chunks  # no headings: one section, spanning the pages that have text
    assert (chunk.metadata["page"], chunk.metadata["page_end"]) == (1, 4)
    assert body(chunk) == "Text page.\n\nAnother."
    assert doc.warnings and "2-3" in doc.warnings[0]


def test_pdf_with_no_text_at_all_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(EmptyDocumentError, match="OCR"):
        load_document(write_pdf(tmp_path / "blank.pdf", [[], []]), settings())


def test_damaged_pdf_raises_ingestion_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 this is not really a pdf")

    with pytest.raises(IngestionError, match=r"broken\.pdf"):
        load_document(broken, settings())


def test_pypdf_parser_on_a_real_policy_finds_its_sections() -> None:
    doc = load_document(PDFS / "Travel_Policy.pdf", settings(max_tokens=800, target_tokens=400))

    paths = {c.metadata["section_path"] for c in doc.chunks}
    assert {"1 Purpose", "5 Air travel", "6 Lodging", "14 Compliance and exceptions"} <= paths
    text = "\n".join(c.page_content for c in doc.chunks)
    assert "Internal use" not in text  # the running header on every page
    # (pypdf only recognises numbered headings, so unnumbered ones such as the acknowledgement
    # form are kept as text; Docling finds them, see tests/test_ingestion_live.py.)


# ---------------------------------------------------------------- sections


def test_section_levels_and_paths_come_from_numbering() -> None:
    blocks = [
        Block("heading", "Travel Policy", 1),  # cover heading: part of the overview
        para("Policy No. HR-TRV-003"),
        heading("Document Control"),
        para("Owner: People team"),
        heading("6. Merit Increase Guidelines"),
        heading("6.1 Merit budget"),
        para("The budget is 3.5%."),
        heading("Special cases"),
        para("New hires are prorated."),
        heading("7. Bonus"),
        para("Paid in March."),
    ]

    sections = build_sections(blocks)

    assert [(s.path_text, s.level) for s in sections] == [
        ("Overview", 1),
        ("Document Control", 1),
        ("6 Merit Increase Guidelines", 1),
        ("6 Merit Increase Guidelines > 6.1 Merit budget", 2),
        ("6 Merit Increase Guidelines > 6.1 Merit budget > Special cases", 3),
        ("7 Bonus", 1),
    ]
    assert [b.text for b in sections[0].blocks] == ["Travel Policy", "Policy No. HR-TRV-003"]


def test_skipped_sections_and_their_children_are_not_chunked() -> None:
    blocks = [
        heading("Contents"),
        item("1. Purpose"),
        heading("1. Purpose"),
        para("Why this policy exists."),
        heading("Employee Acknowledgment"),
        para("I agree to follow this policy."),
        heading("9. Appendix"),
        para("Forms."),
        heading("9.1 Signature form"),
        para("Name: ____"),
        heading("10. Contacts"),
        para("HR service center."),
    ]

    drafts = chunk_blocks(blocks, skip_sections=["^contents$", "acknowledgment", "^appendix$"])

    assert [d.section.path_text for d in drafts] == ["1 Purpose", "10 Contacts"]


# ----------------------------------------------------------------- chunker


def test_a_section_that_fits_stays_whole() -> None:
    blocks = [heading("1. Purpose"), para("First paragraph."), para("Second paragraph.")]

    (draft,) = chunk_blocks(blocks)

    assert draft.text == "First paragraph.\n\nSecond paragraph."
    assert (draft.part, draft.parts) == (1, 1)


def test_long_sections_split_at_block_boundaries_with_overlap_inside_the_section() -> None:
    sentences = [para(f"Rule {i}: claims need a receipt and manager approval.") for i in range(12)]
    blocks = [heading("4. Claims"), *sentences, heading("5. Other"), para("Short section.")]

    drafts = chunk_blocks(blocks, overlap_tokens=20)  # each paragraph is about 13 tokens

    claims = [d for d in drafts if d.section.number == "4"]
    assert len(claims) > 1
    assert [d.part for d in claims] == list(range(1, len(claims) + 1))
    assert all(d.parts == len(claims) for d in claims)
    assert all(COUNT(d.text) <= 120 for d in claims)
    for d in claims:  # never cut inside a paragraph
        assert all(line.startswith("Rule ") for line in d.text.split("\n\n"))
    for a, b in pairwise(claims):  # consecutive parts overlap
        assert b.text.split("\n\n")[0] in a.text
    assert drafts[-1].text == "Short section."  # no overlap across sections
    first_rules = {line for d in claims for line in d.text.split("\n\n")}
    assert len(first_rules) == 12


def test_split_tables_repeat_their_header_and_never_cut_a_row() -> None:
    rows = [[f"City {i}", f"${100 + i}", "Regional towns and cities"] for i in range(15)]
    table = Block("table", page=3, header=["City", "Cap", "Examples"], rows=rows)

    drafts = chunk_blocks([heading("6. Lodging"), table])

    assert len(drafts) > 1
    for d in drafts:
        lines = d.text.split("\n")
        assert lines[0] == "| City | Cap | Examples |"
        assert lines[1] == "|---|---|---|"
        assert all(line.startswith("| City ") and line.endswith(" |") for line in lines[2:])
    assert {line for d in drafts for line in d.text.split("\n")[2:]} == {
        f"| City {i} | ${100 + i} | Regional towns and cities |" for i in range(15)
    }


def test_a_rule_keeps_its_exception_and_a_list_keeps_its_introduction() -> None:
    filler = [para(f"Filler paragraph number {i} about travel booking.") for i in range(8)]
    blocks = [
        heading("5. Air travel"),
        *filler,
        para("Business class is not permitted."),
        para("Unless the VP approves it in writing for medical reasons."),
        para("The following are not reimbursed:"),
        item("Minibar charges."),
        item("Spa treatments."),
    ]

    texts = [d.text for d in chunk_blocks(blocks)]

    assert any("Business class is not permitted.\n\nUnless the VP approves" in t for t in texts)
    assert any("not reimbursed:\n- Minibar charges." in t for t in texts)


def test_only_an_oversized_block_is_split_inside() -> None:
    huge = " ".join(f"Sentence {i} explains one more rule." for i in range(40))

    drafts = chunk_blocks([heading("9. Long"), para(huge)])

    assert len(drafts) > 1
    assert all(COUNT(d.text) <= 120 for d in drafts)


def test_definition_tables_become_one_chunk_per_term() -> None:
    glossary = Block(
        "table",
        page=8,
        header=["Term", "Meaning"],
        rows=[["Compa-ratio", "Salary divided by the range midpoint"], ["Lump sum", "One-off"]],
    )
    blocks = [heading("12. Definitions"), para("Terms used in this policy:"), glossary]

    drafts = chunk_blocks(blocks)

    definitions = [d for d in drafts if d.chunk_type == "definition"]
    assert [(d.term, d.text) for d in definitions] == [
        ("Compa-ratio", "Compa-ratio: Salary divided by the range midpoint"),
        ("Lump sum", "Lump sum: One-off"),
    ]
    assert [d.text for d in drafts if d.chunk_type == "section"] == ["Terms used in this policy:"]


# ---------------------------------------------------------- metadata, IDs


REGISTRY = PolicyRegistry(
    {
        "travel.pdf": {
            "policy_name": "Travel Policy",
            "policy_id": "HR-TRV-003",
            "policy_category": "travel",
            "version": 2.0,
            "effective_date": "2027-01-01",
            "applies_to": ["employees", "interns"],
        }
    },
    defaults={"jurisdiction": "US"},
)


def travel_pdf(tmp_path: Path, extra: list[str] | None = None) -> Path:
    return write_pdf(
        tmp_path / "travel.pdf",
        [["Cover text."], ["1. Purpose", "Why we travel.", "5. Air travel", *(extra or [])]],
        title="Travel",
    )


def test_chunks_carry_policy_and_section_metadata_and_a_header(tmp_path: Path) -> None:
    doc = load_document(travel_pdf(tmp_path, ["Economy class."]), settings(), registry=REGISTRY)

    air = next(c for c in doc.chunks if c.metadata["section_number"] == "5")
    m = air.metadata
    assert air.page_content == (
        "Policy: Travel Policy (HR-TRV-003, v2.0, effective 2027-01-01, US)\n"
        "Section: 5 Air travel\n\nEconomy class."
    )
    assert (m["policy_category"], m["jurisdiction"], m["applies_to"]) == (
        "travel",
        "US",
        ["employees", "interns"],
    )
    assert (m["title"], m["is_current"], m["chunk_type"]) == ("Travel Policy", True, "section")
    assert (m["page"], m["page_end"], m["section_title"]) == (2, 2, "Air travel")
    assert citation(m) == "Travel Policy §5 Air travel (travel.pdf, page 2)"
    assert doc.title == "Travel Policy"
    assert json.dumps(m)  # JSON-friendly for the Qdrant payload
    assert str(tmp_path) not in json.dumps(m)


def test_parts_of_a_split_section_share_a_section_id(tmp_path: Path) -> None:
    lines = [f"Rule {i}: book economy and keep every receipt." for i in range(20)]
    doc = load_document(travel_pdf(tmp_path, lines), settings(), registry=REGISTRY)

    air = [c for c in doc.chunks if c.metadata["section_number"] == "5"]
    assert len(air) > 1
    assert len({c.metadata["section_id"] for c in air}) == 1
    assert [c.metadata["section_part"] for c in air] == list(range(1, len(air) + 1))


def test_unregistered_file_uses_defaults_and_its_title_with_a_warning(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "other.pdf", [["Hello."]], title="Gift Policy")

    doc = load_document(pdf, settings(), registry=REGISTRY)

    assert doc.title == "Gift Policy"
    assert doc.chunks[0].metadata["jurisdiction"] == "US"
    assert any("not in the policy registry" in w for w in doc.warnings)


def test_ids_and_hashes_are_deterministic(tmp_path: Path) -> None:
    pdf = travel_pdf(tmp_path, [f"Rule {i}: keep receipts." for i in range(20)])

    first, second = load_document(pdf, settings()), load_document(pdf, settings())

    assert len(first.chunks) > 1
    assert (first.doc_id, first.doc_hash) == (second.doc_id, second.doc_hash)
    assert [c.id for c in first.chunks] == [c.id for c in second.chunks]
    assert [c.metadata["content_hash"] for c in first.chunks] == [
        c.metadata["content_hash"] for c in second.chunks
    ]


def test_changed_file_keeps_doc_id_but_changes_hash(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "a.pdf", [["Version one."]])
    before = load_document(pdf, settings())
    write_pdf(pdf, [["Version two."]])
    after = load_document(pdf, settings())

    assert before.doc_id == after.doc_id
    assert before.doc_hash != after.doc_hash


def test_source_name_overrides_temporary_file_name(tmp_path: Path) -> None:
    tmp = write_pdf(tmp_path / "upload_8f3a.tmp", [["Uploaded text."]])

    doc = load_document(tmp, settings(), source="report.pdf")

    assert (doc.source, doc.file_type) == ("report.pdf", FileType.PDF)
    assert doc.chunks[0].metadata["source"] == "report.pdf"


def test_contextual_headers_can_be_turned_off(tmp_path: Path) -> None:
    doc = load_document(write_pdf(tmp_path / "a.pdf", [["Just the text."]]),
                        settings(contextual_headers=False))  # fmt: skip

    assert doc.chunks[0].page_content == "Just the text."


@pytest.mark.parametrize("name", ["notes.txt", "data.csv", "report.docx"])
def test_unsupported_file_types_are_rejected(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"x")

    with pytest.raises(UnsupportedFileTypeError, match=r"Supported: \.pdf"):
        load_document(path, settings())


def test_extensions_are_case_insensitive(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "SHOUT.PDF", [["Hello."]])

    assert load_document(pdf, settings()).file_type is FileType.PDF


def test_find_supported_files_returns_only_pdfs(tmp_path: Path) -> None:
    write_pdf(tmp_path / "b.pdf", [["B."]])
    (tmp_path / "sub").mkdir()
    write_pdf(tmp_path / "sub" / "a.PDF", [["A."]])
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    assert [f.name for f in find_supported_files(tmp_path)] == ["b.pdf", "a.PDF"]


# --------------------------------------------------------------- registry


def write_registry(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_registry_loads_yaml_with_defaults(tmp_path: Path) -> None:
    path = write_registry(
        tmp_path,
        "defaults:\n  jurisdiction: US\npolicies:\n"
        "  Travel.PDF:\n    policy_name: Travel Policy\n    effective_date: 2027-01-01\n",
    )

    policy, warnings = PolicyRegistry.load(path).lookup("travel.pdf", title="x")

    assert (policy.policy_name, policy.jurisdiction, str(policy.effective_date)) == (
        "Travel Policy",
        "US",
        "2027-01-01",
    )
    assert warnings == []


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ("    policy_name: T\n    efective_date: 2027-01-01\n", "efective_date"),  # typo
        ("    policy_name: T\n    effective_date: someday\n", "effective_date"),
        ("    policy_name: T\n    is_current: maybe\n", "is_current"),
    ],
)
def test_registry_rejects_invalid_entries(tmp_path: Path, entry: str, error: str) -> None:
    path = write_registry(tmp_path, f"policies:\n  t.pdf:\n{entry}")

    with pytest.raises(RegistryError, match=error):
        PolicyRegistry.load(path)


def test_registry_entry_hash_changes_only_for_that_file() -> None:
    a = PolicyRegistry({"a.pdf": {"policy_name": "A"}, "b.pdf": {"policy_name": "B"}})
    b = PolicyRegistry({"a.pdf": {"policy_name": "A2"}, "b.pdf": {"policy_name": "B"}})

    assert a.entry_hash("a.pdf") != b.entry_hash("a.pdf")
    assert a.entry_hash("b.pdf") == b.entry_hash("b.pdf")


def test_the_sample_registry_covers_every_hr_pdf() -> None:
    registry = PolicyRegistry.load(PDFS / "registry.yaml")

    for pdf in PDFS.glob("*.pdf"):
        assert pdf.name in registry, pdf.name


# --------------------------------------------------------------- pipeline


class FakeParser:
    """Returns a fixed ParsedDocument and counts calls (for tables and the parse cache)."""

    name = "fake-v1"

    def __init__(self, parsed: ParsedDocument) -> None:
        self.parsed, self.calls = parsed, 0

    def parse(self, path: Path) -> ParsedDocument:
        self.calls += 1
        return self.parsed


def fake_document() -> ParsedDocument:
    table = Block(
        "table",
        page=2,
        header=["Role", "Contact"],
        rows=[["Payroll", "jane.doe@gmail.com"], ["HR", "hr@acme.com"]],
    )
    return ParsedDocument("Fake", [heading("3. Contacts"), table], pages=2)


def test_table_cells_are_sanitized_without_breaking_rows(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "f.pdf", [["x"]])
    pipeline = IngestionPipeline(settings(), parser=FakeParser(fake_document()))

    doc = pipeline.load(pdf)

    assert body(doc.chunks[0]) == (
        "| Role | Contact |\n|---|---|\n| Payroll | <EMAIL> |\n| HR | <EMAIL> |"
    )
    assert doc.pii_scan["redactions"] == {"EMAIL_ADDRESS": 2}


def test_parse_results_are_cached_by_file_hash(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "f.pdf", [["x"]])
    parser = FakeParser(fake_document())
    pipeline = IngestionPipeline(settings(), parser=parser, cache_dir=tmp_path / "cache")

    first = pipeline.load(pdf)
    second = pipeline.load(pdf)

    assert parser.calls == 1
    assert [c.page_content for c in first.chunks] == [c.page_content for c in second.chunks]
    assert list((tmp_path / "cache").glob("fake-v1-*.json"))


def test_fingerprint_changes_with_chunking_registry_and_parser() -> None:
    base = IngestionPipeline(settings())
    fingerprint = base.fingerprint("a.pdf")

    assert IngestionPipeline(settings(target_tokens=50)).fingerprint("a.pdf") != fingerprint
    registry = PolicyRegistry({"a.pdf": {"policy_name": "A"}})
    assert IngestionPipeline(settings(), registry=registry).fingerprint("a.pdf") != fingerprint
    other = FakeParser(fake_document())
    assert IngestionPipeline(settings(), parser=other).fingerprint("a.pdf") != fingerprint
    assert base.fingerprint("a.pdf") == fingerprint


def test_settings_reject_inconsistent_sizes() -> None:
    with pytest.raises(ValueError, match="target_tokens"):
        ChunkingSettings(target_tokens=900, max_tokens=800)
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingSettings(target_tokens=100, overlap_tokens=100)


# ---------------------------------------------------------------- samples


def test_sample_files_all_load() -> None:
    files = find_supported_files(SAMPLES)
    assert files and {f.suffix for f in files} == {".pdf"}

    for path in files:
        doc = load_document(path, settings(max_tokens=800, target_tokens=400))
        assert doc.chunks, path.name
        assert not doc.warnings, (path.name, doc.warnings)


def test_pdf_loader_is_a_langchain_base_loader(tmp_path: Path) -> None:
    from langchain_core.document_loaders import BaseLoader

    from hybrid_rag.ingestion.loaders import PdfPageLoader

    loader = PdfPageLoader(write_pdf(tmp_path / "a.pdf", [["One."], ["Two."]], title="T"))

    assert isinstance(loader, BaseLoader)
    pages = list(loader.lazy_load())
    assert [(p.page_content.strip(), p.metadata["page"], p.metadata["title"]) for p in pages] == [
        ("One.", 1, "T"),
        ("Two.", 2, "T"),
    ]
