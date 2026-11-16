import json
from itertools import pairwise
from pathlib import Path

import pytest

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion import (
    EmptyDocumentError,
    FileType,
    IngestionError,
    UnsupportedFileTypeError,
    citation,
    find_supported_files,
    load_document,
)
from hybrid_rag.ingestion.text import clean_text
from tests.pdf_fixture import write_pdf

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


def settings(**overrides) -> ChunkingSettings:
    return ChunkingSettings(**{"chunk_size": 200, "chunk_overlap": 40, **overrides})


def body(chunk) -> str:
    """Chunk text without the contextual header."""
    return chunk.page_content.split("\n\n", 1)[1]


# -------------------------------------------------------------- clean_text


def test_clean_text_normalises_whitespace() -> None:
    raw = "a\r\nb\t\tc  d \n\n\n\ne\x00"
    assert clean_text(raw) == "a\nb c d\n\ne"


def test_clean_text_joins_hyphenated_line_breaks_when_asked() -> None:
    assert clean_text("hy-\nbrid", join_hyphenated=True) == "hybrid"
    assert clean_text("hy-\nbrid") == "hy-\nbrid"


# --------------------------------------------------------------------- PDF


def test_pdf_chunks_carry_page_numbers_and_title(tmp_path: Path) -> None:
    pdf = write_pdf(
        tmp_path / "guide.pdf",
        [["Remote work is allowed three days a week."], ["Annual leave is 25 days."]],
        title="Staff Guide",
    )

    doc = load_document(pdf, settings())

    assert doc.file_type is FileType.PDF
    assert doc.title == "Staff Guide"
    assert [c.metadata["page"] for c in doc.chunks] == [1, 2]
    assert citation(doc.chunks[1].metadata) == "guide.pdf, page 2"
    assert doc.chunks[0].page_content.startswith(
        "Document: Staff Guide (guide.pdf)\nLocation: page 1\n\n"
    )
    assert doc.stats == {"pages": 2, "pages_with_text": 2, "chunks": 2}


def test_long_pdf_page_is_split_without_crossing_pages(tmp_path: Path) -> None:
    lines = [f"Sentence number {i} about robots and safety rules." for i in range(30)]
    pdf = write_pdf(tmp_path / "long.pdf", [lines, ["Short second page."]])

    doc = load_document(pdf, settings(chunk_size=300, chunk_overlap=60))

    page_one = [c for c in doc.chunks if c.metadata["page"] == 1]
    assert len(page_one) > 1
    assert all(len(body(c)) <= 300 for c in doc.chunks)
    assert doc.chunks[-1].metadata["page"] == 2
    assert body(doc.chunks[-1]) == "Short second page."


def test_chunks_respect_size_and_overlap(tmp_path: Path) -> None:
    words = [f"word{i}" for i in range(200)]
    lines = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]  # 40 short lines
    pdf = write_pdf(tmp_path / "words.pdf", [lines])

    doc = load_document(pdf, settings(chunk_size=200, chunk_overlap=40))

    bodies = [body(c) for c in doc.chunks]
    assert len(bodies) > 1
    assert all(len(b) <= 200 for b in bodies)
    for previous, current in pairwise(bodies):
        assert current.split(" ")[0] in previous  # each chunk starts inside the previous one


def test_pdf_pages_without_text_are_skipped_with_a_warning(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "scan.pdf", [["Text page."], [], [], ["Another."]])

    doc = load_document(pdf, settings())

    assert [c.metadata["page"] for c in doc.chunks] == [1, 4]
    assert doc.warnings and "2-3" in doc.warnings[0]


def test_pdf_with_no_text_at_all_is_rejected(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "blank.pdf", [[], []])

    with pytest.raises(EmptyDocumentError, match="OCR"):
        load_document(pdf, settings())


def test_pdf_without_title_uses_file_name(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "notes.pdf", [["Hello."]])

    doc = load_document(pdf, settings())

    assert doc.title == "notes"
    assert doc.chunks[0].page_content.startswith("Document: notes.pdf\n")


def test_damaged_pdf_raises_ingestion_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 this is not really a pdf")

    with pytest.raises(IngestionError, match=r"broken\.pdf"):
        load_document(broken, settings())


# ---------------------------------------------------------- IDs and schema


def test_ids_and_hashes_are_deterministic(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "a.pdf", [["Some text. " * 5] * 20])

    first = load_document(pdf, settings())
    second = load_document(pdf, settings())

    assert len(first.chunks) > 1
    assert first.doc_id == second.doc_id
    assert first.doc_hash == second.doc_hash
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

    assert doc.source == "report.pdf"
    assert doc.file_type is FileType.PDF
    assert doc.chunks[0].metadata["source"] == "report.pdf"


def test_metadata_is_json_serialisable_and_has_no_local_paths(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "x.pdf", [["Hello."]])

    for chunk in load_document(pdf, settings()).chunks:
        dumped = json.dumps(chunk.metadata)
        assert str(tmp_path) not in dumped
        assert chunk.id == chunk.metadata["chunk_id"]


def test_contextual_headers_can_be_turned_off(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "a.pdf", [["Just the text."]])

    doc = load_document(pdf, settings(contextual_headers=False))

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

    files = find_supported_files(tmp_path)

    assert [f.name for f in files] == ["b.pdf", "a.PDF"]


# ----------------------------------------------------------------- samples


def test_sample_files_all_load() -> None:
    files = find_supported_files(SAMPLES)
    assert files and {f.suffix for f in files} == {".pdf"}

    for path in files:
        doc = load_document(path, ChunkingSettings())
        assert doc.chunks, path.name
        assert not doc.warnings, (path.name, doc.warnings)


# ------------------------------------------------- LangChain loader class


def test_pdf_loader_is_a_langchain_base_loader(tmp_path: Path) -> None:
    from langchain_core.document_loaders import BaseLoader

    from hybrid_rag.ingestion.loaders import PdfPageLoader

    pdf = write_pdf(tmp_path / "a.pdf", [["One."], ["Two."]], title="T")
    loader = PdfPageLoader(pdf)

    assert isinstance(loader, BaseLoader)
    pages = list(loader.lazy_load())
    # Raw pypdf text; cleaning happens in load_pdf().
    assert [(p.page_content.strip(), p.metadata["page"], p.metadata["title"]) for p in pages] == [
        ("One.", 1, "T"),
        ("Two.", 2, "T"),
    ]
