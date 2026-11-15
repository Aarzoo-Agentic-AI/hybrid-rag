import json
from itertools import pairwise
from pathlib import Path

import pytest

from hybrid_rag.config import ChunkingSettings
from hybrid_rag.ingestion import (
    EmptyDocumentError,
    FileType,
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


def test_clean_text_can_keep_line_numbers() -> None:
    assert clean_text("\n\nx\n\n\n\ny  \n", collapse_blank_lines=False) == "\n\nx\n\n\n\ny"


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


# --------------------------------------------------------------------- TXT


def test_txt_line_numbers_match_the_file(tmp_path: Path) -> None:
    lines = [f"Line {i}: the LR-50 courier robot delivers parcels." for i in range(1, 41)]
    txt = tmp_path / "log.txt"
    txt.write_text("\n\n" + "\n".join(lines) + "\n", encoding="utf-8")  # two leading blank lines

    doc = load_document(txt, settings())

    assert len(doc.chunks) > 1
    original = txt.read_text(encoding="utf-8").split("\n")
    for chunk in doc.chunks:
        start, end = chunk.metadata["line_start"], chunk.metadata["line_end"]
        expected = "\n".join(original[start - 1 : end]).strip()
        assert body(chunk) == expected


def test_txt_chunks_respect_size_and_overlap(tmp_path: Path) -> None:
    words = " ".join(f"word{i}" for i in range(400))
    txt = tmp_path / "words.txt"
    txt.write_text(words, encoding="utf-8")

    doc = load_document(txt, settings(chunk_size=200, chunk_overlap=40))

    bodies = [body(c) for c in doc.chunks]
    assert all(len(b) <= 200 for b in bodies)
    for previous, current in pairwise(bodies):
        assert current.split(" ")[0] in previous  # each chunk starts inside the previous one


def test_txt_in_windows_encoding_is_decoded(tmp_path: Path) -> None:
    txt = tmp_path / "legacy.txt"
    txt.write_bytes("Café prices: 5 € per coffee.".encode("cp1252"))

    doc = load_document(txt, settings())

    assert "Café prices: 5 € per coffee." in doc.chunks[0].page_content


def test_empty_txt_is_rejected(tmp_path: Path) -> None:
    txt = tmp_path / "empty.txt"
    txt.write_text("  \n\n  ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(txt, settings())


# --------------------------------------------------------------------- CSV


CSV_TEXT = (
    "﻿sku,name,price,notes\n"  # Excel-style byte-order mark
    "A-1,Widget,9.99,\n"
    '"B-2","Gadget, Pro",19.50,"two\nlines"\n'
    ",,,\n"
    "C-3,Gizmo,5,cheap\n"
)


def test_csv_rows_become_column_value_chunks(tmp_path: Path) -> None:
    csv_file = tmp_path / "products.csv"
    csv_file.write_text(CSV_TEXT, encoding="utf-8")

    doc = load_document(csv_file, settings())

    assert len(doc.chunks) == 3
    first = doc.chunks[0]
    assert body(first) == "sku: A-1\nname: Widget\nprice: 9.99"  # empty "notes" left out
    assert first.metadata["columns"] == ["sku", "name", "price", "notes"]  # no BOM in "sku"
    assert body(doc.chunks[1]) == "sku: B-2\nname: Gadget, Pro\nprice: 19.50\nnotes: two\nlines"
    assert [c.metadata["row_start"] for c in doc.chunks] == [1, 2, 4]  # row 3 was empty
    assert citation(doc.chunks[2].metadata) == "products.csv, row 4"
    assert doc.warnings == ["1 empty row(s) skipped"]


def test_csv_rows_can_be_grouped(tmp_path: Path) -> None:
    csv_file = tmp_path / "products.csv"
    csv_file.write_text(CSV_TEXT, encoding="utf-8")

    doc = load_document(csv_file, settings(csv_rows_per_chunk=2))

    assert len(doc.chunks) == 2
    assert (doc.chunks[0].metadata["row_start"], doc.chunks[0].metadata["row_end"]) == (1, 2)
    assert body(doc.chunks[0]).startswith("Row 1\nsku: A-1")
    assert "\n\nRow 2\nsku: B-2" in body(doc.chunks[0])
    assert citation(doc.chunks[0].metadata) == "products.csv, rows 1-2"


def test_csv_without_data_rows_is_rejected(tmp_path: Path) -> None:
    csv_file = tmp_path / "header_only.csv"
    csv_file.write_text("a,b,c\n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(csv_file, settings())


# ---------------------------------------------------------- IDs and schema


def test_ids_and_hashes_are_deterministic(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("Some text. " * 50, encoding="utf-8")

    first = load_document(txt, settings())
    second = load_document(txt, settings())

    assert first.doc_id == second.doc_id
    assert first.doc_hash == second.doc_hash
    assert [c.id for c in first.chunks] == [c.id for c in second.chunks]
    assert [c.metadata["content_hash"] for c in first.chunks] == [
        c.metadata["content_hash"] for c in second.chunks
    ]


def test_changed_file_keeps_doc_id_but_changes_hash(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("Version one.", encoding="utf-8")
    before = load_document(txt, settings())
    txt.write_text("Version two.", encoding="utf-8")
    after = load_document(txt, settings())

    assert before.doc_id == after.doc_id
    assert before.doc_hash != after.doc_hash


def test_source_name_overrides_temporary_file_name(tmp_path: Path) -> None:
    tmp = tmp_path / "upload_8f3a.tmp"
    tmp.write_text("Uploaded text.", encoding="utf-8")

    doc = load_document(tmp, settings(), source="report.txt")

    assert doc.source == "report.txt"
    assert doc.file_type is FileType.TXT
    assert doc.chunks[0].metadata["source"] == "report.txt"


def test_metadata_is_json_serialisable_and_has_no_local_paths(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "x.pdf", [["Hello."]])

    for chunk in load_document(pdf, settings()).chunks:
        dumped = json.dumps(chunk.metadata)
        assert str(tmp_path) not in dumped
        assert chunk.id == chunk.metadata["chunk_id"]


def test_contextual_headers_can_be_turned_off(tmp_path: Path) -> None:
    txt = tmp_path / "a.txt"
    txt.write_text("Just the text.", encoding="utf-8")

    doc = load_document(txt, settings(contextual_headers=False))

    assert doc.chunks[0].page_content == "Just the text."


def test_unsupported_file_type_is_rejected(tmp_path: Path) -> None:
    docx = tmp_path / "report.docx"
    docx.write_bytes(b"x")

    with pytest.raises(UnsupportedFileTypeError, match=r"\.pdf, \.txt, \.csv"):
        load_document(docx, settings())


def test_extensions_are_case_insensitive(tmp_path: Path) -> None:
    txt = tmp_path / "SHOUT.TXT"
    txt.write_text("Hello.", encoding="utf-8")

    assert load_document(txt, settings()).file_type is FileType.TXT


# ----------------------------------------------------------------- samples


def test_sample_files_all_load() -> None:
    files = find_supported_files(SAMPLES)
    assert {f.suffix for f in files} == {".pdf", ".txt", ".csv"}

    for path in files:
        doc = load_document(path, ChunkingSettings())
        assert doc.chunks, path.name
        assert not doc.warnings, (path.name, doc.warnings)


def test_sample_release_notes_keep_error_codes_intact() -> None:
    doc = load_document(SAMPLES / "lumen_release_notes.txt", ChunkingSettings())

    text = "\n".join(c.page_content for c in doc.chunks)
    for code in ("E-1001", "E-2204", "E-3301", "E-4012", "E-4107", "E-5500"):
        assert code in text


# ------------------------------------------------- LangChain loader classes


def test_loaders_are_langchain_base_loaders(tmp_path: Path) -> None:
    from langchain_core.document_loaders import BaseLoader

    from hybrid_rag.ingestion.loaders import CsvRowLoader, PdfPageLoader, TextFileLoader

    pdf = write_pdf(tmp_path / "a.pdf", [["One."], ["Two."]], title="T")
    txt = tmp_path / "a.txt"
    txt.write_text("Hello.", encoding="utf-8")
    csv_file = tmp_path / "a.csv"
    csv_file.write_text("x,y\n1,2\n", encoding="utf-8")

    for loader in (PdfPageLoader(pdf), TextFileLoader(txt), CsvRowLoader(csv_file)):
        assert isinstance(loader, BaseLoader)

    pages = list(PdfPageLoader(pdf).lazy_load())
    # Raw pypdf text; cleaning happens in load_pdf().
    assert [(p.page_content.strip(), p.metadata["page"], p.metadata["title"]) for p in pages] == [
        ("One.", 1, "T"),
        ("Two.", 2, "T"),
    ]
    assert TextFileLoader(txt).load()[0].page_content == "Hello."
    (row,) = CsvRowLoader(csv_file).load()
    assert (row.page_content, row.metadata) == ("x: 1\ny: 2", {"row": 1, "columns": ["x", "y"]})


def test_csv_values_in_extra_unnamed_columns_are_kept(tmp_path: Path) -> None:
    from hybrid_rag.ingestion.loaders import CsvRowLoader

    csv_file = tmp_path / "ragged.csv"
    csv_file.write_text("a,b\n1,2,extra\n", encoding="utf-8")

    (row,) = CsvRowLoader(csv_file).load()

    assert row.page_content == "a: 1\nb: 2\ncolumn 3: extra"


def test_damaged_pdf_raises_ingestion_error(tmp_path: Path) -> None:
    from hybrid_rag.ingestion import IngestionError

    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 this is not really a pdf")

    with pytest.raises(IngestionError, match=r"broken\.pdf"):
        load_document(broken, settings())
