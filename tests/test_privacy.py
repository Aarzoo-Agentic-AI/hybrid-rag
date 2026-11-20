"""Sensitive-data sanitization. All values below are fake."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from hybrid_rag.access import Principal
from hybrid_rag.config import ChunkingSettings, EmbeddingSettings, PrivacySettings
from hybrid_rag.embedding import build_embeddings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionPipeline, load_document
from hybrid_rag.privacy import (
    PatternDetector,
    SensitiveDataError,
    build_sanitizer,
    default_sanitizer,
    disabled_sanitizer,
    lowercase_vocabulary,
    plausible_name,
)
from hybrid_rag.store import DocumentIndex
from tests.parse_fixtures import DoclingFixtureParser
from tests.pdf_fixture import write_pdf

ADMIN = Principal.tenant_admin("default")  # sees every document of the test tenant

PDFS = Path(__file__).resolve().parents[1] / "data" / "pdfs"
CHUNKING = ChunkingSettings(
    target_tokens=100, max_tokens=200, overlap_tokens=12, tokenizer="approx"
)
PIPELINE = IngestionPipeline(CHUNKING)
ALL = ["EMAIL_ADDRESS", "PHONE_NUMBER", "EMPLOYEE_ID", "US_SSN", "CREDIT_CARD", "IBAN_CODE"]


def found(text: str) -> list[tuple[str, str]]:
    detector = PatternDetector(ALL, r"\bEMP-?\d{5,7}\b")
    return [(f.entity, text[f.start : f.end]) for f in detector.detect(text)]


@pytest.fixture(scope="module")
def presidio():
    """The production sanitizer: patterns + Presidio with the small spaCy model."""
    return build_sanitizer(PrivacySettings(allowlist=["hr@dummymocktest.example"]))


# ------------------------------------------------------------ pattern detector


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Write to jane.doe@gmail.com today", [("EMAIL_ADDRESS", "jane.doe@gmail.com")]),
        ("Call (415) 555-0134.", [("PHONE_NUMBER", "(415) 555-0134")]),
        ("Call +1 415.555.0134", [("PHONE_NUMBER", "+1 415.555.0134")]),
        ("SSN 123-45-6789", [("US_SSN", "123-45-6789")]),
        ("card 4111 1111 1111 1111", [("CREDIT_CARD", "4111 1111 1111 1111")]),
        ("IBAN GB82 WEST 1234 5698 7654 32", [("IBAN_CODE", "GB82 WEST 1234 5698 7654 32")]),
        ("employee EMP-104233 joined", [("EMPLOYEE_ID", "EMP-104233")]),
    ],
)
def test_patterns_find_sensitive_values(text: str, expected: list) -> None:
    assert found(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Hotel cap $300 per night; VP approval over $3,000.",
        "Effective January 1, 2027. Policy No. HR-TRV-003.",
        "Call the HR Service Center at 1-800-XXX-XXXX.",
        "Dates 2027-01-15 and 2026-12-31; 401(k) match of 4%.",
        "card 4111 1111 1111 1112",  # fails the Luhn check
        "SSN-like 000-12-3456 and 666-12-3456",  # invalid area numbers
        "IBAN GB00 WEST 1234 5698 7654 32",  # fails the mod-97 check
        "phone 123-456-7890",  # area codes can't start with 1
    ],
)
def test_patterns_ignore_policy_content_and_invalid_numbers(text: str) -> None:
    assert found(text) == []


# --------------------------------------------------------- name plausibility


def test_plausible_name_rejects_policy_words() -> None:
    vocabulary = lowercase_vocabulary(["the policy handbook lists required notices"])

    assert plausible_name("Maria Gonzalez", vocabulary)
    assert plausible_name("Dr. Ahmed Khan", vocabulary)
    assert not plausible_name("Roth", vocabulary)  # one word
    assert not plausible_name("Tier 2", vocabulary)  # digits
    assert not plausible_name("bariatric care", vocabulary)  # lower case
    assert not plausible_name("Policy Handbook", vocabulary)  # words used in lower case
    assert not plausible_name("Required Notices", vocabulary)


def test_email_addresses_do_not_count_as_lowercase_words() -> None:
    vocabulary = lowercase_vocabulary(["Contact maria.gonzalez@gmail.com or www.example.com/maria"])

    assert "maria" not in vocabulary
    assert plausible_name("Maria Gonzalez", vocabulary)


# ----------------------------------------------------------------- sanitizer


def unit(text: str, page: int = 1) -> Document:
    return Document(page_content=text, metadata={"page": page})


def test_redaction_uses_typed_placeholders_and_counts(presidio) -> None:
    units, _, report = presidio.sanitize_units(
        [unit("Ask Maria Gonzalez (maria.gonzalez@gmail.com, 415-555-0134) about EMP-104233.")],
        title="Bonus FAQ",
        source="faq.pdf",
    )

    assert units[0].page_content == "Ask <PERSON> (<EMAIL>, <PHONE>) about <EMPLOYEE_ID>."
    assert report.status == "redacted"
    assert report.counts == {"EMAIL_ADDRESS": 1, "EMPLOYEE_ID": 1, "PERSON": 1, "PHONE_NUMBER": 1}


def test_allowlisted_business_contacts_are_kept(presidio) -> None:
    text = "Questions: hr@dummymocktest.example or 1-800-XXX-XXXX."

    units, _, report = presidio.sanitize_units([unit(text)], title="T", source="t.pdf")

    assert units[0].page_content == text
    assert report.status == "clean"


def test_allowlist_patterns_match_role_mailboxes() -> None:
    sanitizer = build_sanitizer(
        PrivacySettings(detector="patterns", allowlist_patterns=[r"^(hr|payroll)@acme\.com$"])
    )

    units, _, _ = sanitizer.sanitize_units(
        [unit("payroll@acme.com or john.smith@acme.com")], title="T", source="t.pdf"
    )

    assert units[0].page_content == "payroll@acme.com or <EMAIL>"


def test_policy_vocabulary_survives_the_name_detector(presidio) -> None:
    text = (
        "Bereavement leave is 5 days. The Roth 401(k) plan has a Vesting schedule. "
        "Tier 2 cities such as Chicago have a $220 hotel cap. Salary ranges are reviewed."
    )

    units, _, report = presidio.sanitize_units([unit(text)], title="Benefits", source="b.pdf")

    assert units[0].page_content == text
    assert report.status == "clean"


def test_high_risk_data_quarantines_the_document_without_revealing_it(presidio) -> None:
    units = [unit("Policy text."), unit("Deposit: SSN 123-45-6789, card 4111 1111 1111 1111.", 3)]

    with pytest.raises(SensitiveDataError) as caught:
        presidio.sanitize_units(units, title="T", source="payroll.pdf")

    error = caught.value
    assert error.found == {"CREDIT_CARD": 1, "US_SSN": 1}
    assert error.pages == [3]
    assert "quarantined" in str(error) and "page(s) 3" in str(error)
    assert "6789" not in str(error) and "4111" not in str(error)


def test_the_title_is_sanitized_too(presidio) -> None:
    _, title, report = presidio.sanitize_units(
        [unit("Overview of the review.")], title="Review of Maria Gonzalez", source="r.pdf"
    )

    assert title == "Review of <PERSON>"
    assert report.counts == {"PERSON": 1}


def test_verification_catches_anything_the_first_pass_missed() -> None:
    sanitizer = build_sanitizer(PrivacySettings(detector="patterns"))
    leaked = [Document(page_content="Reach me at jane@gmail.com", metadata={"page": 2})]

    with pytest.raises(SensitiveDataError, match="survived sanitization") as caught:
        sanitizer.verify(leaked, source="x.pdf")

    assert caught.value.stage == "verify"
    assert "jane@gmail.com" not in str(caught.value)


def test_policy_hash_changes_with_the_settings() -> None:
    a = build_sanitizer(PrivacySettings(detector="patterns"))
    b = build_sanitizer(PrivacySettings(detector="patterns", allowlist=["hr@x.com"]))

    assert a.policy != b.policy
    assert a.policy == build_sanitizer(PrivacySettings(detector="patterns")).policy


def test_settings_reject_an_entity_that_is_both_redacted_and_quarantined() -> None:
    with pytest.raises(ValueError, match="both redacted and quarantined"):
        PrivacySettings(redact=["US_SSN"], quarantine=["US_SSN"])


# ------------------------------------------------------------------ pipeline


def write_doc(tmp_path: Path, lines: list[str], name: str = "policy.pdf") -> Path:
    return write_pdf(tmp_path / name, [lines], title="Expense FAQ")


def test_load_document_redacts_before_chunking(tmp_path: Path, presidio) -> None:
    pdf = write_doc(
        tmp_path, ["Receipts go to Maria Gonzalez (maria.gonzalez@gmail.com) within 30 days."]
    )

    doc = load_document(pdf, CHUNKING, sanitizer=presidio)

    text = "\n".join(c.page_content for c in doc.chunks)
    assert "Gonzalez" not in text and "gmail" not in text
    assert "<PERSON>" in text and "within 30 days" in text
    scan = doc.chunks[0].metadata["pii_scan"]
    assert scan["status"] == "redacted"
    assert scan["redactions"] == {"EMAIL_ADDRESS": 1, "PERSON": 1}
    assert scan["policy"] == presidio.policy


def test_load_document_uses_the_pattern_sanitizer_by_default(tmp_path: Path) -> None:
    doc = load_document(write_doc(tmp_path, ["Mail jane.doe@gmail.com."]), CHUNKING)

    assert "<EMAIL>" in doc.chunks[0].page_content
    assert doc.pii_scan["detector"] == default_sanitizer().name == "patterns"


def test_disabled_sanitizer_marks_chunks_not_scanned(tmp_path: Path) -> None:
    pdf = write_doc(tmp_path, ["Mail jane.doe@gmail.com."])

    doc = load_document(pdf, CHUNKING, sanitizer=disabled_sanitizer())

    assert "jane.doe@gmail.com" in doc.chunks[0].page_content
    assert doc.pii_scan["status"] == "not_scanned"


def test_quarantined_file_is_reported_and_not_indexed(tmp_path: Path) -> None:
    write_doc(tmp_path, ["Clean policy text."], name="good.pdf")
    write_doc(tmp_path, ["Bank: SSN 123-45-6789."], name="payroll_export.pdf")
    index = DocumentIndex(
        QdrantClient(":memory:"), "t", build_embeddings(EmbeddingSettings(provider="hashing"))
    )

    outcomes = {o.path.name: o for o in ingest_paths([tmp_path], index, PIPELINE)}

    assert outcomes["good.pdf"].ok
    bad = outcomes["payroll_export.pdf"]
    assert bad.quarantined and not bad.ok
    assert [d.source for d in index.list_documents(ADMIN)] == ["good.pdf"]
    stored = [p.payload["page_content"] for p in index.client.scroll("t", limit=100)[0]]
    assert not any("123-45-6789" in text for text in stored)


def test_changed_privacy_settings_reindex_unchanged_files(tmp_path: Path) -> None:
    write_doc(tmp_path, ["Mail payroll@acme.com for help."])
    index = DocumentIndex(
        QdrantClient(":memory:"), "t", build_embeddings(EmbeddingSettings(provider="hashing"))
    )
    strict = build_sanitizer(PrivacySettings(detector="patterns"))
    relaxed = build_sanitizer(PrivacySettings(detector="patterns", allowlist=["payroll@acme.com"]))

    ingest_paths([tmp_path], index, IngestionPipeline(CHUNKING, sanitizer=strict))
    same = ingest_paths([tmp_path], index, IngestionPipeline(CHUNKING, sanitizer=strict))
    changed = ingest_paths([tmp_path], index, IngestionPipeline(CHUNKING, sanitizer=relaxed))

    assert same[0].result.status == "unchanged"
    assert changed[0].result.status == "updated"
    (info,) = index.list_documents(ADMIN)
    assert (info.pii_status, info.pii_policy) == ("clean", relaxed.policy)


# ------------------------------------------------------------- real HR PDFs


def test_hr_policies_are_clean_and_keep_their_content(presidio) -> None:
    """The sample policies have no personal data; nothing in them may be redacted."""
    pdfs = sorted(PDFS.glob("*.pdf"))
    assert len(pdfs) == 5

    for pdf in pdfs:
        doc = load_document(
            pdf,
            ChunkingSettings(tokenizer="approx"),
            sanitizer=presidio,
            parser=DoclingFixtureParser(),
        )
        assert doc.pii_scan["status"] == "clean", (pdf.name, doc.pii_scan["redactions"])
