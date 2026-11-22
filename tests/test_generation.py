"""Answer generation with scripted LLMs (no API calls) over an in-memory index."""

from datetime import date
from pathlib import Path

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from hybrid_rag.access import Principal
from hybrid_rag.config import (
    AnswerSettings,
    ChunkingSettings,
    EmbeddingSettings,
    PrivacySettings,
    RerankerSettings,
    RetrievalSettings,
    Settings,
)
from hybrid_rag.embedding import build_embeddings
from hybrid_rag.generation import (
    NOT_FOUND,
    AnswerService,
    Source,
    format_sources,
    parse_citations,
    select_sources,
)
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionPipeline, PolicyRegistry
from hybrid_rag.ingestion.tokens import ApproxTokenCounter
from hybrid_rag.privacy import build_sanitizer
from hybrid_rag.store import DocumentIndex
from tests.llm_fakes import Script, make_failover, openai_server_error
from tests.pdf_fixture import write_pdf
from tests.test_retrieval import KeywordCrossEncoder

TOKENS = ApproxTokenCounter()
EMPLOYEE = Principal("acme", "u-1", "internal")
HR_EMAIL = "hr@acme.example"
SANITIZER = build_sanitizer(PrivacySettings(detector="patterns", allowlist=[HR_EMAIL]))


def settings(**answer) -> Settings:
    return Settings.model_construct(
        retrieval=RetrievalSettings(fused_limit=10, top_k=4),
        reranker=RerankerSettings(),
        chunking=ChunkingSettings(tokenizer="approx"),
        answer=AnswerSettings(**{"min_rerank_score": None, "contact": HR_EMAIL, **answer}),
    )


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> DocumentIndex:
    return build_index(tmp_path_factory.mktemp("gen"))


def index_for_tracing() -> DocumentIndex:
    import tempfile

    return build_index(Path(tempfile.mkdtemp()))


def build_index(folder: Path) -> DocumentIndex:
    write_pdf(
        folder / "travel.pdf",
        [
            [
                "1. Air travel",
                "Economy class is required for flights of six hours or less.",
                "Premium economy is allowed for flights over six hours.",
                "2. Hotels",
                "The nightly hotel cap in Boston is 300 dollars.",
            ]
        ],
    )
    write_pdf(folder / "leave.pdf", [["1. Paid time off", "Employees get 15 PTO days a year."]])
    registry = PolicyRegistry(
        {
            "travel.pdf": {"policy_name": "Travel Policy", "policy_id": "HR-TRV-003"},
            "leave.pdf": {"policy_name": "Leave Policy"},
        },
        defaults={
            "effective_date": "2027-01-01",
            "access": {"classification": "internal", "visibility": "tenant"},
        },
    )
    index = DocumentIndex(
        QdrantClient(":memory:"), "g", build_embeddings(EmbeddingSettings(provider="hashing"))
    )
    pipeline = IngestionPipeline(
        ChunkingSettings(tokenizer="approx"), registry=registry, tenant_id="acme"
    )
    assert all(o.ok for o in ingest_paths([folder], index, pipeline))
    return index


def service(index, *scripts: Script, rewrite: Script | None = None, **answer) -> AnswerService:
    llm, _, _ = make_failover(*scripts, max_attempts=2)
    rewrite_llm = make_failover(rewrite)[0] if rewrite else None
    return AnswerService(
        settings(**answer),
        index,
        llm,
        rewrite_llm=rewrite_llm,
        sanitizer=SANITIZER,
        tokens=TOKENS,
        cross_encoder=KeywordCrossEncoder(),
        today=date(2026, 11, 21),
    )


def prompt_text(script: Script, call: int = -1) -> str:
    return "\n".join(str(m.content) for m in script.seen[call])


# ------------------------------------------------------------- answering


def test_answer_cites_its_sources_and_drops_invalid_markers(index) -> None:
    script = Script(["Economy class is required up to six hours [1]. Business class is fine [9]."])

    answer = service(index, script).answer("Can I fly economy class?", EMPLOYEE)

    assert answer.status == "answered" and answer.grounded
    assert "[9]" not in answer.text and "[1]" in answer.text
    (cited,) = answer.citations
    assert cited.number == 1 and cited.title == "Travel Policy"
    assert cited.effective_date == "2027-01-01" and cited.policy_id == "HR-TRV-003"
    assert "Travel Policy" in cited.label and "travel.pdf" in cited.label
    assert answer.sources[0].cited is True
    assert answer.provider["provider"] == "p0" and not answer.provider["failed_over"]


def test_prompt_has_numbered_labelled_sources_and_todays_date(index) -> None:
    script = Script(["ok [1]"])

    service(index, script).answer("hotel cap Boston", EMPLOYEE)

    system, human = script.seen[0]
    assert "Today is 2026-11-21" in system.content
    assert "[1] Travel Policy" in human.content and "effective 2027-01-01" in human.content
    assert "Question: hotel cap Boston" in human.content
    assert "Policy:" not in human.content  # contextual headers are stripped from the text
    assert "Boston" not in system.content  # sources go in the human turn, never the system one


def test_not_found_reply_becomes_the_contact_hr_answer(index) -> None:
    answer = service(index, Script([NOT_FOUND])).answer("Can I bring my dog?", EMPLOYEE)

    assert answer.status == "not_found" and not answer.grounded
    assert HR_EMAIL in answer.text and answer.citations == []


def test_an_answer_without_citations_is_flagged(index) -> None:
    answer = service(index, Script(["You get 15 PTO days."])).answer("PTO days?", EMPLOYEE)

    assert answer.status == "answered" and not answer.grounded
    assert "could not be linked to a policy source" in answer.text


def test_low_confidence_skips_the_llm(index) -> None:
    script = Script([])

    answer = service(index, script, min_rerank_score=100.0).answer("hotel cap", EMPLOYEE)

    assert answer.status == "low_confidence" and script.calls == 0
    assert answer.sources  # what was found is still reported, for debugging


def test_callers_with_nothing_to_read_get_the_same_answer_without_an_llm_call(index) -> None:
    script = Script([])
    outsider = Principal("globex", "u-1", "restricted")

    none = service(index, script).answer("hotel cap Boston", outsider)
    low = service(index, script, min_rerank_score=100.0).answer("hotel cap Boston", EMPLOYEE)

    assert none.status == "no_sources" and script.calls == 0 and none.sources == []
    assert none.text == low.text  # never reveals whether a hidden policy matched


def test_failover_to_the_second_provider(index) -> None:
    failing = Script([openai_server_error(), openai_server_error()])
    backup = Script(["15 PTO days a year [1]."])

    answer = service(index, failing, backup).answer("How many PTO days?", EMPLOYEE)

    assert answer.grounded and answer.provider["provider"] == "p1"
    assert answer.provider["failed_over"] is True and failing.calls == 2


# --------------------------------------------------------------- privacy


def test_personal_data_in_the_question_never_reaches_the_llm(index) -> None:
    script = Script(["15 days [1]."])

    answer = service(index, script).answer(
        "I'm jane.doe@gmail.com, call me on 415-555-0134: how many PTO days?", EMPLOYEE
    )

    prompt = prompt_text(script)
    assert "jane.doe@gmail.com" not in prompt and "415-555-0134" not in prompt
    assert "<EMAIL>" in prompt and "<PHONE>" in prompt
    assert answer.redactions == {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1}
    assert "jane.doe" not in answer.question


def test_the_answer_gets_a_final_pattern_check(index) -> None:
    script = Script([f"Call 415-555-0134 or write to {HR_EMAIL} [1]."])

    answer = service(index, script).answer("PTO days?", EMPLOYEE)

    assert "415-555-0134" not in answer.text and "<PHONE>" in answer.text
    assert HR_EMAIL in answer.text  # allowlisted contacts stay
    assert answer.output_redactions == {"PHONE_NUMBER": 1}


# ---------------------------------------------------------- conversation


def test_follow_ups_are_rewritten_from_redacted_history(index) -> None:
    rewrite = Script(["How many PTO days do interns get?"])
    answer_script = Script(["15 days [1]."])
    history = [("human", "How many PTO days do I get? I'm jane.doe@gmail.com"), ("ai", "15 [1].")]

    answer = service(index, answer_script, rewrite=rewrite).answer(
        "and interns?", EMPLOYEE, history=history
    )

    assert answer.rewritten and answer.question == "How many PTO days do interns get?"
    assert "jane.doe@gmail.com" not in prompt_text(rewrite)
    assert "Question: How many PTO days do interns get?" in prompt_text(answer_script)


def test_without_history_nothing_is_rewritten(index) -> None:
    rewrite = Script([])

    answer = service(index, Script(["ok [1]"]), rewrite=rewrite).answer("PTO days?", EMPLOYEE)

    assert rewrite.calls == 0 and not answer.rewritten


# ------------------------------------------------------------- streaming


def test_streaming_yields_tokens_then_the_final_answer(index) -> None:
    events = list(service(index, Script(["You get 15 PTO days [1]."])).stream("PTO?", EMPLOYEE))

    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    (final,) = [e["answer"] for e in events if e["type"] == "answer"]
    assert events[-1]["type"] == "answer"
    assert tokens.strip() == "You get 15 PTO days [1]."
    assert final.grounded and final.text == "You get 15 PTO days [1]."


def test_streaming_never_shows_not_found(index) -> None:
    events = list(service(index, Script([NOT_FOUND])).stream("dog?", EMPLOYEE))

    assert [e["type"] for e in events] == ["answer"]
    assert events[0]["answer"].status == "not_found"


# ------------------------------------------------------ context building


def doc(chunk_id: str, text: str, **meta) -> Document:
    metadata = {"chunk_id": chunk_id, "source": "p.pdf", "title": "Policy", **meta}
    return Document(page_content=text, metadata=metadata, id=chunk_id)


def test_sources_are_deduplicated_ordered_and_kept_within_budget() -> None:
    docs = [doc("a", "short rule"), doc("a", "short rule"), doc("b", "short rule")]
    docs += [doc("c", "x " * 2000), doc("d", "another short rule")]

    sources = select_sources(docs, TOKENS, max_tokens=100)

    assert [s.doc.id for s in sources] == ["a", "d"]  # duplicate id, duplicate text, too long
    assert [s.number for s in sources] == [1, 2]
    assert format_sources(sources).startswith("[1] Policy")


def test_the_best_source_always_goes_in() -> None:
    (only,) = select_sources([doc("big", "x " * 2000)], TOKENS, max_tokens=10)

    assert only.doc.id == "big"


def test_citation_markers_are_parsed() -> None:
    sources = [Source(1, doc("a", "t1")), Source(2, doc("b", "t2"))]

    text, citations, invalid = parse_citations("A [2]. B [1, 7]. C [8].", sources)

    assert text == "A [2]. B [1]. C."
    assert [c.number for c in citations] == [2, 1]
    assert invalid == [7, 8]
