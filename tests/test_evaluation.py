"""The golden dataset, the metrics and the runners (no models, no Qdrant)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from hybrid_rag.evaluation import (
    CATEGORIES,
    GoldenError,
    GoldenQuestion,
    Requirement,
    best_threshold,
    facts_found,
    first_ranks,
    gate_rows,
    load_golden,
    normalize,
    parse_golden,
    recall_at,
    reciprocal_rank,
    render_answers,
    render_retrieval,
    run_answers,
    run_retrieval,
    summarize,
)
from hybrid_rag.generation.types import Answer, Citation

GOLDEN = Path(__file__).resolve().parent.parent / "eval" / "golden.yaml"
CALLERS = ("employee", "manager", "hr-comp", "admin")
SOURCES = {"trv": "Travel.pdf", "merit": "Merit.pdf"}


def golden(*questions: dict[str, Any]) -> Any:
    return parse_golden({"sources": SOURCES, "questions": list(questions)}, callers=CALLERS)


def q(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "trv-hotel",
        "question": "Hotel cap in Boston?",
        "category": "table_lookup",
        "style": "keyword",
        "expected": ["trv::6 Lodging"],
        "facts": ["$300"],
    } | overrides


def doc(source: str, section: str, score: float | None = None) -> Document:
    meta: dict[str, Any] = {"source": source, "section_path": section}
    if score is not None:
        meta["rerank_score"] = score
    return Document(page_content="x", metadata=meta)


# ------------------------------------------------------------------ the golden file


def test_the_golden_file_is_valid_and_covers_every_category() -> None:
    g = load_golden(GOLDEN, callers=CALLERS)

    counts = {c: sum(1 for x in g.questions if x.category == c) for c in CATEGORIES}
    assert 50 <= len(g.questions) <= 100
    assert all(counts[c] >= 5 for c in CATEGORIES), counts
    assert {x.style for x in g.questions} == {"keyword", "paraphrase"}
    assert len(g.sha256) == 64
    # Every restricted question is checked against a caller without access.
    merit = g.sources["merit"]
    for x in g.questions:
        if merit in x.expected_sources:
            assert x.caller != "employee" and (x.hidden_from or x.category == "follow_up"), x.id


def test_labels_resolve_aliases_and_alternatives() -> None:
    g = golden(q(expected=[["trv::6 Lodging", "merit::7 Bonus"], "trv::8 Meals"]))

    (question,) = g.questions
    assert question.expected[0].options == {("Travel.pdf", "6 Lodging"), ("Merit.pdf", "7 Bonus")}
    assert question.expected[1].options == {("Travel.pdf", "8 Meals")}
    assert question.expected_sources == {"Travel.pdf", "Merit.pdf"}
    assert g.labels() == {
        ("Travel.pdf", "6 Lodging"),
        ("Merit.pdf", "7 Bonus"),
        ("Travel.pdf", "8 Meals"),
    }
    assert question.facts == (("$300",),)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expected": ["hr::1 Intro"]}, "unknown source alias 'hr'"),
        ({"expected": ["6 Lodging"]}, "must be '<alias>::<section path>'"),
        ({"expected": []}, "need at least one expected section"),
        ({"category": "unanswerable"}, "unanswerable questions have no expected"),
        ({"category": "trivia"}, "category must be one of"),
        ({"style": "fuzzy"}, "style must be one of"),
        ({"caller": "ceo"}, "unknown caller 'ceo'"),
        ({"caller": "employee", "hidden_from": ["employee"]}, "caller is also in hidden_from"),
        ({"category": "follow_up"}, "need history and standalone"),
        ({"standalone": "x"}, "only follow_up questions"),
        ({"id": "Bad ID"}, "lowercase words"),
        ({"answer": "x"}, "unknown field 'answer'"),
        ({"facts": ["|"]}, "empty fact"),
    ],
)
def test_invalid_questions_are_refused_with_the_reason(override: dict, message: str) -> None:
    with pytest.raises(GoldenError, match=message.replace("(", r"\(").replace(")", r"\)")):
        golden(q(**override))


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(GoldenError, match="duplicate id"):
        golden(q(), q())


def test_follow_ups_keep_history_and_score_the_standalone_question() -> None:
    g = golden(
        q(
            category="follow_up",
            question="And in Chicago?",
            history=[{"user": "Hotel cap in New York?", "assistant": "$300 [1]."}],
            standalone="Hotel cap in Chicago?",
        )
    )

    (question,) = g.questions
    assert question.history == (("user", "Hotel cap in New York?"), ("assistant", "$300 [1]."))
    assert question.retrieval_query == "Hotel cap in Chicago?"


def test_select_keeps_only_the_given_ids() -> None:
    g = golden(q(), q(id="trv-meals"))

    assert [x.id for x in g.select(["trv-meals"]).questions] == ["trv-meals"]
    with pytest.raises(GoldenError, match="Unknown question IDs: nope"):
        g.select(["nope"])


# ------------------------------------------------------------------ metrics


A, B, C = ("a.pdf", "1"), ("b.pdf", "2"), ("c.pdf", "3")


def test_ranks_recall_and_mrr() -> None:
    reqs = [Requirement(frozenset({A})), Requirement(frozenset({B, C}))]  # B or C meets the second

    ranks = first_ranks([("x", "x"), C, A, B], reqs)

    assert ranks == [3, 2]
    assert recall_at(ranks, 1) == 0
    assert recall_at(ranks, 2) == 0.5
    assert recall_at(ranks, 3) == 1
    assert reciprocal_rank(ranks) == 0.5
    assert first_ranks([], reqs) == [None, None]
    assert reciprocal_rank([None, None]) == 0
    with pytest.raises(ValueError):
        recall_at([], 5)


def test_fact_matching_ignores_dash_style_case_and_thousands_separators() -> None:
    nbh, nbsp, em = chr(0x2011), chr(0xA0), chr(0x2014)
    text = f"Up to $20,000 lifetime {em} Premium{nbh}economy for flights over 6{nbsp}hours."

    assert normalize(text) == "up to $20000 lifetime - premium-economy for flights over 6 hours."
    assert facts_found(text, [("$20,000",), ("premium-economy",), ("6 hours",)]) == [True] * 3
    assert facts_found(text, [("business class", "first class"), ("lifetime",)]) == [False, True]


def test_gate_rows_and_best_threshold() -> None:
    answerable = [-2.0, -4.0, -9.0]
    unanswerable = [-10.0, -5.0, None]  # None: nothing retrieved, always refused

    rows = {r["threshold"]: r for r in gate_rows(answerable, unanswerable, [-9.5, -4.5])}

    assert rows[-9.5]["kept"] == 1 and rows[-9.5]["refused"] == pytest.approx(2 / 3)
    assert rows[-4.5]["kept"] == pytest.approx(2 / 3) and rows[-4.5]["refused"] == 1
    best = best_threshold(answerable, unanswerable)
    assert best is not None and best["balanced"] == pytest.approx(5 / 6)
    assert best_threshold(answerable, []) is None


# ------------------------------------------------------------------ runners


def _golden_for_runs() -> Any:
    return golden(
        q(),
        q(
            id="merit-matrix",
            question="Merit matrix?",
            category="access",
            caller="manager",
            hidden_from=["employee"],
            expected=["merit::6.2 Merit matrix"],
            facts=["2.5%"],
        ),
        {
            "id": "na-ceo",
            "question": "Who is the CEO?",
            "category": "unanswerable",
            "style": "keyword",
        },
    )


def _search(query: str, caller: str, mode: str) -> list[Document]:
    lodging, merit, other = (
        doc("Travel.pdf", "6 Lodging"),
        doc("Merit.pdf", "6.2 Merit matrix"),
        doc("Travel.pdf", "1 Purpose"),
    )
    if query.startswith("Hotel"):
        return [other, lodging] if mode == "sparse" else [lodging, other]
    if query.startswith("Merit"):
        return [merit] if caller == "manager" else [other]
    return [doc("Travel.pdf", "1 Purpose", score=-11.0)]


def test_run_retrieval_scores_each_mode_and_checks_restricted_sources() -> None:
    g = _golden_for_runs()

    runs, leaks = run_retrieval(g, ["dense", "sparse"], _search)

    dense, sparse = (
        summarize(r for r in runs if r.mode == "dense"),
        summarize(r for r in runs if r.mode == "sparse"),
    )
    assert dense["n"] == 2 and dense["recall@1"] == 1 and dense["mrr"] == 1
    assert sparse["recall@1"] == 0.5 and sparse["mrr"] == 0.75
    assert [(lk.id, lk.caller, lk.leaked) for lk in leaks] == [
        ("merit-matrix", "employee", []),
        ("merit-matrix", "employee", []),
    ]
    na = next(r for r in runs if r.id == "na-ceo")
    assert na.ranks == [] and na.top_rerank_score == -11.0

    report = render_retrieval(
        g, runs, leaks, ["dense", "sparse"], {"commit": "abc"}, current_threshold=-12.0
    )
    assert "| `dense` | 100% |" in report
    assert "No chunk from a restricted source was returned" in report


def test_a_leak_is_reported() -> None:
    g = _golden_for_runs()

    def leaky(query: str, caller: str, mode: str) -> list[Document]:
        return (
            [doc("Merit.pdf", "6.2 Merit matrix")]
            if query.startswith("Merit")
            else _search(query, caller, mode)
        )

    _, leaks = run_retrieval(g, ["dense"], leaky)

    assert leaks[0].leaked == [("Merit.pdf", "6.2 Merit matrix")]


def _citation(source: str, section: str) -> Citation:
    return Citation(1, "label", "title", source, section, 1, 1, None, None, "doc", "chunk")


def test_run_answers_scores_facts_citations_refusals_and_hidden_runs() -> None:
    g = _golden_for_runs()
    answers = {
        ("trv-hotel", "employee"): Answer("q", "The cap is $300 [1].", "answered", grounded=True,
                                          citations=[_citation("Travel.pdf", "6 Lodging")]),
        ("merit-matrix", "manager"): Answer("q", "2.0% to 3.0% [1].", "answered",
                                            citations=[_citation("Merit.pdf", "6.2 Merit matrix")]),
        ("merit-matrix", "employee"): Answer("q", "I can't find that.", "not_found"),
        ("na-ceo", "employee"): Answer("q", "I can't find that.", "low_confidence"),
    }  # fmt: skip

    runs = run_answers(g, lambda question, caller: answers[(question.id, caller)])

    by = {(r.id, r.caller): r for r in runs}
    assert by[("trv-hotel", "employee")].correct
    assert by[("merit-matrix", "manager")].citation_hit
    assert by[("merit-matrix", "manager")].facts == [False]  # wrong numbers
    assert not by[("merit-matrix", "manager")].correct
    assert by[("merit-matrix", "employee")].hidden and by[("merit-matrix", "employee")].correct
    assert by[("na-ceo", "employee")].correct
    report = render_answers(g, runs, {})
    assert "missing facts: 2.5%" in report


def test_a_hidden_run_that_reveals_restricted_facts_is_wrong() -> None:
    g = _golden_for_runs()

    def ask(question: GoldenQuestion, caller: str) -> Answer:
        return Answer(
            "q",
            "It's 2.5% [1].",
            "answered",
            citations=[_citation("Merit.pdf", "6.2 Merit matrix")],
        )

    runs = run_answers(g.select(["merit-matrix"]), ask)

    hidden = next(r for r in runs if r.hidden)
    assert not hidden.correct and hidden.leaked == [("Merit.pdf", "6.2 Merit matrix")]
    assert "contains restricted facts: 2.5%" in render_answers(g, runs, {})
