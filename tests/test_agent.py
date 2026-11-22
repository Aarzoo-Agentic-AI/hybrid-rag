"""Agentic retrieval (LangGraph): the loop, its limits, and that it never widens access."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from hybrid_rag.access import Principal
from hybrid_rag.api import create_app
from hybrid_rag.generation.agent import build_agent, parse_grade, run_agent
from tests.llm_fakes import Script, make_failover
from tests.test_api import Env

SUFFICIENT = '{"sufficient": true, "missing": "", "queries": []}'


def need(*queries: str) -> str:
    quoted = ", ".join(f'"{q}"' for q in queries)
    return f'{{"sufficient": false, "missing": "more", "queries": [{quoted}]}}'


def grader(*replies: str):
    script = Script(list(replies))
    llm, _, _ = make_failover(script, max_attempts=1)
    return llm, script


def doc(chunk_id: str) -> Document:
    return Document(page_content=f"text {chunk_id}", metadata={"chunk_id": chunk_id})


class FakeSearch:
    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[Document]:
        self.queries.append(query)
        return [doc(c) for c in self.results.get(query, [])]


def agent_for(search: FakeSearch, *replies: str, max_rounds: int = 2):
    llm, script = grader(*replies)
    agent = build_agent(
        search, llm, lambda docs: "\n".join(d.page_content for d in docs), max_rounds=max_rounds
    )
    return agent, script


# ------------------------------------------------------------------ the loop


def test_sufficient_sources_need_one_search() -> None:
    search = FakeSearch({"q": ["a", "b"]})
    agent, script = agent_for(search, SUFFICIENT)

    state = run_agent(agent, "q")

    assert search.queries == ["q"] and state["rounds"] == 1 and script.calls == 1
    assert [d.metadata["chunk_id"] for d in state["docs"]] == ["a", "b"]


def test_missing_information_triggers_new_queries_and_merges_results() -> None:
    search = FakeSearch({"q": ["a", "b"], "sabbatical years": ["b", "c"]})
    agent, script = agent_for(search, need("sabbatical years"), SUFFICIENT)

    state = run_agent(agent, "q")

    assert search.queries == ["q", "sabbatical years"] and state["asked"] == [
        "q",
        "sabbatical years",
    ]
    assert [d.metadata["chunk_id"] for d in state["docs"]] == [
        "a",
        "b",
        "c",
    ]  # first round first, no duplicates
    assert script.calls == 2


def test_the_number_of_extra_searches_is_capped() -> None:
    search = FakeSearch({})
    agent, script = agent_for(search, need("x1"), need("x2"), need("x3"), max_rounds=2)

    state = run_agent(agent, "q")

    assert search.queries == ["q", "x1", "x2"]  # the first search plus two extra
    assert state["rounds"] == 3 and script.calls == 2  # no grading after the last round


def test_repeated_queries_are_not_run_again() -> None:
    search = FakeSearch({"q": ["a"]})
    agent, script = agent_for(search, need("Q", "q "))

    state = run_agent(agent, "q")

    assert search.queries == ["q"] and state["rounds"] == 1 and script.calls == 1


@pytest.mark.parametrize(
    ("text", "sufficient", "queries"),
    [
        (SUFFICIENT, True, []),
        (need("a", "b"), False, ["a", "b"]),
        ('Here: {"sufficient": false, "queries": "one"}', False, ["one"]),
        ('{"sufficient": "no"}', True, []),  # only an explicit false means "search again"
        ("not json", True, []),
        ("[1, 2]", True, []),
    ],
)
def test_grades_are_parsed_defensively(text: str, sufficient: bool, queries: list[str]) -> None:
    grade = parse_grade(text)

    assert (grade.sufficient, grade.queries) == (sufficient, queries)


# ------------------------------------------------------------------ in the answer service


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path, ["The answer is in the handbook [1]."] * 20)


def test_the_agentic_strategy_answers_with_the_extra_sources(env) -> None:
    env.services.answers.agent_llm, script = grader(need("QXMANAGER duties"), SUFFICIENT)
    hr = Principal("acme", "u-2", "confidential", groups=["hr-compensation"])

    answer = env.services.answers.answer("QXHANDBOOK rules?", hr, strategy="agentic")

    assert answer.strategy == "agentic" and answer.queries == [
        "QXHANDBOOK rules?",
        "QXMANAGER duties",
    ]
    assert answer.status == "answered" and script.calls == 2
    assert {s.source for s in answer.sources} >= {"handbook.pdf"}


def test_agent_queries_never_widen_what_the_caller_may_read(env) -> None:
    """The grader asks for a restricted document's term; the employee still can't get it."""
    env.services.answers.agent_llm, _ = grader(
        need("QXSALARY bands", "QXPERSONAL file"), SUFFICIENT
    )
    employee = Principal("acme", "u-1", "internal")

    answer = env.services.answers.answer("QXHANDBOOK rules?", employee, strategy="agentic")

    assert "QXSALARY bands" in answer.queries  # the search ran...
    assert {s.source for s in answer.sources}.isdisjoint({"salary_bands.pdf", "personal_file.pdf"})


def test_single_is_the_default_and_agentic_needs_an_agent_llm(env) -> None:
    employee = Principal("acme", "u-1", "internal")

    answer = env.services.answers.answer("QXHANDBOOK rules?", employee)

    assert answer.strategy == "single" and answer.queries == ["QXHANDBOOK rules?"]
    with pytest.raises(ValueError, match="agent LLM"):
        env.services.answers.answer("QXHANDBOOK rules?", employee, strategy="agentic")


def test_the_api_takes_a_strategy_and_shows_the_searches_in_debug(env) -> None:
    env.services.answers.agent_llm, _ = grader(need("QXMANAGER duties"), SUFFICIENT)

    with TestClient(create_app(env.services, configure_logging=False)) as client:
        r = client.post(
            "/query",
            json={"question": "QXHANDBOOK rules?", "strategy": "agentic", "debug": True},
            headers=env.headers("employee"),
        )
        streamed = client.post(
            "/query",
            json={"question": "QXHANDBOOK rules?", "strategy": "single", "stream": True},
            headers=env.headers("employee"),
        )

    debug = r.json()["debug"]
    assert r.status_code == 200 and debug["strategy"] == "agentic" and len(debug["queries"]) == 2
    assert streamed.status_code == 200 and "event: answer" in streamed.text
    entries = [e for e in env.audit() if e["route"] == "/query"]
    assert [(e["strategy"], e["searches"]) for e in entries] == [("agentic", 2), ("single", 1)]
    assert all("QXMANAGER" not in str(e) for e in entries)  # query text stays out of the audit log
