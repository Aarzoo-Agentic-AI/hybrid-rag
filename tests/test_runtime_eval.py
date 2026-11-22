"""Runtime evaluation: user feedback, the sampled faithfulness check and the runtime report.

The API runs with the same in-memory setup as tests/test_api.py; LangSmith is a fake client
that records the feedback it would receive.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hybrid_rag.api import create_app
from hybrid_rag.api.audit import AuditLog
from hybrid_rag.api.feedback import (
    FeedbackLog,
    LangSmithFeedback,
    RecentRequest,
    RecentRequests,
    run_id_for,
)
from hybrid_rag.api.runtime_eval import RuntimeEvaluator
from hybrid_rag.evaluation.runtime_report import build_report, percentile, render_report
from hybrid_rag.generation import Answer
from hybrid_rag.generation.judge import parse_judgement
from tests.llm_fakes import Script, make_failover
from tests.test_api import Env

JUDGED = '{"score": 0.5, "unsupported": ["The cap is $400"]}'


PROJECT_ID = "6f1c0d9e-0000-4000-8000-000000000001"


class FakeLangSmith:
    """Records the feedback a LangSmith client would send."""

    def __init__(self) -> None:
        self.feedback: list[dict[str, Any]] = []
        self.project_lookups = 0

    def read_project(self, *, project_name: str) -> Any:
        self.project_lookups += 1
        return type("Project", (), {"id": PROJECT_ID})()

    def create_feedback(self, run_id: uuid.UUID, **kwargs: Any) -> None:
        self.feedback.append({"run_id": run_id, **kwargs})


def sent(request_id: str, key: str, score: float, comment: str | None) -> dict[str, Any]:
    """What the API should send: on the answer trace, with its project's ID."""
    run_id = run_id_for(request_id)
    return {"run_id": run_id, "key": key, "score": score, "comment": comment,
            "trace_id": run_id, "session_id": PROJECT_ID}  # fmt: skip


def evaluator(audit: AuditLog, replies: list, *, rate: float = 1.0, **kwargs: Any):
    judge, _, _ = make_failover(Script(replies), max_attempts=1)
    return RuntimeEvaluator(judge, audit, sample_rate=rate, rng=random.Random(0), **kwargs)


def answered(context: str = "[1] The cap is $300.") -> Answer:
    return Answer(question="q", text="The cap is $400 [1].", status="answered", context=context)


@pytest.fixture
def env(tmp_path: Path) -> Env:
    e = Env(tmp_path, ["The answer is in the handbook [1]."] * 20)
    e.feedback_path = tmp_path / "feedback.jsonl"
    e.services.feedback = FeedbackLog(e.feedback_path)
    e.fake = FakeLangSmith()
    e.services.trace_client = e.fake
    e.services.langsmith = LangSmithFeedback(e.fake, "test-project")
    return e


@pytest.fixture
def client(env: Env):
    with TestClient(create_app(env.services, configure_logging=False)) as c:
        yield c


def ask(client, env: Env, who: str = "employee", question: str = "QXHANDBOOK rules?") -> str:
    r = client.post("/query", json={"question": question}, headers=env.headers(who))
    assert r.status_code == 200, r.text
    return r.json()["request_id"]


def feedback_log(env: Env) -> list[dict]:
    return [json.loads(line) for line in env.feedback_path.read_text().splitlines()]


# ------------------------------------------------------------------ feedback


def test_the_asker_can_rate_an_answer_once(client, env) -> None:
    request_id = ask(client, env)

    r = client.post(
        "/feedback",
        json={"request_id": request_id, "rating": "up", "comment": "Clear, thanks"},
        headers=env.headers("employee"),
    )
    again = client.post(
        "/feedback",
        json={"request_id": request_id, "rating": "down"},
        headers=env.headers("employee"),
    )

    assert r.status_code == 200 and r.json() == {
        "request_id": request_id,
        "rating": "up",
        "recorded": True,
    }
    assert again.status_code == 409 and again.json()["error"]["code"] == "conflict"
    (row,) = feedback_log(env)
    assert row["rating"] == "up" and row["comment"] == "Clear, thanks" and "question" not in row
    assert any(
        e["route"] == "/feedback" and e["rated_request_id"] == request_id for e in env.audit()
    )
    assert env.fake.feedback == [sent(request_id, "user_rating", 1, "Clear, thanks")]


def test_a_thumbs_down_keeps_the_masked_question_and_comment(client, env) -> None:
    request_id = ask(client, env, question="QXHANDBOOK rules? I'm jane@corp.example")

    r = client.post(
        "/feedback",
        json={"request_id": request_id, "rating": "down", "comment": "Wrong. Call 415-555-0134"},
        headers=env.headers("employee"),
    )

    assert r.status_code == 200
    (row,) = feedback_log(env)
    assert "<EMAIL>" in row["question"] and "jane@corp.example" not in row["question"]
    assert "415-555-0134" not in row["comment"]
    assert env.fake.feedback[0]["score"] == 0


@pytest.mark.parametrize("who", ["hr", "globex-admin", "admin"])
def test_only_the_asker_can_rate_and_others_get_404(client, env, who: str) -> None:
    request_id = ask(client, env)

    r = client.post(
        "/feedback", json={"request_id": request_id, "rating": "down"}, headers=env.headers(who)
    )

    assert r.status_code == 404
    assert not env.feedback_path.exists()


def test_unknown_requests_and_bad_bodies_are_refused(client, env) -> None:
    unknown = client.post(
        "/feedback", json={"request_id": "nope", "rating": "up"}, headers=env.headers("employee")
    )
    bad = client.post(
        "/feedback", json={"request_id": "x", "rating": "meh"}, headers=env.headers("employee")
    )
    no_key = client.post("/feedback", json={"request_id": "x", "rating": "up"})

    assert (unknown.status_code, bad.status_code, no_key.status_code) == (404, 422, 401)


def test_recent_requests_forget_old_and_overflowing_entries(monkeypatch) -> None:
    recent = RecentRequests(capacity=2, window_s=60)
    for rid in ("a", "b", "c"):
        recent.remember(RecentRequest(rid, "acme", "u-1", "q", "answered", time.time()))

    assert recent.lookup("a", "acme", "u-1") is None  # pushed out
    assert recent.lookup("c", "acme", "u-1") is not None
    monkeypatch.setattr(time, "time", lambda: 10**10)  # far in the future
    assert recent.lookup("c", "acme", "u-1") is None  # outside the window


def test_run_ids_are_stable_per_request() -> None:
    assert run_id_for("abc") == run_id_for("abc") != run_id_for("abd")


def test_the_project_id_is_looked_up_once_and_errors_never_raise() -> None:
    fake = FakeLangSmith()
    sender = LangSmithFeedback(fake, "p")

    assert sender.send("a", "user_rating", 1) and sender.send("b", "user_rating", 0)
    assert fake.project_lookups == 1

    class Broken(FakeLangSmith):
        def read_project(self, *, project_name: str) -> Any:
            raise RuntimeError("no such project")

    broken = Broken()
    assert not LangSmithFeedback(broken, "p").send("a", "user_rating", 1)
    assert broken.feedback == []


# ------------------------------------------------------------------ the faithfulness check


def test_a_sampled_answer_is_judged_in_the_background(tmp_path: Path) -> None:
    audit, fake = AuditLog(tmp_path / "audit.jsonl"), FakeLangSmith()
    ev = evaluator(audit, [JUDGED], langsmith=LangSmithFeedback(fake, "p"))

    assert ev.maybe_check(request_id="r1", tenant_id="acme", user_id="u-1", answer=answered())
    assert ev.wait_idle()

    (row,) = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert row["route"] == "runtime_eval" and row["score"] == 0.5
    assert row["unsupported"] == ["The cap is $400"] and row["error"] is None
    assert fake.feedback == [sent("r1", "faithfulness", 0.5, "The cap is $400")]


def test_only_answered_and_sampled_questions_are_checked(tmp_path: Path) -> None:
    audit = AuditLog(tmp_path / "audit.jsonl")
    never = evaluator(audit, [JUDGED], rate=0.0)
    always = evaluator(audit, [JUDGED])
    refused = Answer(question="q", text="I can't find that.", status="not_found")

    assert not never.maybe_check(request_id="r", tenant_id="t", user_id="u", answer=answered())
    assert not always.maybe_check(request_id="r", tenant_id="t", user_id="u", answer=refused)
    assert not always.maybe_check(
        request_id="r", tenant_id="t", user_id="u", answer=answered(context="")
    )


def test_checks_beyond_max_pending_are_skipped(tmp_path: Path) -> None:
    ev = evaluator(AuditLog(None), [JUDGED] * 5, max_pending=1)
    ev._pending = 1  # one check already waiting

    assert not ev.maybe_check(request_id="r", tenant_id="t", user_id="u", answer=answered())
    assert ev.skipped == 1


@pytest.mark.parametrize("reply", ["Looks fine to me.", RuntimeError("judge down")])
def test_a_failed_check_is_recorded_not_raised(tmp_path: Path, reply) -> None:
    audit, fake = AuditLog(tmp_path / "audit.jsonl"), FakeLangSmith()
    ev = evaluator(audit, [reply], langsmith=LangSmithFeedback(fake, "p"))

    ev.maybe_check(request_id="r1", tenant_id="acme", user_id="u-1", answer=answered())
    assert ev.wait_idle()

    (row,) = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert row["score"] is None and row["error"]
    assert fake.feedback == []


def test_the_api_samples_answers_for_the_check(client, env) -> None:
    env.services.runtime_eval = evaluator(
        env.services.audit, [JUDGED], langsmith=env.services.langsmith
    )

    request_id = ask(client, env)
    assert env.services.runtime_eval.wait_idle()

    (row,) = [e for e in env.audit() if e["route"] == "runtime_eval"]
    assert row["request_id"] == request_id and row["user_id"] == "u-1" and row["score"] == 0.5


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"score": 1, "unsupported": []}', (1.0, [])),
        ('Sure: {"score": 0.25, "unsupported": "the 2027 date"} done', (0.25, ["the 2027 date"])),
        ('{"score": 1.5}', None),
        ('{"unsupported": []}', None),
        ("no json here", None),
    ],
)
def test_judge_replies_are_parsed_defensively(text: str, expected) -> None:
    result = parse_judgement(text)

    assert (None if result is None else (result.score, result.unsupported)) == expected


def test_the_sources_text_never_leaves_the_service() -> None:
    answer = answered()

    assert "context" not in answer.to_dict()


# ------------------------------------------------------------------ report


def row(ts: str, **fields: Any) -> dict[str, Any]:
    return {"ts": f"2026-11-{ts}", **fields}


def query_row(ts: str, status: str, latency_ms: int, **fields: Any) -> dict[str, Any]:
    return row(ts, route="/query", answer_status=status, latency_ms=latency_ms, **fields)


def test_the_report_summarises_days_and_drafts_golden_candidates() -> None:
    audit = [
        query_row("20T10:00", "answered", 7000, grounded=True),
        query_row("20T11:00", "answered", 9000, grounded=False, failed_over=True),
        query_row("20T12:00", "not_found", 3000),
        row("20T12:01", route="runtime_eval", request_id="r9", score=0.4, unsupported=["$400"]),
        query_row("19T09:00", "answered", 1, grounded=True),  # before --since
        "not a dict",
    ]
    feedback = [
        row("20T13:00", request_id="ABCDEF123", rating="down", question='Is "Tier 2" Denver?'),
        row("20T13:05", request_id="r2", rating="up"),
    ]

    report = build_report([a for a in audit if isinstance(a, dict)], feedback, since="2026-11-20")

    (day,) = report.days
    assert (day.queries, day.statuses["answered"], day.ungrounded, day.failed_over) == (3, 2, 1, 1)
    assert (day.up, day.down, day.scores, day.low) == (1, 1, [0.4], 1)
    text = render_report(report)
    assert "| 2026-11-20 | 3 | 67% | 33% |" in text and "| 7.0 s | 9.0 s |" in text
    assert "`r9`: 0.40. Unsupported: $400" in text
    assert '- id: fb-abcdef12\n  question: "Is \\"Tier 2\\" Denver?"' in text


def test_an_empty_report_says_so_and_percentiles_use_nearest_rank() -> None:
    assert "No requests" in render_report(build_report([], []))
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([1, 2, 3, 4], 95) == 4
    assert percentile([], 50) is None
