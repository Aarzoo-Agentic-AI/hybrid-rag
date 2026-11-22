"""Run the golden set through retrieval (every mode) and, optionally, through the answer service.

Both runners take a callable rather than building the pipeline, so tests pass fakes and the
script (eval/evaluate.py) passes the real retrievers and ``AnswerService``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.documents import Document

from hybrid_rag.evaluation.golden import GoldenQuestion, GoldenSet, Label
from hybrid_rag.evaluation.metrics import (
    facts_found,
    first_ranks,
    mean,
    median,
    recall_at,
    reciprocal_rank,
)
from hybrid_rag.generation.types import Answer

KS = (1, 3, 5, 10)
REFUSALS = ("not_found", "low_confidence", "no_sources")

Search = Callable[[str, str, str], list[Document]]  # (query, caller, mode) -> ranked chunks
Ask = Callable[[GoldenQuestion, str], Answer]  # (question, caller) -> answer


def label(doc: Document) -> Label:
    return (str(doc.metadata.get("source")), str(doc.metadata.get("section_path")))


# ------------------------------------------------------------------ retrieval


@dataclass
class RetrievalRun:
    id: str
    category: str
    style: str
    caller: str
    mode: str
    query: str
    labels: list[Label]
    ranks: list[int | None]  # per requirement; empty for unanswerable questions
    latency_s: float
    top_rerank_score: float | None  # hybrid_rerank only: what the confidence gate looks at

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    def recall(self, k: int) -> float:
        return recall_at(self.ranks, k)

    @property
    def rr(self) -> float:
        return reciprocal_rank(self.ranks)


@dataclass
class LeakRun:
    """A question asked by a caller who must not see its sources."""

    id: str
    caller: str
    mode: str
    leaked: list[Label]  # retrieved chunks from the question's expected sources


def run_retrieval(
    golden: GoldenSet,
    modes: Sequence[str],
    search: Search,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[RetrievalRun], list[LeakRun]]:
    runs: list[RetrievalRun] = []
    leaks: list[LeakRun] = []
    for mode in modes:
        for q in golden.questions:
            started = time.perf_counter()
            docs = search(q.retrieval_query, q.caller, mode)
            latency = time.perf_counter() - started
            labels = [label(d) for d in docs]
            top = docs[0].metadata.get("rerank_score") if docs else None
            runs.append(
                RetrievalRun(
                    id=q.id,
                    category=q.category,
                    style=q.style,
                    caller=q.caller,
                    mode=mode,
                    query=q.retrieval_query,
                    labels=labels,
                    ranks=first_ranks(labels, q.expected),
                    latency_s=latency,
                    top_rerank_score=float(top) if top is not None else None,
                )
            )
            for caller in q.hidden_from:
                hidden = q.expected_sources
                seen = [label(d) for d in search(q.retrieval_query, caller, mode)]
                leaks.append(LeakRun(q.id, caller, mode, [lab for lab in seen if lab[0] in hidden]))
        if progress:
            progress(f"{mode}: {len(golden.questions)} questions")
    return runs, leaks


def summarize(runs: Iterable[RetrievalRun], ks: Sequence[int] = KS) -> dict[str, Any]:
    """Mean Recall@k and MRR over answerable questions, and the median latency over all."""
    runs = list(runs)
    scored = [r for r in runs if r.answerable]
    out: dict[str, Any] = {"n": len(scored)}
    for k in ks:
        out[f"recall@{k}"] = mean(r.recall(k) for r in scored)
    out["mrr"] = mean(r.rr for r in scored)
    out["median_latency_s"] = median(r.latency_s for r in runs)
    return out


def group(runs: Iterable[RetrievalRun], key: str) -> dict[str, list[RetrievalRun]]:
    groups: dict[str, list[RetrievalRun]] = {}
    for r in runs:
        groups.setdefault(getattr(r, key), []).append(r)
    return groups


# ------------------------------------------------------------------ answers


@dataclass
class AnswerRun:
    id: str
    category: str
    caller: str
    hidden: bool  # asked by a caller in hidden_from: must not reveal the sources
    status: str
    grounded: bool
    text: str
    rewritten_question: str
    facts: list[bool]
    cited: list[Label]
    citation_hit: bool  # a citation meets at least one requirement
    leaked: list[Label]  # hidden runs: cited sections from the hidden sources
    latency_s: float
    provider: str | None
    model: str | None
    failed_over: bool
    searches: int = 1  # retrieval queries run (the agentic strategy can add some)
    correct: bool = field(default=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_answers(
    golden: GoldenSet, ask: Ask, *, progress: Callable[[str], None] | None = None
) -> list[AnswerRun]:
    runs: list[AnswerRun] = []
    jobs = [(q, q.caller, False) for q in golden.questions]
    jobs += [(q, caller, True) for q in golden.questions for caller in q.hidden_from]
    for i, (q, caller, hidden) in enumerate(jobs, start=1):
        started = time.perf_counter()
        answer = ask(q, caller)
        runs.append(_score(q, caller, hidden, answer, time.perf_counter() - started))
        if progress:
            progress(f"answers {i}/{len(jobs)}: {q.id} as {caller}: {runs[-1].status}")
    return runs


def _score(q: GoldenQuestion, caller: str, hidden: bool, a: Answer, latency: float) -> AnswerRun:
    cited = [(c.source, c.section) for c in a.citations]
    hit = any(req.met_by(lab) for req in q.expected for lab in cited)
    found = facts_found(a.text, q.facts) if a.status == "answered" else [False] * len(q.facts)
    leaked = [lab for lab in cited if lab[0] in q.expected_sources] if hidden else []
    provider = a.provider or {}
    correct = _correct(q, hidden, a.status, hit, found, leaked)
    return AnswerRun(
        id=q.id,
        category=q.category,
        caller=caller,
        hidden=hidden,
        status=a.status,
        grounded=a.grounded,
        text=a.text,
        rewritten_question=a.question,
        facts=found,
        cited=cited,
        citation_hit=hit,
        leaked=leaked,
        latency_s=latency,
        provider=provider.get("provider"),
        model=provider.get("model"),
        failed_over=bool(provider.get("failed_over")),
        searches=max(1, len(a.queries)),
        correct=correct,
    )


def _correct(
    q: GoldenQuestion, hidden: bool, status: str, hit: bool, found: list[bool], leaked: list[Label]
) -> bool:
    if hidden:
        # The caller may not read the sources: any refusal is right, and so is an answer that
        # neither cites them nor contains their facts.
        return not leaked and not any(found)
    if not q.answerable:
        return status in REFUSALS
    return status == "answered" and hit and all(found)


def rescore(golden: GoldenSet, runs: Iterable[AnswerRun]) -> list[AnswerRun]:
    """Check saved answers again against ``golden`` (after correcting a fact or an expected
    section), without asking the LLM again. Answers to questions no longer in the set are
    dropped."""
    by_id = {q.id: q for q in golden.questions}
    out = []
    for r in runs:
        q = by_id.get(r.id)
        if q is None:
            continue
        found = facts_found(r.text, q.facts) if r.status == "answered" else [False] * len(q.facts)
        hit = any(req.met_by(lab) for req in q.expected for lab in r.cited)
        leaked = [lab for lab in r.cited if lab[0] in q.expected_sources] if r.hidden else []
        r.facts, r.citation_hit, r.leaked = found, hit, leaked
        r.correct = _correct(q, r.hidden, r.status, hit, found, leaked)
        out.append(r)
    return out


def load_answer_runs(data: dict[str, Any]) -> list[AnswerRun]:
    """``answers.json`` back into runs (labels are stored as lists)."""
    runs = []
    for item in data["runs"]:
        item = {
            **item,
            "cited": [tuple(x) for x in item["cited"]],
            "leaked": [tuple(x) for x in item["leaked"]],
        }
        runs.append(AnswerRun(**item))
    return runs
