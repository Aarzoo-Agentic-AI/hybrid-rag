"""Markdown reports for eval/results/ (retrieval.md, answers.md)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hybrid_rag.evaluation.golden import CATEGORIES, STYLES, GoldenSet
from hybrid_rag.evaluation.metrics import best_threshold, gate_rows, mean
from hybrid_rag.evaluation.runner import KS, AnswerRun, LeakRun, RetrievalRun, group, summarize


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}" if x in (0, 1) else f"{x:.1%}"


def _num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def _meta_lines(meta: dict[str, Any]) -> list[str]:
    return [f"- **{k}:** {v}" for k, v in meta.items()]


def overall_table(runs: Sequence[RetrievalRun], modes: Sequence[str]) -> list[str]:
    by_mode = group(runs, "mode")
    rows = []
    for mode in modes:
        s = summarize(by_mode.get(mode, []))
        rows.append(
            [f"`{mode}`", *(_pct(s[f"recall@{k}"]) for k in KS), _num(s["mrr"]),
             f"{s['median_latency_s']:.2f} s" if s["median_latency_s"] is not None else "-"]
        )  # fmt: skip
    return _table(["Mode", *(f"Recall@{k}" for k in KS), "MRR", "Median latency"], rows)


def render_retrieval(
    golden: GoldenSet,
    runs: Sequence[RetrievalRun],
    leaks: Sequence[LeakRun],
    modes: Sequence[str],
    meta: dict[str, Any],
    *,
    gate_mode: str = "hybrid_rerank",
    current_threshold: float | None = None,
    focus_k: int = 5,
) -> str:
    answerable = [q for q in golden.questions if q.answerable]
    out = ["# Retrieval evaluation", "", *_meta_lines(meta), ""]
    out += [
        "## Overall",
        "",
        f"{len(answerable)} answerable questions, each asked as its `caller`. A question with "
        "several expected sections scores the share it found. Ranks are per chunk, as sent to "
        "the LLM.",
        "",
        *overall_table(runs, modes),
        "",
    ]

    for key, values, title in (
        ("category", CATEGORIES, "category"),
        ("style", STYLES, "question style"),
    ):
        out += [f"## Recall@{focus_k} by {title}", ""]
        rows = []
        for value in values:
            if value == "unanswerable":
                continue
            n = sum(1 for q in answerable if getattr(q, key) == value)
            if not n:
                continue
            cells = []
            for mode in modes:
                sel = [r for r in runs if r.mode == mode and getattr(r, key) == value]
                cells.append(_pct(mean(r.recall(focus_k) for r in sel if r.answerable)))
            rows.append([value, str(n), *cells])
        out += [*_table([title.capitalize(), "n", *(f"`{m}`" for m in modes)], rows), ""]

    out += ["## Access: restricted sources", ""]
    if leaks:
        leaked = [lk for lk in leaks if lk.leaked]
        checks = len({(lk.id, lk.caller) for lk in leaks})
        out.append(
            f"{checks} restricted questions were also asked by a caller who may not read their "
            f"sources, in every mode ({len(leaks)} searches). "
            + (
                "**No chunk from a restricted source was returned.**"
                if not leaked
                else f"**{len(leaked)} searches returned restricted chunks:**"
            )
        )
        for lk in leaked:
            out.append(f"- `{lk.id}` as {lk.caller} ({lk.mode}): {lk.leaked}")
    else:
        out.append("No question has `hidden_from`.")
    out.append("")

    gate = [r for r in runs if r.mode == gate_mode]
    if gate and any(r.top_rerank_score is not None for r in gate):
        pos = [r.top_rerank_score for r in gate if r.answerable]
        neg = [r.top_rerank_score for r in gate if not r.answerable]
        out += [
            f"## Confidence gate (`{gate_mode}`)",
            "",
            "The answer service skips the LLM when the best reranker score is below "
            "`answer.min_rerank_score`. *Kept*: answerable questions that still reach the LLM. "
            "*Refused*: unanswerable questions the gate stops (the LLM can still refuse the rest "
            "with its not-found answer).",
            "",
        ]
        thresholds = sorted({float(t) for t in range(-11, 1)} | ({current_threshold} - {None}))
        rows = []
        for row in gate_rows(pos, neg, thresholds):
            mark = " (current)" if row["threshold"] == current_threshold else ""
            rows.append([f"{row['threshold']:g}{mark}", _pct(row["kept"]), _pct(row["refused"])])
        out += _table(["Threshold", "Kept", "Refused"], rows)
        best = best_threshold(pos, neg)
        if best:
            out += [
                "",
                f"Best balanced threshold: **{best['threshold']:g}** (kept {_pct(best['kept'])}, "
                f"refused {_pct(best['refused'])}).",
            ]
        lo_pos = min((s for s in pos if s is not None), default=None)
        hi_neg = max((s for s in neg if s is not None), default=None)
        out += [
            f"Lowest answerable score: {_num(lo_pos, 2)}; highest unanswerable score: "
            f"{_num(hi_neg, 2)}.",
            "",
        ]
        out += ["Unanswerable questions, best score:", ""]
        for r in sorted((r for r in gate if not r.answerable), key=lambda r: r.id):
            top = r.labels[0][1] if r.labels else "nothing"
            out.append(f"- `{r.id}`: {_num(r.top_rerank_score, 2)} ({top})")
        out.append("")

    best_mode = max(modes, key=lambda m: summarize(group(runs, "mode").get(m, []))["mrr"] or 0)
    misses = [r for r in runs if r.mode == best_mode and r.answerable and r.recall(focus_k) < 1]
    by_id = {q.id: q for q in golden.questions}
    out += [f"## Misses in `{best_mode}` (Recall@{focus_k} below 100%)", ""]
    if not misses:
        out.append("None.")
    for r in misses:
        q = by_id[r.id]
        want = "; ".join(
            " or ".join(f"{s}::{p}" for s, p in sorted(req.options)) for req in q.expected
        )
        got = ", ".join(f"{s}::{p}" for s, p in r.labels[:3]) or "nothing"
        found = ", ".join("-" if rank is None else f"#{rank}" for rank in r.ranks)
        out += [f"- **`{r.id}`** ({r.category}): {q.retrieval_query}",
                f"  - expected: {want} (found at {found})", f"  - top 3: {got}"]  # fmt: skip
    out.append("")
    return "\n".join(out)


def render_answers(golden: GoldenSet, runs: Sequence[AnswerRun], meta: dict[str, Any]) -> str:
    out = ["# Answer evaluation", "", *_meta_lines(meta), ""]
    main = [r for r in runs if not r.hidden]
    hidden = [r for r in runs if r.hidden]
    answerable = [r for r in main if r.category != "unanswerable"]
    unanswerable = [r for r in main if r.category == "unanswerable"]
    out += [
        "## Summary",
        "",
        *_table(
            ["Check", "Result"],
            [
                [
                    "Answerable: answered, cites an expected section, contains every fact",
                    f"{sum(r.correct for r in answerable)}/{len(answerable)}",
                ],
                [
                    "Answerable: answered",
                    f"{sum(r.status == 'answered' for r in answerable)}/{len(answerable)}",
                ],
                [
                    "Answerable: cites an expected section",
                    f"{sum(r.citation_hit for r in answerable)}/{len(answerable)}",
                ],
                [
                    "Answerable: contains every fact",
                    f"{sum(bool(r.facts) and all(r.facts) for r in answerable)}/{len(answerable)}",
                ],
                [
                    "Unanswerable: refused",
                    f"{sum(r.correct for r in unanswerable)}/{len(unanswerable)}",
                ],
                [
                    "Restricted questions asked without access: nothing revealed",
                    f"{sum(r.correct for r in hidden)}/{len(hidden)}",
                ],
                ["Failed over to another provider", str(sum(r.failed_over for r in runs))],
            ],
        ),
        "",
        "## By category",
        "",
    ]
    rows = []
    for cat in CATEGORIES:
        sel = [r for r in main if r.category == cat]
        if sel:
            rows.append([cat, f"{sum(r.correct for r in sel)}/{len(sel)}"])
    out += [*_table(["Category", "Correct"], rows), ""]

    by_id = {q.id: q for q in golden.questions}
    wrong = [r for r in runs if not r.correct]
    out += ["## Not correct", ""]
    if not wrong:
        out.append("None.")
    for r in wrong:
        q = by_id[r.id]
        pairs = list(zip(q.facts, r.facts, strict=True))
        why = []
        if r.hidden:
            why.append(f"asked as {r.caller}, who may not read the sources")
            if r.leaked:
                why.append(f"cited {r.leaked}")
            revealed = [" | ".join(f) for f, ok in pairs if ok]
            if revealed:
                why.append("contains restricted facts: " + "; ".join(revealed))
        else:
            why.append(f"status {r.status}")
            if q.answerable and not r.citation_hit:
                why.append("no expected citation")
            missing = [" | ".join(f) for f, ok in pairs if not ok]
            if missing:
                why.append("missing facts: " + "; ".join(missing))
        text = r.text.replace("\n", " ")
        out += [f"- **`{r.id}`** ({', '.join(why)})", f"  - {q.question}",
                f"  - answer: {text[:400]}{'…' if len(text) > 400 else ''}"]  # fmt: skip
    out.append("")
    return "\n".join(out)
