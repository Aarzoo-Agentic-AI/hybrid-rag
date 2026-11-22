"""The runtime report: what live traffic looks like, from the audit and feedback logs.

Per day: questions, answer status, ungrounded answers, failovers, latency, user ratings and
the faithfulness check (``runtime_eval``). Then the answers to look at: thumbs-down ratings and
low faithfulness scores, with thumbs-down questions drafted as golden-set candidates (their
question text is stored, masked, only with a thumbs-down).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("answered", "not_found", "low_confidence", "no_sources")


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    """Every readable line of a JSONL log (a missing file is empty; bad lines are skipped)."""
    if path is None or not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def percentile(values: Sequence[float], p: float) -> float | None:
    """Nearest-rank percentile (p from 0 to 100)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


@dataclass
class Day:
    date: str
    queries: int = 0
    statuses: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STATUSES, 0))
    ungrounded: int = 0
    failed_over: int = 0
    latencies: list[float] = field(default_factory=list)
    up: int = 0
    down: int = 0
    scores: list[float] = field(default_factory=list)
    low: int = 0


@dataclass
class RuntimeReport:
    days: list[Day]
    thumbs_down: list[dict[str, Any]]
    low_faithfulness: list[dict[str, Any]]
    low_threshold: float


def build_report(
    audit: Iterable[dict[str, Any]],
    feedback: Iterable[dict[str, Any]],
    *,
    since: str | None = None,
    low_threshold: float = 0.8,
) -> RuntimeReport:
    """``since``: an ISO date (YYYY-MM-DD); earlier lines are left out."""
    days: dict[str, Day] = {}

    def day(row: dict[str, Any]) -> Day | None:
        date = str(row.get("ts", ""))[:10]
        if not date or (since and date < since):
            return None
        return days.setdefault(date, Day(date))

    low: list[dict[str, Any]] = []
    for row in audit:
        d = day(row)
        if d is None:
            continue
        if row.get("route") == "/query":
            d.queries += 1
            status = row.get("answer_status")
            if status in d.statuses:
                d.statuses[status] += 1
            if status == "answered" and not row.get("grounded"):
                d.ungrounded += 1
            if row.get("failed_over"):
                d.failed_over += 1
            if isinstance(row.get("latency_ms"), int | float):
                d.latencies.append(float(row["latency_ms"]))
        elif row.get("route") == "runtime_eval" and isinstance(row.get("score"), int | float):
            d.scores.append(float(row["score"]))
            if row["score"] < low_threshold:
                d.low += 1
                low.append(row)

    down: list[dict[str, Any]] = []
    for row in feedback:
        d = day(row)
        if d is None:
            continue
        if row.get("rating") == "up":
            d.up += 1
        elif row.get("rating") == "down":
            d.down += 1
            down.append(row)

    return RuntimeReport(
        days=[days[k] for k in sorted(days)],
        thumbs_down=down,
        low_faithfulness=sorted(low, key=lambda r: r["score"]),
        low_threshold=low_threshold,
    )


def _pct(n: int, total: int) -> str:
    return f"{n / total:.0%}" if total else "-"


def render_report(report: RuntimeReport) -> str:
    out = ["# Runtime report", ""]
    if not report.days:
        return "\n".join([*out, "No requests in the logs for this period.", ""])
    out += [
        "| Day | Questions | Answered | Not found | Low confidence | No sources | Ungrounded "
        "| Failovers | p50 latency | p95 latency | Rated up | Rated down | Down rate | Checked "
        f"| Faithfulness | Below {report.low_threshold:g} |",
        "|" + "---|" * 16,
    ]
    for d in report.days:
        rated = d.up + d.down
        p50, p95 = percentile(d.latencies, 50), percentile(d.latencies, 95)
        mean = sum(d.scores) / len(d.scores) if d.scores else None
        out.append(
            "| "
            + " | ".join(
                [
                    d.date,
                    str(d.queries),
                    *(_pct(d.statuses[s], d.queries) for s in STATUSES),
                    _pct(d.ungrounded, d.statuses["answered"]),
                    str(d.failed_over),
                    f"{p50 / 1000:.1f} s" if p50 is not None else "-",
                    f"{p95 / 1000:.1f} s" if p95 is not None else "-",
                    str(d.up),
                    str(d.down),
                    _pct(d.down, rated),
                    str(len(d.scores)),
                    f"{mean:.2f}" if mean is not None else "-",
                    str(d.low),
                ]
            )
            + " |"
        )
    out += [
        "",
        "*Ungrounded*: answered without a valid citation, as a share of answered questions. "
        "*Checked*: answers scored by the faithfulness check (a sample).",
        "",
        f"## Thumbs down ({len(report.thumbs_down)})",
        "",
    ]
    if not report.thumbs_down:
        out.append("None.")
    for row in report.thumbs_down:
        comment = f' — "{row["comment"]}"' if row.get("comment") else ""
        out.append(
            f"- `{row.get('request_id')}` ({str(row.get('ts', ''))[:16]}, "
            f"{row.get('answer_status')}): {row.get('question') or '(no question)'}{comment}"
        )
    out += [
        "",
        f"## Faithfulness below {report.low_threshold:g} ({len(report.low_faithfulness)})",
        "",
    ]
    if not report.low_faithfulness:
        out.append("None.")
    for row in report.low_faithfulness:
        claims = "; ".join(row.get("unsupported") or []) or "no claims listed"
        out.append(f"- `{row.get('request_id')}`: {row['score']:.2f}. Unsupported: {claims}")
    candidates = [r for r in report.thumbs_down if r.get("question")]
    if candidates:
        out += [
            "",
            "## Golden-set candidates",
            "",
            "Thumbs-down questions, drafted for `eval/golden.yaml`. Fill in the category, the "
            "expected sections and the facts before adding them.",
            "",
            "```yaml",
        ]
        for row in candidates:
            question = str(row["question"]).replace('"', '\\"')
            out += [
                f"- id: fb-{str(row.get('request_id', ''))[:8].lower()}",
                f'  question: "{question}"',
                "  category: fact            # TODO",
                "  style: paraphrase",
                "  expected: []              # TODO: alias::section path",
                "  facts: []                 # TODO",
            ]
        out.append("```")
    out.append("")
    return "\n".join(out)
