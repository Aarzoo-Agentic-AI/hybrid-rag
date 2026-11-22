"""Summarise live traffic from the audit and feedback logs (runtime evaluation).

Usage:
    uv run python scripts/runtime_report.py                     # everything, to the terminal
    uv run python scripts/runtime_report.py --since 2026-11-01 --out logs/runtime_report.md
    docker compose exec api python scripts/runtime_report.py    # inside the Docker stack

Per day: questions, answer status, ungrounded answers, failovers, latency (p50/p95), user
ratings and faithfulness scores; then the thumbs-down answers (drafted as golden-set
candidates) and the low faithfulness scores. Reads api.audit_log and api.feedback_log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hybrid_rag.config import get_settings
from hybrid_rag.evaluation.runtime_report import build_report, read_jsonl, render_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", metavar="YYYY-MM-DD", help="leave out earlier days")
    parser.add_argument("--low", type=float, default=0.8, help="faithfulness threshold to list")
    parser.add_argument("--out", help="write the report here instead of printing it")
    args = parser.parse_args()

    settings = get_settings()
    report = build_report(
        read_jsonl(settings.api.audit_log),
        read_jsonl(settings.api.feedback_log),
        since=args.since,
        low_threshold=args.low,
    )
    text = render_report(report)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
