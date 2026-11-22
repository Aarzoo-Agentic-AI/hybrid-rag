"""Ask the HR policy assistant a question, as a given caller.

Usage:
    uv run python scripts/ask.py "How many PTO days do I get after 4 years?"
    uv run python scripts/ask.py "What merit increase does a Meets rating get?" --as manager
    uv run python scripts/ask.py "Can I fly premium economy?" --stream
    uv run python scripts/ask.py "And for flights to Asia?" --history "Can I fly premium economy?"
    uv run python scripts/ask.py "What is the hotel cap in Boston?" --debug   # sources and scores

Callers (--as) are demo principals in documents.tenant_id; in the API (step 6) principals come
from the caller's identity. Each question costs one LLM call (two with --history). With
LangSmith tracing on, the question (redacted) and the retrieved policy text are traced.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from hybrid_rag.access import Principal
from hybrid_rag.config import get_settings
from hybrid_rag.generation import Answer, AnswerService
from hybrid_rag.llm import LLMError
from hybrid_rag.retrieval import MODES
from hybrid_rag.store import StoreError, build_document_index
from hybrid_rag.tracing import flush_traces

DEMO_CALLERS = {
    "employee": {"user_id": "u-demo-employee", "clearance": "internal"},
    "manager": {
        "user_id": "u-demo-manager",
        "clearance": "confidential",
        "roles": ["people_manager"],
    },
    "hr-comp": {"user_id": "u-demo-hr", "clearance": "confidential", "groups": ["hr-compensation"]},
}


def print_answer(answer: Answer, *, debug: bool, streamed: bool) -> None:
    if not streamed:
        print(answer.text)
    print(f"\n[{answer.status}{'' if answer.grounded else ', not grounded'}]", end=" ")
    if answer.provider:
        p = answer.provider
        failover = f", failed over after {p['attempts'] - 1} attempt(s)" if p["failed_over"] else ""
        print(f"answered by {p['provider']}/{p['model']} in {p['latency_s']}s{failover}")
    else:
        print("no LLM call")
    if answer.rewritten:
        print(f"Rewritten question: {answer.question}")
    if answer.redactions:
        print(f"Removed from the question: {answer.redactions}")
    for c in answer.citations:
        effective = f", effective {c.effective_date}" if c.effective_date else ""
        print(f"  [{c.number}] {c.label}{effective}")
    if debug:
        print("\nSources in the prompt:")
        for s in answer.sources:
            score = f"{s.rerank_score:.2f}" if s.rerank_score is not None else "-"
            print(
                f"  [{s.number}] {'cited ' if s.cited else '      '} rerank {score:>6}  "
                f"{s.source} (chunk {s.chunk_id[:8]}, access v{s.access_version})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--as", dest="caller", choices=DEMO_CALLERS, default="employee")
    parser.add_argument("--mode", choices=MODES, help="retrieval mode (default: retrieval.mode)")
    parser.add_argument("--history", action="append", default=[], help="an earlier question")
    parser.add_argument("--stream", action="store_true", help="print the answer as it's written")
    parser.add_argument("--debug", action="store_true", help="show the sources and scores")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")  # section signs and quotes in citations

    settings = get_settings()
    principal = Principal(settings.documents.tenant_id, **DEMO_CALLERS[args.caller])
    history = [("human", h) for h in args.history]
    try:
        service = AnswerService.from_settings(settings, build_document_index(settings))
    except (StoreError, LLMError) as exc:
        print(f"error: {exc}")
        return 1

    started = time.perf_counter()
    try:
        if args.stream:
            answer = None
            for event in service.stream(args.question, principal, history=history, mode=args.mode):
                if event["type"] == "token":
                    print(event["text"], end="", flush=True)
                else:
                    answer = event["answer"]
            if answer and not answer.provider:
                print(answer.text, end="")  # answered without the LLM: nothing was streamed
            print()
            assert answer is not None
        else:
            answer = service.answer(args.question, principal, history=history, mode=args.mode)
    except LLMError as exc:
        print(f"error: {exc}")
        return 1
    print_answer(answer, debug=args.debug, streamed=args.stream)
    print(f"({time.perf_counter() - started:.1f}s in total)")
    flush_traces()
    return 0


if __name__ == "__main__":
    sys.exit(main())
