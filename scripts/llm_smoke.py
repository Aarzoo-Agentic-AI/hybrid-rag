"""Smoke test: send one tiny prompt through the real FailoverChatModel for each task.

Usage:
    uv run python scripts/llm_smoke.py            # invoke + stream for every task
    uv run python scripts/llm_smoke.py --task answer

Costs a fraction of a cent per run. Providers without an API key are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from hybrid_rag.config import get_settings
from hybrid_rag.llm import FAILOVER_METADATA_KEY, LLMError, LLMFactory

PROMPT = "Reply with exactly one word: pong"


def run_task(factory: LLMFactory, task: str) -> bool:
    print(f"\n=== task: {task} ===")
    try:
        model = factory.for_task(task)  # type: ignore[arg-type]
    except LLMError as exc:
        print(f"  could not build model: {exc}")
        return False
    print(f"  providers (in order): {', '.join(s.label for s in model.slots)}")

    ok = True
    try:
        started = time.perf_counter()
        message = model.invoke(PROMPT)
        info = message.response_metadata[FAILOVER_METADATA_KEY]
        print(f"  invoke -> {message.content!r}")
        print(
            f"    answered by {info['provider']}/{info['model']} "
            f"(failed over: {info['failed_over']}, attempts: {info['attempts']}, "
            f"{time.perf_counter() - started:.2f}s)"
        )
        if message.usage_metadata:
            u = message.usage_metadata
            print(f"    tokens: {u['input_tokens']} in / {u['output_tokens']} out")
    except Exception as exc:
        print(f"  invoke FAILED: {type(exc).__name__}: {exc}")
        ok = False

    try:
        full = None
        for chunk in model.stream(PROMPT):
            full = chunk if full is None else full + chunk
        info = full.response_metadata[FAILOVER_METADATA_KEY] if full else {}
        print(f"  stream -> {full.content if full else ''!r} (by {info.get('provider')})")
    except Exception as exc:
        print(f"  stream FAILED: {type(exc).__name__}: {exc}")
        ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["answer", "query_rewrite"], action="append")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    factory = LLMFactory(get_settings())
    tasks = args.task or ["answer", "query_rewrite"]
    results = [run_task(factory, t) for t in tasks]

    print("\ncircuit breakers:", factory.health())
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
