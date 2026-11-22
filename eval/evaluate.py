"""Evaluate retrieval (and optionally answers) against the golden dataset.

Usage:
    uv run python eval/evaluate.py                      # all four modes, no LLM (a few minutes)
    uv run python eval/evaluate.py --answers            # also answer every question (OpenAI calls)
    uv run python eval/evaluate.py --mode hybrid --mode hybrid_rerank --only trv-hotel-boston
    uv run python eval/evaluate.py --reranker BAAI/bge-reranker-v2-m3 --out eval/results/bge-v2-m3
    uv run python eval/evaluate.py --rescore eval/results        # re-check saved answers, no LLM

Writes retrieval.md / retrieval.json (and answers.md / answers.json) to --out. Needs Qdrant with
data/pdfs indexed (scripts/ingest.py). Questions are asked as demo principals in
documents.tenant_id, the same as the demo API keys. See eval/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.retrievers import BaseRetriever

from hybrid_rag.access import Principal
from hybrid_rag.config import Settings, get_settings
from hybrid_rag.evaluation import (
    KS,
    GoldenError,
    GoldenQuestion,
    GoldenSet,
    Label,
    load_golden,
    render_answers,
    render_retrieval,
    run_answers,
    run_retrieval,
)
from hybrid_rag.evaluation.report import overall_table
from hybrid_rag.evaluation.runner import load_answer_runs, rescore
from hybrid_rag.generation import AnswerService
from hybrid_rag.generation.types import Answer
from hybrid_rag.retrieval import MODES, build_retriever
from hybrid_rag.store import DocumentIndex, StoreError, build_document_index
from hybrid_rag.store.security import scoped
from hybrid_rag.tracing import flush_traces

ROOT = Path(__file__).resolve().parent.parent

# The demo principals (the same as scripts/make_api_key.py --demo).
CALLERS: dict[str, dict[str, Any]] = {
    "employee": {"user_id": "u-demo-employee", "clearance": "internal"},
    "manager": {
        "user_id": "u-demo-manager",
        "clearance": "confidential",
        "roles": ["people_manager"],
    },
    "hr-comp": {"user_id": "u-demo-hr", "clearance": "confidential", "groups": ["hr-compensation"]},
    "admin": {"user_id": "u-demo-admin", "clearance": "restricted", "admin": True},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", default=str(ROOT / "eval" / "golden.yaml"))
    parser.add_argument("--out", default=str(ROOT / "eval" / "results"))
    parser.add_argument("--mode", action="append", choices=MODES, help="default: all four")
    parser.add_argument("--only", action="append", metavar="ID", help="only these questions")
    parser.add_argument("--answers", action="store_true", help="also answer (calls the LLM)")
    parser.add_argument("--answer-mode", choices=MODES, help="default: retrieval.mode")
    parser.add_argument(
        "--strategy",
        choices=["single", "agentic"],
        help="with --answers (default: answer.strategy)",
    )
    parser.add_argument("--reranker", help="reranker model (default: reranker.model)")
    parser.add_argument(
        "--no-retrieval", action="store_true", help="with --answers: skip retrieval"
    )
    parser.add_argument(
        "--min-rerank-score",
        metavar="SCORE|off",
        help="override answer.min_rerank_score for --answers ('off': no confidence gate)",
    )
    parser.add_argument(
        "--rescore",
        metavar="DIR",
        help="re-check DIR/answers.json against the golden set (after fixing a fact), no LLM",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    try:
        golden = load_golden(args.golden, callers=CALLERS)
        if args.only:
            golden = golden.select(args.only)
    except (GoldenError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    if args.rescore:
        return rescore_saved(golden, Path(args.rescore))

    settings = get_settings()
    if args.reranker:
        settings.reranker.model = args.reranker
    if args.min_rerank_score is not None:
        gate = args.min_rerank_score.lower()
        settings.answer.min_rerank_score = None if gate == "off" else float(gate)
    tenant = settings.documents.tenant_id
    principals = {name: Principal(tenant, **spec) for name, spec in CALLERS.items()}
    try:
        index = build_document_index(settings)
    except StoreError as exc:
        print(f"error: {exc}")
        return 1

    indexed = indexed_labels(index, principals["admin"])
    missing = sorted(golden.labels() - indexed)
    if missing:
        print("error: these expected sections are not in the index (typo, or not ingested?):")
        for source, section in missing:
            print(f"  {source} :: {section}")
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = run_meta(settings, golden, len(indexed))
    print(f"Golden set: {len(golden.questions)} questions; {meta['index']}")

    if not args.no_retrieval:
        modes = args.mode or list(MODES)
        search_settings = settings.model_copy(deep=True)
        search_settings.retrieval.top_k = max(KS)
        search_settings.retrieval.fused_limit = max(settings.retrieval.fused_limit, max(KS))
        retrievers: dict[tuple[str, str], BaseRetriever] = {}

        def search(query: str, caller: str, mode: str) -> list:
            key = (caller, mode)
            if key not in retrievers:  # chunks as retrieved, without small-to-big expansion
                retrievers[key] = build_retriever(
                    search_settings,
                    index,
                    principals[caller],
                    mode=mode,
                    expand=False,  # type: ignore[arg-type]
                )
            index.embeddings.encoder.clear()  # every mode pays for embedding the query
            return retrievers[key].invoke(query)

        runs, leaks = run_retrieval(golden, modes, search, progress=lambda m: print(f"  {m}"))
        retrieval_meta = {
            **meta,
            "retrieval": (
                f"top {max(KS)} chunks; hybrid prefetch {settings.retrieval.dense_limit} dense + "
                f"{settings.retrieval.sparse_limit} sparse; reranker on "
                f"{search_settings.retrieval.fused_limit} fused candidates; no section expansion"
            ),
        }
        report = render_retrieval(
            golden,
            runs,
            leaks,
            modes,
            retrieval_meta,
            current_threshold=settings.answer.min_rerank_score,
        )
        (out / "retrieval.md").write_text(report, encoding="utf-8")
        write_json(
            out / "retrieval.json",
            {
                "meta": retrieval_meta,
                "runs": [vars(r) for r in runs],
                "leaks": [vars(x) for x in leaks],
            },
        )
        print("\n" + "\n".join(overall_table(runs, modes)))
        print(f"\nWrote {out / 'retrieval.md'}")

    if args.answers:
        service = AnswerService.from_settings(settings, index)
        mode = args.answer_mode or settings.retrieval.mode
        strategy = args.strategy or settings.answer.strategy

        def ask(q: GoldenQuestion, caller: str) -> Answer:
            try:
                return service.answer(
                    q.question,
                    principals[caller],
                    history=list(q.history),
                    mode=mode,
                    strategy=strategy,  # type: ignore[arg-type]
                )
            except Exception as exc:  # one failure shouldn't lose the whole run
                return Answer(question=q.question, text=f"error: {exc}", status="error")  # type: ignore[arg-type]

        answer_runs = run_answers(golden, ask, progress=lambda m: print(f"  {m}"))
        answers_meta = {
            **meta,
            "answer mode": f"`{mode}`",
            "strategy": f"`{strategy}`",
            "sources sent": (
                f"top {settings.retrieval.top_k} chunks, "
                f"at most {settings.answer.context_max_tokens} tokens"
            ),
            "llm": llm_targets(settings),
            "confidence gate": f"min_rerank_score {settings.answer.min_rerank_score}",
        }
        (out / "answers.md").write_text(
            render_answers(golden, answer_runs, answers_meta), encoding="utf-8"
        )
        write_json(
            out / "answers.json",
            {"meta": answers_meta, "runs": [r.to_dict() for r in answer_runs]},
        )
        correct = sum(r.correct for r in answer_runs)
        print(f"\nAnswers: {correct}/{len(answer_runs)} correct. Wrote {out / 'answers.md'}")

    flush_traces()
    return 0


def rescore_saved(golden: GoldenSet, folder: Path) -> int:
    path = folder / "answers.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: can't read {path}: {exc}")
        return 1
    runs = rescore(golden, load_answer_runs(data))
    meta = {
        **data["meta"],
        "golden set": f"v{golden.version}, sha256 {golden.sha256[:12]}",
        "rescored": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC") + " (same answers)",
    }
    (folder / "answers.md").write_text(render_answers(golden, runs, meta), encoding="utf-8")
    write_json(path, {"meta": meta, "runs": [r.to_dict() for r in runs]})
    print(f"Rescored {len(runs)} answers: {sum(r.correct for r in runs)} correct. Wrote {folder}")
    return 0


def indexed_labels(index: DocumentIndex, admin: Principal) -> set[Label]:
    labels: set[Label] = set()
    offset = None
    while True:
        points, offset = index.client.scroll(
            index.collection,
            scroll_filter=scoped(admin),
            limit=256,
            offset=offset,
            with_payload=["metadata.source", "metadata.section_path"],
            with_vectors=False,
        )
        for p in points:
            m = (p.payload or {}).get("metadata", {})
            labels.add((str(m.get("source")), str(m.get("section_path"))))
        if offset is None:
            return labels


def run_meta(settings: Settings, golden: GoldenSet, sections: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for q in golden.questions:
        counts[q.category] = counts.get(q.category, 0) + 1
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": git_commit(),
        "golden set": (
            f"v{golden.version}, {len(golden.questions)} questions, sha256 {golden.sha256[:12]} "
            f"({', '.join(f'{k} {v}' for k, v in counts.items())})"
        ),
        "index": f"tenant `{settings.documents.tenant_id}`, {sections} distinct sections",
        "embeddings": f"`{settings.embedding.model}` (dense + sparse)",
        "reranker": f"`{settings.reranker.model}`",
        "machine": (
            f"{platform.processor() or platform.machine()}, Python {platform.python_version()}"
        ),
    }


def llm_targets(settings: Settings) -> str:
    targets = settings.llm.answer
    return ", then ".join(f"`{t.provider}/{t.model}`" for t in targets)


def git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout.strip()
        return f"{sha}{' (uncommitted changes)' if dirty else ''}" if sha else "unknown"
    except OSError:
        return "unknown"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
