"""Search the indexed policies as a given caller, to try retrieval modes by hand.

Usage:
    uv run python scripts/search.py "How many PTO days do I get after 4 years?"
    uv run python scripts/search.py "merit matrix" --as manager --mode hybrid_rerank
    uv run python scripts/search.py "hotel cap in Boston" --mode sparse --category travel --text
    uv run python scripts/search.py "COBRA" --compare          # all four modes side by side

Callers (--as) are demo principals in documents.tenant_id. In the API (step 6) principals
come from the caller's identity, never from a flag. hybrid_rerank downloads the reranker
(reranker.model) on first use.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from hybrid_rag.access import Principal
from hybrid_rag.config import get_settings
from hybrid_rag.ingestion import citation
from hybrid_rag.retrieval import MODES, RelevanceFilters, build_retriever
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
    "admin": None,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("--as", dest="caller", choices=DEMO_CALLERS, default="employee")
    parser.add_argument("--mode", choices=MODES, help="default: retrieval.mode")
    parser.add_argument("--compare", action="store_true", help="run all four modes")
    parser.add_argument("-k", type=int, help="results (default: retrieval.top_k)")
    parser.add_argument("--category", action="append", help="policy_category filter")
    parser.add_argument("--all-versions", action="store_true", help="include superseded ones")
    parser.add_argument("--no-expand", action="store_true", help="no small-to-big expansion")
    parser.add_argument("--text", action="store_true", help="print each result's text")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    settings = get_settings()
    if args.k:
        settings.retrieval.top_k = args.k
        settings.retrieval.fused_limit = max(settings.retrieval.fused_limit, args.k)
    tenant = settings.documents.tenant_id
    spec = DEMO_CALLERS[args.caller]
    principal = Principal.tenant_admin(tenant) if spec is None else Principal(tenant, **spec)
    filters = RelevanceFilters(
        current_only=not args.all_versions, policy_categories=args.category or None
    )
    try:
        index = build_document_index(settings)
    except StoreError as exc:
        print(f"error: {exc}")
        return 1

    print(f"As {args.caller} ({principal.clearance}) in tenant {tenant!r}: {args.query!r}")
    for mode in MODES if args.compare else [args.mode or settings.retrieval.mode]:
        retriever = build_retriever(
            settings, index, principal, mode=mode, filters=filters, expand=not args.no_expand
        )
        started = time.perf_counter()
        hits = retriever.invoke(args.query)
        print(f"\n{mode} ({time.perf_counter() - started:.1f}s, {len(hits)} results)")
        for i, h in enumerate(hits, start=1):
            m = h.metadata
            scores = f"retrieval #{m.get('retrieval_rank')} {m.get('retrieval_score') or 0:.3f}"
            if "rerank_score" in m:
                scores += f" | rerank {m['rerank_score']:.3f}"  # scale depends on the model
            expanded = " [whole section]" if m.get("section_expanded") else ""
            print(f"  {i}. {citation(m)}{expanded}\n     {scores}")
            if args.text:
                print("     " + h.page_content.replace("\n", "\n     "))
    flush_traces()
    return 0


if __name__ == "__main__":
    sys.exit(main())
