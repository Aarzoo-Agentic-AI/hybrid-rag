"""Evaluation against the golden dataset (eval/golden.yaml). Run it with eval/evaluate.py."""

from hybrid_rag.evaluation.golden import (
    CATEGORIES,
    STYLES,
    GoldenError,
    GoldenQuestion,
    GoldenSet,
    Label,
    Requirement,
    load_golden,
    parse_golden,
)
from hybrid_rag.evaluation.metrics import (
    best_threshold,
    facts_found,
    first_ranks,
    gate_rows,
    normalize,
    recall_at,
    reciprocal_rank,
)
from hybrid_rag.evaluation.report import render_answers, render_retrieval
from hybrid_rag.evaluation.runner import (
    KS,
    AnswerRun,
    LeakRun,
    RetrievalRun,
    run_answers,
    run_retrieval,
    summarize,
)

__all__ = [
    "CATEGORIES",
    "KS",
    "STYLES",
    "AnswerRun",
    "GoldenError",
    "GoldenQuestion",
    "GoldenSet",
    "Label",
    "LeakRun",
    "Requirement",
    "RetrievalRun",
    "best_threshold",
    "facts_found",
    "first_ranks",
    "gate_rows",
    "load_golden",
    "normalize",
    "parse_golden",
    "recall_at",
    "reciprocal_rank",
    "render_answers",
    "render_retrieval",
    "run_answers",
    "run_retrieval",
    "summarize",
]
