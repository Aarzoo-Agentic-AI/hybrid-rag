"""Retrieval and answer metrics. Pure functions over ranked labels and answer text.

Ranks are per chunk, as retrieval returns them (what the LLM would see):
- **Recall@k:** the share of a question's requirements met by a chunk in the top k.
- **MRR:** 1 / the rank of the first chunk that meets any requirement (0 if none).
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Sequence

from hybrid_rag.evaluation.golden import Label, Requirement

_DASHES = re.compile(
    "[" + "".join(map(chr, (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212))) + "]"
)
_SPACES = re.compile("[" + "".join(map(chr, (0xA0, 0x202F, 0x2009))) + r"\s]+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_QUOTES = str.maketrans({chr(0x2018): "'", chr(0x2019): "'", chr(0x201C): '"', chr(0x201D): '"'})


def first_ranks(labels: Sequence[Label], requirements: Sequence[Requirement]) -> list[int | None]:
    """For each requirement, the 1-based rank of the first chunk that meets it, or None."""
    ranks: list[int | None] = []
    for req in requirements:
        ranks.append(next((i for i, lab in enumerate(labels, start=1) if req.met_by(lab)), None))
    return ranks


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    if not ranks:
        raise ValueError("recall needs at least one requirement")
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def reciprocal_rank(ranks: Sequence[int | None]) -> float:
    found = [r for r in ranks if r is not None]
    return 1.0 / min(found) if found else 0.0


def normalize(text: str) -> str:
    """For fact matching: lower case, plain dashes, quotes and spaces, no thousands commas."""
    text = _DASHES.sub("-", text.translate(_QUOTES))
    text = _THOUSANDS.sub("", text)
    return _SPACES.sub(" ", text).strip().lower()


def facts_found(text: str, facts: Sequence[Sequence[str]]) -> list[bool]:
    """For each fact, whether any of its spellings appears in ``text``."""
    clean = normalize(text)
    return [any(normalize(alt) in clean for alt in fact) for fact in facts]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def gate_rows(
    answerable: Sequence[float | None],
    unanswerable: Sequence[float | None],
    thresholds: Iterable[float],
) -> list[dict[str, float]]:
    """How a ``min_rerank_score`` threshold splits the questions.

    A score of None (nothing retrieved) is always refused. ``kept`` is the share of answerable
    questions that would still reach the LLM; ``refused`` the share of unanswerable ones the
    gate stops; ``balanced`` their mean.
    """
    rows = []
    for t in thresholds:
        kept = mean(1.0 if s is not None and s >= t else 0.0 for s in answerable)
        refused = mean(1.0 if s is None or s < t else 0.0 for s in unanswerable)
        kept, refused = kept or 0.0, refused or 0.0
        rows.append(
            {"threshold": t, "kept": kept, "refused": refused, "balanced": (kept + refused) / 2}
        )
    return rows


def best_threshold(
    answerable: Sequence[float | None], unanswerable: Sequence[float | None], *, step: float = 0.25
) -> dict[str, float] | None:
    """The threshold with the best balanced accuracy (ties: the one keeping most answerable)."""
    scores = [s for s in (*answerable, *unanswerable) if s is not None]
    if not scores or not answerable or not unanswerable:
        return None
    lo, hi = min(scores) - step, max(scores) + step
    n = int((hi - lo) / step) + 1
    rows = gate_rows(answerable, unanswerable, (round(lo + i * step, 2) for i in range(n)))
    return max(rows, key=lambda r: (r["balanced"], r["kept"]))
