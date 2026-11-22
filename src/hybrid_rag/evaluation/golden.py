"""The golden dataset: HR questions with the policy sections that answer them (eval/golden.yaml).

A question's ``expected`` sections are *requirements*: each one must be found, and each may be
met by any of several sections (the same rule is often stated in two policies). Labels are
``(source file, section path)``, not chunk IDs, so the dataset survives re-chunking.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CATEGORIES = (
    "fact",
    "table_lookup",
    "exception",
    "cross_policy",
    "definition",
    "follow_up",
    "access",
    "unanswerable",
)
STYLES = ("keyword", "paraphrase")
LABEL_SEPARATOR = "::"
_ID = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

Label = tuple[str, str]  # (source file, section path)


class GoldenError(ValueError):
    """The golden dataset is invalid. The message lists every problem."""


@dataclass(frozen=True)
class Requirement:
    """One thing retrieval must find: any of ``options`` meets it."""

    options: frozenset[Label]

    def met_by(self, label: Label) -> bool:
        return label in self.options


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    category: str
    style: str
    caller: str = "employee"
    expected: tuple[Requirement, ...] = ()
    facts: tuple[tuple[str, ...], ...] = ()  # each fact: its accepted spellings
    history: tuple[tuple[str, str], ...] = ()  # (role, text), oldest first
    standalone: str | None = None
    hidden_from: tuple[str, ...] = ()
    notes: str | None = None

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"

    @property
    def retrieval_query(self) -> str:
        """What retrieval is scored on: the hand-written standalone form of a follow-up."""
        return self.standalone or self.question

    @property
    def expected_sources(self) -> frozenset[str]:
        return frozenset(source for r in self.expected for source, _ in r.options)


@dataclass(frozen=True)
class GoldenSet:
    version: int
    sources: dict[str, str]  # alias -> file name
    questions: tuple[GoldenQuestion, ...]
    sha256: str = ""
    path: str = ""

    def labels(self) -> set[Label]:
        return {label for q in self.questions for r in q.expected for label in r.options}

    def select(self, ids: Collection[str]) -> GoldenSet:
        unknown = sorted(set(ids) - {q.id for q in self.questions})
        if unknown:
            raise GoldenError(f"Unknown question IDs: {', '.join(unknown)}")
        keep = tuple(q for q in self.questions if q.id in set(ids))
        return GoldenSet(self.version, self.sources, keep, self.sha256, self.path)


def load_golden(path: str | Path, *, callers: Collection[str]) -> GoldenSet:
    """Load and validate ``path``. ``callers`` are the principal names questions may use."""
    path = Path(path)
    raw = path.read_bytes()
    data = yaml.safe_load(raw) or {}
    golden = parse_golden(data, callers=callers)
    return GoldenSet(
        golden.version,
        golden.sources,
        golden.questions,
        sha256=hashlib.sha256(raw).hexdigest(),
        path=str(path),
    )


def parse_golden(data: dict[str, Any], *, callers: Collection[str]) -> GoldenSet:
    errors: list[str] = []
    sources = data.get("sources") or {}
    if not isinstance(sources, dict) or not sources:
        raise GoldenError("golden set: `sources` must map aliases to file names")
    questions: list[GoldenQuestion] = []
    seen: set[str] = set()
    for i, item in enumerate(data.get("questions") or []):
        qid = str(item.get("id", f"#{i + 1}"))
        problems: list[str] = []
        try:
            q = _question(item, sources, set(callers), problems)
        except (TypeError, ValueError, AttributeError) as exc:
            problems.append(f"malformed ({exc})")
            q = None
        if qid in seen:
            problems.append("duplicate id")
        seen.add(qid)
        errors.extend(f"{qid}: {p}" for p in problems)
        if q is not None and not problems:
            questions.append(q)
    if not questions and not errors:
        errors.append("no questions")
    if errors:
        raise GoldenError("Invalid golden set:\n  " + "\n  ".join(errors))
    return GoldenSet(int(data.get("version", 1)), dict(sources), tuple(questions))


def _question(
    item: dict[str, Any], sources: dict[str, str], callers: set[str], problems: list[str]
) -> GoldenQuestion:
    known = {
        "id", "question", "category", "style", "caller", "expected", "facts",
        "history", "standalone", "hidden_from", "notes",
    }  # fmt: skip
    for key in sorted(set(item) - known):
        problems.append(f"unknown field {key!r}")
    qid = str(item.get("id", ""))
    if not _ID.match(qid):
        problems.append("id must be lowercase words joined by '-'")
    question = str(item.get("question") or "").strip()
    if not question:
        problems.append("empty question")
    category = item.get("category")
    if category not in CATEGORIES:
        problems.append(f"category must be one of {', '.join(CATEGORIES)}")
    style = item.get("style")
    if style not in STYLES:
        problems.append(f"style must be one of {', '.join(STYLES)}")
    caller = item.get("caller", "employee")
    hidden_from = tuple(item.get("hidden_from") or ())
    for name in (caller, *hidden_from):
        if name not in callers:
            problems.append(f"unknown caller {name!r}")
    if caller in hidden_from:
        problems.append("caller is also in hidden_from")

    expected = tuple(_requirement(e, sources, problems) for e in item.get("expected") or ())
    facts = tuple(
        tuple(alt.strip() for alt in str(f).split("|") if alt.strip())
        for f in item.get("facts") or ()
    )
    if any(not f for f in facts):
        problems.append("empty fact")
    if category == "unanswerable":
        if expected or facts or hidden_from:
            problems.append("unanswerable questions have no expected, facts or hidden_from")
    elif not expected:
        problems.append("answerable questions need at least one expected section")
    if hidden_from and not expected:
        problems.append("hidden_from needs expected sections")

    history: list[tuple[str, str]] = []
    for turn in item.get("history") or ():
        if set(turn) - {"user", "assistant"} or "user" not in turn:
            problems.append("history turns need `user` (and optionally `assistant`)")
            continue
        history.append(("user", str(turn["user"])))
        if turn.get("assistant"):
            history.append(("assistant", str(turn["assistant"])))
    standalone = item.get("standalone")
    if category == "follow_up" and (not history or not standalone):
        problems.append("follow_up questions need history and standalone")
    if category != "follow_up" and (history or standalone):
        problems.append("only follow_up questions have history or standalone")

    return GoldenQuestion(
        id=qid,
        question=question,
        category=str(category),
        style=str(style),
        caller=str(caller),
        expected=expected,
        facts=facts,
        history=tuple(history),
        standalone=str(standalone).strip() if standalone else None,
        hidden_from=hidden_from,
        notes=item.get("notes"),
    )


def _requirement(entry: Any, sources: dict[str, str], problems: list[str]) -> Requirement:
    options = entry if isinstance(entry, list) else [entry]
    labels: set[Label] = set()
    for option in options:
        alias, sep, section = str(option).partition(LABEL_SEPARATOR)
        if not sep or not section.strip():
            problems.append(f"label {option!r} must be '<alias>{LABEL_SEPARATOR}<section path>'")
        elif alias not in sources:
            problems.append(f"unknown source alias {alias!r} in {option!r}")
        else:
            labels.add((sources[alias], section.strip()))
    if not options:
        problems.append("empty requirement")
    return Requirement(frozenset(labels))
