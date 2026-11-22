"""User feedback on answers (``POST /feedback``).

- **Who may rate:** only the caller who asked. ``RecentRequests`` remembers, in memory, who
  asked each recent question (``api.recent_requests`` entries for ``api.feedback_window_hours``).
  Feedback on anything else answers 404, like a missing request, so it reveals nothing.
- **Linked to the trace:** each request's LangSmith trace gets the ID ``run_id_for(request_id)``,
  so a rating is attached to it as ``user_rating`` feedback with no lookup.
- **What's stored** (``api.feedback_log``): the rating and the comment, both after the privacy
  masking used for questions. A thumbs-down also stores the masked question, so it can become
  a golden-set question (``scripts/runtime_report.py`` lists them). The API docs say so.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langsmith.utils import get_tracer_project

logger = logging.getLogger(__name__)

_RUN_NAMESPACE = uuid.UUID("5d1f0c2e-8c1b-4e5a-9f4e-2b6d8a7c3e10")


def run_id_for(request_id: str) -> uuid.UUID:
    """The LangSmith run ID of a request's answer trace (stable: derived from the request ID)."""
    return uuid.uuid5(_RUN_NAMESPACE, request_id)


@dataclass(frozen=True)
class RecentRequest:
    request_id: str
    tenant_id: str
    user_id: str
    question: str  # masked (and rewritten) question, as in the Answer
    answer_status: str
    asked_at: float  # time.time()


class RecentRequests:
    """Who asked each recent question, so only they may rate the answer. In memory: after a
    restart, older requests can't be rated."""

    def __init__(self, capacity: int = 10_000, window_s: float = 7 * 24 * 3600) -> None:
        self.capacity = capacity
        self.window_s = window_s
        self._items: OrderedDict[str, RecentRequest] = OrderedDict()
        self._rated: set[str] = set()
        self._lock = threading.Lock()

    def remember(self, item: RecentRequest) -> None:
        with self._lock:
            self._items[item.request_id] = item
            self._items.move_to_end(item.request_id)
            while len(self._items) > self.capacity:
                old, _ = self._items.popitem(last=False)
                self._rated.discard(old)

    def lookup(self, request_id: str, tenant_id: str, user_id: str) -> RecentRequest | None:
        """The request, if this caller asked it within the window; otherwise None."""
        with self._lock:
            item = self._items.get(request_id)
        if item is None or (item.tenant_id, item.user_id) != (tenant_id, user_id):
            return None
        if time.time() - item.asked_at > self.window_s:
            return None
        return item

    def mark_rated(self, request_id: str) -> bool:
        """True the first time; False if the request was already rated."""
        with self._lock:
            if request_id in self._rated:
                return False
            self._rated.add(request_id)
            return True


class FeedbackLog:
    """One JSON line per rating. Never fails the request."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(
            {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), **record},
            sort_keys=True,
            default=str,
        )
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:
            logger.exception("Couldn't write the feedback log")


class LangSmithFeedback:
    """Sends feedback (``user_rating``, ``faithfulness``) to a request's answer trace.

    LangSmith wants the trace's project ID with run-level feedback (sending it without is
    deprecated), so the ID of the tracing project (``LANGSMITH_PROJECT``) is looked up once,
    on first use, and cached. Errors are logged, never raised: feedback must not fail a request.
    """

    def __init__(self, client: Any, project_name: str | None = None) -> None:
        self.client = client
        self.project_name = project_name or get_tracer_project()
        self._project_id: str | None = None
        self._lock = threading.Lock()

    def project_id(self) -> str | None:
        with self._lock:
            if self._project_id is None:
                try:
                    self._project_id = str(
                        self.client.read_project(project_name=self.project_name).id
                    )
                except Exception:
                    logger.warning("LangSmith project %r not found yet", self.project_name)
            return self._project_id

    def send(self, request_id: str, key: str, score: float, comment: str | None = None) -> bool:
        run_id = run_id_for(request_id)
        session_id = self.project_id()
        if session_id is None:
            return False
        try:
            # The answer run is the trace's root, so its trace ID is its run ID.
            self.client.create_feedback(
                run_id,
                key=key,
                score=score,
                comment=comment,
                trace_id=run_id,
                session_id=session_id,
            )
            return True
        except Exception:
            logger.exception("Couldn't send %s feedback for %s to LangSmith", key, request_id)
            return False
