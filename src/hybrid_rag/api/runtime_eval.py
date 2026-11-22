"""Runtime evaluation: a faithfulness check on a sample of live answers (``runtime_eval``).

After an answered question, ``RuntimeEvaluator.maybe_check`` picks it with probability
``sample_rate`` and hands it to one background thread, so the response is never delayed. The
judge (``llm.judge``) scores how much of the answer the sources support. The result goes to:
- **the audit log:** a ``runtime_eval`` line with the request ID, the score and the unsupported
  claims (masked by the privacy check, though they come from an answer that already passed it);
- **LangSmith,** when tracing is on: ``faithfulness`` feedback on the request's answer trace.

If more than ``max_pending`` checks are waiting (a burst of traffic, or a slow judge), new ones
are skipped rather than queued, so memory stays bounded. ``scripts/runtime_report.py`` turns
the audit and feedback logs into a report.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any

from langchain_core.language_models import BaseChatModel
from langsmith import tracing_context

from hybrid_rag.api.audit import AuditLog
from hybrid_rag.api.feedback import LangSmithFeedback
from hybrid_rag.generation import Answer
from hybrid_rag.generation.judge import Faithfulness, judge_faithfulness

logger = logging.getLogger(__name__)


class RuntimeEvaluator:
    def __init__(
        self,
        judge_llm: BaseChatModel,
        audit: AuditLog,
        *,
        sample_rate: float,
        max_pending: int = 50,
        trace_client: Any = None,  # traces the judge's call (with the anonymizer)
        langsmith: LangSmithFeedback | None = None,  # sends the score to the answer's trace
        mask: Callable[[str], str] = lambda text: text,
        rng: random.Random | None = None,
    ) -> None:
        self.judge_llm = judge_llm
        self.audit = audit
        self.sample_rate = sample_rate
        self.max_pending = max_pending
        self.trace_client = trace_client
        self.langsmith = langsmith
        self.mask = mask
        self.rng = rng or random.Random()
        self.skipped = 0  # sampled but dropped because too many were pending
        self._pending = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runtime-eval")

    def maybe_check(self, *, request_id: str, tenant_id: str, user_id: str, answer: Answer) -> bool:
        """Queue a faithfulness check for a sample of answered questions. True if queued."""
        if answer.status != "answered" or not answer.context or not request_id:
            return False
        if self.rng.random() >= self.sample_rate:
            return False
        with self._lock:
            if self._pending >= self.max_pending:
                self.skipped += 1
                return False
            self._pending += 1
        self._pool.submit(self._check, request_id, tenant_id, user_id, answer)
        return True

    def wait_idle(self, timeout: float = 30) -> bool:
        with self._idle:
            return self._idle.wait_for(lambda: self._pending == 0, timeout=timeout)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------ the check

    def _check(self, request_id: str, tenant_id: str, user_id: str, answer: Answer) -> None:
        started = time.perf_counter()
        result: Faithfulness | None = None
        error: str | None = None
        try:
            with self._trace(request_id, tenant_id):
                result = judge_faithfulness(self.judge_llm, answer.context, answer.text)
            if result is None:
                error = "unreadable judge reply"
        except Exception as exc:  # never let a background check take anything down
            error = type(exc).__name__
            logger.warning("Runtime evaluation of %s failed: %s", request_id, error)
        try:
            claims = [self.mask(c) for c in result.unsupported] if result else []
            self.audit.write(
                {
                    "request_id": request_id,
                    "route": "runtime_eval",
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "check": "faithfulness",
                    "score": result.score if result else None,
                    "unsupported": claims,
                    "error": error,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )
            if result is not None and self.langsmith is not None:
                self.langsmith.send(
                    request_id, "faithfulness", result.score, "; ".join(claims) or None
                )
        except Exception:
            logger.exception("Couldn't record the runtime evaluation of %s", request_id)
        finally:
            with self._idle:
                self._pending -= 1
                self._idle.notify_all()

    def _trace(self, request_id: str, tenant_id: str):
        if self.trace_client is None:
            return nullcontext()
        return tracing_context(
            client=self.trace_client,
            metadata={"request_id": request_id, "tenant_id": tenant_id, "route": "runtime_eval"},
            tags=["route:runtime_eval", f"tenant:{tenant_id}"],
        )
