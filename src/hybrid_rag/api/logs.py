"""Structured logging with the request ID on every line.

The request ID (``X-Request-ID``) is kept in a context variable, so any log line written while
handling a request carries it, and so does the request's LangSmith trace (as metadata): a log
line leads to its trace.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
tenant_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)

_STANDARD = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.tenant_id = tenant_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD and value is not None:
                data[key] = value
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def setup_logging(*, json_logs: bool, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(ContextFilter())
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s")
        )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "sentence_transformers",
        "docling",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
