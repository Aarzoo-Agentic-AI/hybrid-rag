"""LangSmith tracing helpers.

LangChain components (chat models, retrievers, LCEL chains) are traced automatically when
``LANGSMITH_TRACING=true``. Our own plain-Python steps (loading, chunking, embedding, Qdrant
writes) are traced with LangSmith's ``@traceable`` decorator, so every step shows up in one
trace tree. With tracing off, ``@traceable`` does nothing but call the function.

Ingestion and embedding runs record **counts, names, statuses and timings only**: never
document text and never vectors. LLM runs do contain prompts and retrieved text; set
``LANGSMITH_HIDE_INPUTS=true`` / ``LANGSMITH_HIDE_OUTPUTS=true`` to keep only their structure
and timings.
"""

from __future__ import annotations

from collections.abc import Callable

from langsmith import Client, traceable
from langsmith.anonymizer import create_anonymizer
from langsmith.run_trees import get_cached_client
from langsmith.utils import tracing_is_enabled

__all__ = ["anonymizing_client", "flush_traces", "traceable", "tracing_is_enabled"]


def flush_traces() -> None:
    """Send any queued runs to LangSmith. Call before a short-lived script exits."""
    if tracing_is_enabled():
        get_cached_client().flush()


def anonymizing_client(mask: Callable[[str], str]) -> Client:
    """A LangSmith client that runs ``mask`` over every string it sends (inputs, outputs,
    metadata), as a last line of defence: the API passes the privacy pattern check, so an
    email, phone number or ID that slipped into a trace is masked before it leaves."""
    return Client(anonymizer=create_anonymizer(lambda text, _path: mask(text)))
