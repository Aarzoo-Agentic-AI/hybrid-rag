"""Answer generation: question -> cited answer, for one caller. See README.md."""

from hybrid_rag.generation.citations import parse_citations
from hybrid_rag.generation.context import Source, format_sources, select_sources
from hybrid_rag.generation.prompts import ANSWER_PROMPT, NOT_FOUND, REWRITE_PROMPT
from hybrid_rag.generation.service import AnswerService
from hybrid_rag.generation.types import Answer, AnswerStatus, Citation, UsedSource

__all__ = [
    "ANSWER_PROMPT",
    "NOT_FOUND",
    "REWRITE_PROMPT",
    "Answer",
    "AnswerService",
    "AnswerStatus",
    "Citation",
    "Source",
    "UsedSource",
    "format_sources",
    "parse_citations",
    "select_sources",
]
