"""Agentic retrieval with LangGraph: search, judge whether the sources are enough, search again.

The graph covers only retrieval. What it gathers goes through the same source selection,
confidence gate, prompt, citation parsing and privacy check as a single-pass answer
(``AnswerService``), so the two strategies differ only in which chunks reach the LLM.

```
START → retrieve → grade ──sufficient, or out of rounds──→ END
           ↑          │
           └── not sufficient: up to agent.max_queries new queries
```

- **retrieve** runs the caller's retriever (``build_retriever``, with the caller's principal
  and filters) on each pending query and appends chunks it hasn't seen, after the first
  round's. The first round's ranking stays first, so the confidence gate and the best
  sources are those of the question itself.
- **grade** shows a small model (``llm.agent``) the question and the sources as the answering
  LLM would see them, and asks for JSON: ``{"sufficient", "missing", "queries"}``. An
  unreadable reply counts as sufficient: the answer step still says "not found" when the
  sources don't answer the question.

**Security:** the model chooses only query text. It never sees or sets the principal, the
tenant or the filters; every search runs through the same permission-filtered retriever.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You help an HR policy assistant find the right policy text. You get an employee's "
            "question and the policy excerpts found so far. Decide whether the excerpts contain "
            "what's needed to answer the question. If something is missing, write up to "
            "{max_queries} short search queries for it, in the words a policy would use (for "
            "example 'sabbatical years of service' or 'expense approval limits'). Stay on the "
            "question's topic. If the policies seem not to cover it at all, say sufficient: "
            "true rather than guessing queries.\n\n"
            "Reply with JSON only, no other text:\n"
            '{{"sufficient": true or false, "missing": "<what is missing, or empty>", '
            '"queries": ["<search query>", ...]}}',
        ),
        ("human", "Question: {question}\n\nExcerpts found so far:\n{sources}"),
    ]
)

_JSON = re.compile(r"\{.*\}", re.DOTALL)
MAX_QUERY_CHARS = 200


@dataclass(frozen=True)
class Grade:
    sufficient: bool
    missing: str = ""
    queries: list[str] = field(default_factory=list)


def parse_grade(text: str) -> Grade:
    """The grader's JSON reply. Anything unreadable counts as sufficient (no more searching)."""
    match = _JSON.search(text or "")
    if not match:
        return Grade(sufficient=True)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Grade(sufficient=True)
    if not isinstance(data, dict):
        return Grade(sufficient=True)
    queries = data.get("queries") or []
    if not isinstance(queries, list):
        queries = [queries]
    clean = [str(q).strip()[:MAX_QUERY_CHARS] for q in queries if str(q).strip()]
    sufficient = data.get("sufficient") is not False  # anything but an explicit false
    return Grade(sufficient=sufficient, missing=str(data.get("missing") or ""), queries=clean)


class AgentState(TypedDict, total=False):
    question: str  # the standalone question (masked, rewritten if a follow-up)
    pending: list[str]  # queries to run in the next retrieve step
    asked: list[str]  # every query run so far, in order
    docs: list[Document]  # every chunk found, first round first, no duplicates
    rounds: int  # retrieve steps run
    sufficient: bool
    missing: str


def _key(doc: Document) -> str:
    return str(doc.metadata.get("chunk_id") or doc.id or doc.page_content)


def _norm(query: str) -> str:
    return " ".join(query.lower().split())


def build_agent(
    retrieve: Callable[[str], list[Document]],
    grader: BaseChatModel,
    render_sources: Callable[[list[Document]], str],
    *,
    max_rounds: int = 2,
    max_queries: int = 2,
) -> Any:
    """The compiled graph. ``retrieve`` searches as the caller; ``render_sources`` turns chunks
    into the numbered sources the answering LLM would see. ``max_rounds`` counts the extra
    retrieve steps after the first."""
    grade_chain = (GRADE_PROMPT | grader).with_config(run_name="grade_sources")

    def retrieve_step(state: AgentState) -> dict[str, Any]:
        docs = list(state.get("docs") or [])
        seen = {_key(d) for d in docs}
        for query in state.get("pending") or []:
            for doc in retrieve(query):
                if _key(doc) not in seen:
                    seen.add(_key(doc))
                    docs.append(doc)
        return {
            "docs": docs,
            "asked": [*(state.get("asked") or []), *(state.get("pending") or [])],
            "pending": [],
            "rounds": (state.get("rounds") or 0) + 1,
        }

    def grade_step(state: AgentState) -> dict[str, Any]:
        if (state.get("rounds") or 0) > max_rounds:  # out of rounds: no need to ask
            return {"sufficient": True, "pending": []}
        reply = grade_chain.invoke(
            {
                "question": state["question"],
                "sources": render_sources(state.get("docs") or []) or "(nothing found)",
                "max_queries": max_queries,
            }
        )
        content = reply.content
        if not isinstance(content, str):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        grade = parse_grade(content)
        asked = {_norm(q) for q in state.get("asked") or []}
        new = [q for q in grade.queries if _norm(q) not in asked][:max_queries]
        return {"sufficient": grade.sufficient, "missing": grade.missing, "pending": new}

    def next_step(state: AgentState) -> str:
        if state.get("sufficient") or not state.get("pending"):
            return END
        return "retrieve"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("grade", grade_step)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", next_step, ["retrieve", END])
    return graph.compile()


def run_agent(agent: Any, question: str) -> AgentState:
    return agent.invoke(
        {"question": question, "pending": [question], "asked": [], "docs": [], "rounds": 0},
        config={"run_name": "agentic_retrieval"},
    )
