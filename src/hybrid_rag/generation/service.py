"""``AnswerService``: from a question to a cited answer, for one caller.

```
question ─► redact ─► rewrite ─► retrieve ─► sources ─► confident? ──no──► "contact HR"
            personal  follow-ups  (step 4,   numbered,   top rerank         (no LLM call)
            data      (history)   caller's   budget,     score
                                  filter)    deduped        │yes
                                                            ▼
                         answer ◄─ output ◄─ citations ◄─ prompt | FailoverChatModel
                                   check     [n] -> source
```

The middle is one LCEL chain (``RunnablePassthrough.assign`` -> ``RunnableBranch`` ->
``ChatPromptTemplate | FailoverChatModel``), so LangSmith shows each step as a nested run
under one ``answer_question`` trace. Redaction happens *before* the trace starts, so the
question in the trace is the redacted one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from datetime import date
from operator import itemgetter
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langsmith.run_helpers import get_current_run_tree

from hybrid_rag.access import Principal
from hybrid_rag.config import RetrievalMode, Settings
from hybrid_rag.generation.citations import parse_citations
from hybrid_rag.generation.context import Source, format_sources, select_sources
from hybrid_rag.generation.prompts import ANSWER_PROMPT, NOT_FOUND, REWRITE_PROMPT
from hybrid_rag.generation.types import Answer, AnswerStatus, UsedSource
from hybrid_rag.ingestion.tokens import TokenCounter, get_token_counter
from hybrid_rag.llm import LLMFactory
from hybrid_rag.llm.failover import FAILOVER_METADATA_KEY
from hybrid_rag.privacy import Sanitizer, build_sanitizer
from hybrid_rag.retrieval import RelevanceFilters, build_retriever
from hybrid_rag.store import DocumentIndex
from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)

History = Sequence[BaseMessage | tuple[str, str]]

UNGROUNDED_NOTE = (
    "\n\n(This answer could not be linked to a policy source. Please confirm it with {contact}.)"
)


def _fallback(status: AnswerStatus, contact: str) -> str:
    # The same words whether nothing matched or the caller may not read what matched, so the
    # answer never reveals that a restricted policy exists.
    if status == "not_found":
        return f"The HR policies I found don't answer this question. Please contact {contact}."
    return (
        "I couldn't find anything about this in the HR policies available to you. "
        f"Please contact {contact}."
    )


def _answer_summary(a: Answer | None) -> dict[str, Any]:
    if a is None:
        return {}
    return {
        "status": a.status,
        "grounded": a.grounded,
        "citations": [c.label for c in a.citations],
        "sources": len(a.sources),
        "provider": (a.provider or {}).get("provider"),
        "redactions": a.redactions,
        "output_redactions": a.output_redactions,
        "rewritten": a.rewritten,
    }


class AnswerService:
    """Answers questions for callers. Create once (it holds the models) and reuse."""

    def __init__(
        self,
        settings: Settings,
        index: DocumentIndex,
        answer_llm: BaseChatModel,
        *,
        rewrite_llm: BaseChatModel | None = None,
        sanitizer: Sanitizer | None = None,
        tokens: TokenCounter | None = None,
        cross_encoder: Any = None,  # tests pass a fake reranker
        today: date | None = None,
    ) -> None:
        self.settings = settings
        self.index = index
        self.answer_llm = answer_llm
        self.rewrite_llm = rewrite_llm
        self.sanitizer = sanitizer or build_sanitizer(settings.privacy)
        self.tokens = tokens or get_token_counter(settings.chunking.tokenizer)
        self.cross_encoder = cross_encoder
        self._today = today
        # The prompt gets only its own fields, never the retrieved Documents themselves.
        self.generate = (RunnableLambda(_prompt_inputs) | ANSWER_PROMPT | answer_llm).with_config(
            run_name="generate"
        )

    @classmethod
    def from_settings(
        cls, settings: Settings, index: DocumentIndex, factory: LLMFactory | None = None
    ) -> AnswerService:
        factory = factory or LLMFactory(settings)
        return cls(
            settings,
            index,
            factory.for_task("answer"),
            rewrite_llm=factory.for_task("query_rewrite"),
        )

    @property
    def today(self) -> date:
        return self._today or date.today()

    # ------------------------------------------------------------ public

    def answer(
        self,
        question: str,
        principal: Principal,
        *,
        history: History | None = None,
        mode: RetrievalMode | None = None,
        filters: RelevanceFilters | None = None,
    ) -> Answer:
        """Answer ``question`` for ``principal`` from the policies they may read."""
        clean, messages, redactions = self._redact(question, history)
        return self._answer(clean, principal, messages, redactions, mode, filters)

    def stream(
        self,
        question: str,
        principal: Principal,
        *,
        history: History | None = None,
        mode: RetrievalMode | None = None,
        filters: RelevanceFilters | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield ``{"type": "token", "text": ...}`` events while the answer is written, then
        one ``{"type": "answer", "answer": Answer}``. The final answer is authoritative: its
        text has invalid citation markers removed and has passed the output check."""
        clean, messages, redactions = self._redact(question, history)
        yield from self._stream(clean, principal, messages, redactions, mode, filters)

    # ------------------------------------------------------------- steps

    def _redact(
        self, question: str, history: History | None
    ) -> tuple[str, list[BaseMessage], dict[str, int]]:
        """Mask personal data in the question and the history, before anything else sees it."""
        total: dict[str, int] = {}

        def clean(text: str) -> str:
            masked, counts = self.sanitizer.redact_text(text)
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
            return masked

        turns = list(history or [])[-2 * self.settings.answer.max_history_turns :]
        messages: list[BaseMessage] = []
        for turn in turns:
            if isinstance(turn, tuple):
                role, text = turn
                cls = HumanMessage if role in ("human", "user") else AIMessage
                messages.append(cls(content=clean(text)))
            else:
                messages.append(type(turn)(content=clean(str(turn.content))))
        return clean(question.strip()), messages, total

    def _chain(
        self, principal: Principal, mode: RetrievalMode | None, filters: RelevanceFilters | None
    ) -> tuple[Runnable, Runnable]:
        """Two LCEL runnables: retrieve + build the sources, then answer (or not)."""
        retriever = build_retriever(
            self.settings,
            self.index,
            principal,
            mode=mode,
            filters=filters,
            cross_encoder=self.cross_encoder,
            tokens=self.tokens,
        )
        prepare = RunnablePassthrough.assign(
            docs=itemgetter("question") | retriever
        ) | RunnableLambda(self._prepare).with_config(run_name="build_context")
        return prepare, RunnableBranch(
            (lambda x: x["gate"] is not None, RunnableLambda(self._without_llm)),
            RunnablePassthrough.assign(message=self.generate)
            | RunnableLambda(self._finish).with_config(run_name="parse_citations"),
        )

    def _rewrite(self, question: str, history: list[BaseMessage]) -> str:
        if not history or self.rewrite_llm is None:
            return question
        chain = (REWRITE_PROMPT | self.rewrite_llm | StrOutputParser()).with_config(
            run_name="rewrite_question"
        )
        transcript = "\n".join(
            f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {_text(m)}" for m in history
        )
        rewritten = chain.invoke({"question": question, "history": transcript}).strip()
        rewritten = rewritten.strip('"').strip()
        rewritten, _ = self.sanitizer.redact_text(rewritten)  # the rewrite is new text
        return rewritten or question

    def _prepare(self, x: dict[str, Any]) -> dict[str, Any]:
        sources = select_sources(x["docs"], self.tokens, self.settings.answer.context_max_tokens)
        gate: AnswerStatus | None = None
        if not sources:
            gate = "no_sources"
        else:
            threshold = self.settings.answer.min_rerank_score
            top = sources[0].doc.metadata.get("rerank_score")
            if threshold is not None and top is not None and top < threshold:
                gate = "low_confidence"
        return {
            **x,
            "sources": sources,
            "sources_text": format_sources(sources),
            "gate": gate,
            "today": self.today.isoformat(),
            "contact": self.settings.answer.contact,
        }

    def _without_llm(self, x: dict[str, Any]) -> Answer:
        status: AnswerStatus = x["gate"]
        return Answer(
            question=x["question"],
            text=_fallback(status, self.settings.answer.contact),
            status=status,
            sources=_used(x["sources"], cited=set()),
        )

    def _finish(self, x: dict[str, Any]) -> Answer:
        message: BaseMessage = x["message"]
        return self._build_answer(x, _text(message), message.response_metadata)

    def _build_answer(
        self, x: dict[str, Any], raw: str, response_metadata: dict[str, Any]
    ) -> Answer:
        sources: list[Source] = x["sources"]
        provider = response_metadata.get(FAILOVER_METADATA_KEY)
        contact = self.settings.answer.contact
        if raw.strip().startswith(NOT_FOUND):
            return Answer(
                question=x["question"],
                text=_fallback("not_found", contact),
                status="not_found",
                sources=_used(sources, cited=set()),
                provider=provider,
            )
        text, citations, invalid = parse_citations(raw, sources)
        if invalid:
            logger.warning("Answer cited sources that weren't given: %s", sorted(set(invalid)))
        grounded = bool(citations)
        if not grounded:
            text += UNGROUNDED_NOTE.format(contact=contact)
        text, output_redactions = self.sanitizer.check_output(text)
        return Answer(
            question=x["question"],
            text=text,
            status="answered",
            citations=citations,
            grounded=grounded,
            sources=_used(sources, cited={c.number for c in citations}),
            provider=provider,
            output_redactions=output_redactions,
        )

    # ------------------------------------------------------- traced runs

    @traceable(
        name="answer_question",
        run_type="chain",
        process_inputs=lambda i: {
            "question": i["question"],  # already redacted
            "tenant": i["principal"].tenant_id,
            "history_turns": len(i["history"]),
            "mode": i["mode"],
        },
        process_outputs=_answer_summary,
    )
    def _answer(
        self,
        question: str,
        principal: Principal,
        history: list[BaseMessage],
        redactions: dict[str, int],
        mode: RetrievalMode | None,
        filters: RelevanceFilters | None,
    ) -> Answer:
        standalone = self._rewrite(question, history)
        prepare, respond = self._chain(principal, mode, filters)
        x = prepare.invoke({"question": standalone})
        answer: Answer = respond.invoke(x)
        answer.redactions = redactions
        answer.rewritten = standalone != question
        _tag_run(answer)
        return answer

    @traceable(
        name="answer_question",
        run_type="chain",
        process_inputs=lambda i: {
            "question": i["question"],
            "tenant": i["principal"].tenant_id,
            "history_turns": len(i["history"]),
            "mode": i["mode"],
            "stream": True,
        },
        reduce_fn=lambda events: _answer_summary(events[-1]["answer"]) if events else {},
    )
    def _stream(
        self,
        question: str,
        principal: Principal,
        history: list[BaseMessage],
        redactions: dict[str, int],
        mode: RetrievalMode | None,
        filters: RelevanceFilters | None,
    ) -> Iterator[dict[str, Any]]:
        standalone = self._rewrite(question, history)
        prepare, _ = self._chain(principal, mode, filters)
        x = prepare.invoke({"question": standalone})
        if x["gate"] is not None:
            answer = self._without_llm(x)
        else:
            raw, metadata = "", {}
            pending = ""  # held back until we know the reply isn't NOT_FOUND
            decided = False
            for chunk in self.generate.stream(x):
                piece = _text(chunk)
                raw += piece
                metadata |= chunk.response_metadata or {}
                if decided:
                    if piece:
                        yield {"type": "token", "text": piece}
                    continue
                pending += piece
                head = pending.lstrip()
                if head.startswith(NOT_FOUND) or NOT_FOUND.startswith(head):
                    continue  # could still be NOT_FOUND: keep holding
                decided = True
                yield {"type": "token", "text": pending}
            if not decided and pending.strip() and not pending.lstrip().startswith(NOT_FOUND):
                yield {"type": "token", "text": pending}  # a reply shorter than NOT_FOUND
            answer = self._build_answer(x, raw, metadata)
        answer.redactions = redactions
        answer.rewritten = standalone != question
        _tag_run(answer)
        yield {"type": "answer", "answer": answer}


def _tag_run(answer: Answer) -> None:
    """Put the outcome on the current trace as metadata and tags, so traces can be filtered by
    the provider that answered and by status."""
    run = get_current_run_tree()
    if run is None:
        return
    provider = answer.provider or {}
    run.add_metadata(
        {
            "answer_status": answer.status,
            "grounded": answer.grounded,
            "provider": provider.get("provider"),
            "model": provider.get("model"),
            "failed_over": provider.get("failed_over"),
        }
    )
    tags = [f"status:{answer.status}"]
    if provider.get("provider"):
        tags.append(f"provider:{provider['provider']}")
    run.add_tags(tags)


def _prompt_inputs(x: dict[str, Any]) -> dict[str, Any]:
    return {
        "sources": x["sources_text"],
        "question": x["question"],
        "today": x["today"],
        "contact": x["contact"],
    }


def _text(message: BaseMessage | AIMessageChunk) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    )


def _used(sources: Sequence[Source], *, cited: set[int]) -> list[UsedSource]:
    return [
        UsedSource(
            number=s.number,
            chunk_id=str(s.doc.metadata.get("chunk_id") or s.doc.id),
            doc_id=s.doc.metadata.get("doc_id", ""),
            source=s.doc.metadata.get("source", ""),
            access_version=s.doc.metadata.get("access_version"),
            retrieval_rank=s.doc.metadata.get("retrieval_rank"),
            rerank_score=s.doc.metadata.get("rerank_score"),
            cited=s.number in cited,
        )
        for s in sources
    ]
