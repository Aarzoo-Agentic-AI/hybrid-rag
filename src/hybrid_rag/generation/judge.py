"""The faithfulness check used by runtime evaluation: is every claim in an answer supported by
the sources it was written from?

An LLM judge (``llm.judge``, a small model) reads the numbered sources and the answer and
returns JSON: a score from 0 to 1 and the claims it couldn't find in the sources. Parsing is
defensive: a reply that isn't valid JSON gives ``None``, never an exception, because this runs
in the background on live traffic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check answers written by an HR policy assistant. You get the numbered policy "
            "sources the assistant was given and its answer. Decide whether every factual claim "
            "in the answer (numbers, limits, dates, eligibility, who approves what) is supported "
            "by the sources. Ignore citation markers like [1], wording and style, and the "
            "standard pointer to HR. Saying the policies don't cover something is supported if "
            "the sources indeed don't cover it.\n\n"
            "Reply with JSON only, no other text:\n"
            '{{"score": <0 to 1: the share of claims supported>, '
            '"unsupported": [<each unsupported claim, quoted briefly>]}}',
        ),
        ("human", "Sources:\n{sources}\n\nAnswer:\n{answer}"),
    ]
)

_JSON = re.compile(r"\{.*\}", re.DOTALL)
MAX_CLAIMS = 5
MAX_CLAIM_CHARS = 200


@dataclass(frozen=True)
class Faithfulness:
    score: float  # 0 to 1
    unsupported: list[str] = field(default_factory=list)


def parse_judgement(text: str) -> Faithfulness | None:
    """The judge's JSON reply, or None if it can't be read."""
    match = _JSON.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        score = float(data["score"])
    except (ValueError, KeyError, TypeError):
        return None
    if not 0 <= score <= 1:
        return None
    claims = data.get("unsupported") or []
    if not isinstance(claims, list):
        claims = [claims]
    return Faithfulness(
        score=score,
        unsupported=[str(c)[:MAX_CLAIM_CHARS] for c in claims if str(c).strip()][:MAX_CLAIMS],
    )


def judge_faithfulness(llm: BaseChatModel, sources: str, answer: str) -> Faithfulness | None:
    chain = (JUDGE_PROMPT | llm).with_config(run_name="judge_faithfulness")
    reply = chain.invoke({"sources": sources, "answer": answer})
    content = reply.content
    if not isinstance(content, str):  # some providers return a list of content parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return parse_judgement(content)
