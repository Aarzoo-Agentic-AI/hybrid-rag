"""Test doubles for the LLM layer."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx
import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from hybrid_rag.llm.circuit_breaker import CircuitBreaker
from hybrid_rag.llm.failover import FailoverChatModel, ProviderSlot

_REQUEST = httpx.Request("POST", "https://example.invalid/v1/chat")


def status_error(cls: type[Exception], status: int) -> Exception:
    """Build a real SDK status error (e.g. ``openai.RateLimitError``) with a given status code."""
    return cls(f"HTTP {status}", response=httpx.Response(status, request=_REQUEST), body=None)


def openai_rate_limit() -> Exception:
    return status_error(openai.RateLimitError, 429)


def openai_server_error() -> Exception:
    return status_error(openai.InternalServerError, 500)


def openai_auth_error() -> Exception:
    return status_error(openai.AuthenticationError, 401)


def openai_bad_request() -> Exception:
    return status_error(openai.BadRequestError, 400)


def openai_timeout() -> Exception:
    return openai.APITimeoutError(request=_REQUEST)


def anthropic_overloaded() -> Exception:
    return status_error(anthropic.APIStatusError, 529)


@dataclass
class MidStreamFailure:
    """Script item: stream these tokens, then raise ``error``."""

    tokens: list[str]
    error: Exception


@dataclass
class Script:
    """Shared list of outcomes: a ``str`` answer, an ``Exception`` or a ``MidStreamFailure``."""

    items: list[Any]
    calls: int = 0
    seen: list[list[BaseMessage]] = field(default_factory=list)

    def next(self, messages: list[BaseMessage]) -> Any:
        self.calls += 1
        self.seen.append(messages)
        if not self.items:
            raise AssertionError("ScriptedChatModel called more times than scripted")
        return self.items.pop(0)


class ScriptedChatModel(BaseChatModel):
    """A chat model that follows a script: each call takes the next outcome."""

    script: Script

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        item = self.script.next(messages)
        if isinstance(item, MidStreamFailure):
            raise item.error
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=item))])

    def _stream(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> Iterator[ChatGenerationChunk]:
        item = self.script.next(messages)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, MidStreamFailure):
            for token in item.tokens:
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
            raise item.error
        for token in item.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=token + " "))


class FakeClock:
    """A clock that only moves when told to (or when ``sleep`` is called)."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    async def async_sleep(self, seconds: float) -> None:
        self.sleep(seconds)


def make_failover(
    *scripts: Script,
    clock: FakeClock | None = None,
    failure_threshold: int = 100,
    cooldown_s: float = 30.0,
    **kwargs: Any,
) -> tuple[FailoverChatModel, list[CircuitBreaker], FakeClock]:
    """Build a FailoverChatModel over scripted providers named p0, p1, ..."""
    clock = clock or FakeClock()
    breakers = [
        CircuitBreaker(f"p{i}/m", failure_threshold, cooldown_s, clock=clock)
        for i in range(len(scripts))
    ]
    slots = [
        ProviderSlot(f"p{i}", "m", ScriptedChatModel(script=s), breakers[i])
        for i, s in enumerate(scripts)
    ]
    model = FailoverChatModel(
        slots=slots,
        sleep=clock.sleep,
        async_sleep=clock.async_sleep,
        clock=clock,
        random=lambda: 1.0,  # no jitter: delay == cap
        **kwargs,
    )
    return model, breakers, clock
