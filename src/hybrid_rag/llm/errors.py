"""Error types for the LLM layer and the rules for which errors are worth retrying."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import anthropic
import httpx
import openai


class ErrorKind(Enum):
    """How the failover model should react to an error."""

    RETRYABLE = "retryable"  # temporary: retry, then fail over to the next provider
    FATAL = "fatal"  # caller error or bug: fail immediately, no failover


# HTTP status codes that indicate a temporary problem on the provider's side.
# 408 request timeout, 409 conflict (OpenAI uses it for transient lock errors),
# 425 too early, 429 rate limited. Every 5xx (including Anthropic's 529 "overloaded")
# is also treated as temporary.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429})

_RETRYABLE_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    # Connection problems and timeouts (APITimeoutError is a subclass of APIConnectionError).
    openai.APIConnectionError,
    anthropic.APIConnectionError,
    httpx.TimeoutException,
    httpx.TransportError,
    TimeoutError,
    ConnectionError,
)


def classify_error(exc: BaseException) -> ErrorKind:
    """Decide whether ``exc`` is temporary (retry or fail over) or fatal (raise now).

    Unknown exceptions are treated as fatal: they are most likely bugs in our code,
    and switching providers would only hide them.
    """
    if isinstance(exc, _RETRYABLE_EXCEPTION_TYPES):
        return ErrorKind.RETRYABLE

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code in _RETRYABLE_STATUS_CODES or status_code >= 500:
            return ErrorKind.RETRYABLE
        return ErrorKind.FATAL

    return ErrorKind.FATAL


class LLMError(Exception):
    """Base class for errors raised by the LLM layer."""


class NoProvidersConfiguredError(LLMError):
    """None of the configured providers has an API key."""


@dataclass(frozen=True)
class AttemptFailure:
    """One failed call, kept for error reporting."""

    provider: str
    model: str
    attempt: int
    error: BaseException

    def __str__(self) -> str:
        return (
            f"{self.provider}/{self.model} attempt {self.attempt}: "
            f"{type(self.error).__name__}: {self.error}"
        )


class AllProvidersFailedError(LLMError):
    """Every provider was tried (or skipped) and none produced an answer."""

    def __init__(self, failures: list[AttemptFailure], skipped: list[str], reason: str) -> None:
        self.failures = failures
        self.skipped = skipped
        self.reason = reason
        lines = [f"All LLM providers failed ({reason})."]
        lines += [f"  - {f}" for f in failures]
        if skipped:
            lines.append(f"  Skipped (circuit open): {', '.join(skipped)}")
        super().__init__("\n".join(lines))


class StreamInterruptedError(LLMError):
    """A stream failed after it had already sent tokens, so it can't fail over safely."""

    def __init__(self, provider: str, model: str, error: BaseException) -> None:
        self.provider = provider
        self.model = model
        self.error = error
        super().__init__(
            f"Stream from {provider}/{model} failed mid-answer: {type(error).__name__}: {error}"
        )
