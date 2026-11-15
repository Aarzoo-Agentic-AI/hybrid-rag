"""``FailoverChatModel``: a LangChain chat model that tries several provider models in order.

Rules (see FEATURE.md, section 9):
- Temporary errors are retried with exponential backoff and jitter, then the next provider
  is tried.
- Fatal errors (bad request, authentication, unknown exceptions) are raised immediately.
- Each provider has a circuit breaker; providers whose circuit is open are skipped.
- An overall deadline limits the total time spent across all attempts.
- A stream can fail over only before its first token. After that, a failure raises
  ``StreamInterruptedError`` instead of mixing two models' output.

The provider that answered is recorded in ``AIMessage.response_metadata["failover"]``.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import Field

from hybrid_rag.llm.circuit_breaker import CircuitBreaker, CircuitState
from hybrid_rag.llm.errors import (
    AllProvidersFailedError,
    AttemptFailure,
    ErrorKind,
    StreamInterruptedError,
    classify_error,
)

logger = logging.getLogger(__name__)

FAILOVER_METADATA_KEY = "failover"


@dataclass(frozen=True)
class _StreamParent:
    """The failover run's identity during ``stream()``, for nesting provider attempts under it.

    ``BaseChatModel.stream()`` does not pass its run manager to ``_stream()``, so ``stream()``
    and ``astream()`` below hand this to ``_stream()`` through ``_STREAM_PARENT`` instead.
    It has the same attribute names as an LLM run manager, so ``_child_config`` accepts either.
    """

    run_id: uuid.UUID
    inheritable_handlers: list[Any]
    inheritable_tags: list[str]
    inheritable_metadata: dict[str, Any]
    is_async: bool


_STREAM_PARENT: contextvars.ContextVar[_StreamParent | None] = contextvars.ContextVar(
    "failover_stream_parent", default=None
)


class ProviderSlot:
    """One provider model in the failover order, with its circuit breaker."""

    __slots__ = ("breaker", "model", "model_name", "provider")

    def __init__(
        self, provider: str, model_name: str, model: BaseChatModel, breaker: CircuitBreaker
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.model = model
        self.breaker = breaker

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model_name}"

    def __repr__(self) -> str:
        return f"ProviderSlot({self.label!r}, state={self.breaker.state.value})"


class FailoverChatModel(BaseChatModel):
    """Chat model that retries and fails over across an ordered list of provider models."""

    slots: list[ProviderSlot] = Field(min_length=1)
    max_attempts: int = Field(default=3, ge=1)
    base_delay_s: float = Field(default=0.5, ge=0)
    max_delay_s: float = Field(default=8.0, ge=0)
    overall_deadline_s: float | None = Field(default=120.0, gt=0)

    # Injectable for tests.
    sleep: Callable[[float], None] = Field(default=time.sleep, exclude=True)
    async_sleep: Callable[[float], Awaitable[None]] = Field(default=asyncio.sleep, exclude=True)
    clock: Callable[[], float] = Field(default=time.monotonic, exclude=True)
    random: Callable[[], float] = Field(default=random.random, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "failover"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"providers": [s.label for s in self.slots]}

    # ------------------------------------------------- stream() wrappers

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        config, parent = self._prepare_stream_parent(config, is_async=False)
        inner = super().stream(input, config, stop=stop, **kwargs)
        while True:
            token = _STREAM_PARENT.set(parent)  # set only while producing the next chunk
            try:
                chunk = next(inner)
            except StopIteration:
                return
            finally:
                _STREAM_PARENT.reset(token)
            yield chunk

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        config, parent = self._prepare_stream_parent(config, is_async=True)
        inner = super().astream(input, config, stop=stop, **kwargs)
        while True:
            token = _STREAM_PARENT.set(parent)
            try:
                chunk = await anext(inner)
            except StopAsyncIteration:
                return
            finally:
                _STREAM_PARENT.reset(token)
            yield chunk

    def _prepare_stream_parent(
        self, config: RunnableConfig | None, *, is_async: bool
    ) -> tuple[RunnableConfig, _StreamParent]:
        """Fix the run id and callback handlers that ``BaseChatModel.stream()`` will use.

        Handlers are configured here (including the LangSmith tracer when tracing is on) and
        passed in as a callback manager, so the failover run and its provider attempts share
        the same handler instances. Tracers need that to link a child run to its parent.
        """
        config = ensure_config(config)
        manager_cls = AsyncCallbackManager if is_async else CallbackManager
        manager = manager_cls.configure(inheritable_callbacks=config.get("callbacks"))
        run_id = config.get("run_id") or uuid.uuid4()
        parent = _StreamParent(
            run_id=run_id,
            inheritable_handlers=list(manager.inheritable_handlers),
            inheritable_tags=list(config.get("tags") or []),
            inheritable_metadata=dict(config.get("metadata") or {}),
            is_async=is_async,
        )
        return {**config, "callbacks": manager, "run_id": run_id}, parent

    # ------------------------------------------------------------------ sync

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        run = _Run(self)
        for index, slot in run.available_slots():
            for attempt in range(1, self.max_attempts + 1):
                run.check_deadline()
                started = self.clock()
                try:
                    message = slot.model.invoke(
                        messages,
                        stop=stop,
                        config=_child_config(run_manager, slot, attempt),
                        **kwargs,
                    )
                except Exception as exc:
                    delay = run.handle_failure(slot, attempt, exc)
                    if delay is None:
                        break
                    self.sleep(delay)
                    continue
                slot.breaker.record_success()
                info = run.success_info(index, slot, attempt, started)
                return ChatResult(generations=[ChatGeneration(message=_annotate(message, info))])
        raise run.exhausted()

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        parent = run_manager or _STREAM_PARENT.get()
        run = _Run(self)
        for index, slot in run.available_slots():
            for attempt in range(1, self.max_attempts + 1):
                run.check_deadline()
                started = self.clock()
                iterator = iter(
                    slot.model.stream(
                        messages,
                        stop=stop,
                        config=_child_config(parent, slot, attempt),
                        **kwargs,
                    )
                )
                # Failover is only allowed until the first chunk arrives.
                try:
                    chunk: AIMessageChunk | None = next(iterator)
                except StopIteration:
                    chunk = None
                except Exception as exc:
                    delay = run.handle_failure(slot, attempt, exc)
                    if delay is None:
                        break
                    self.sleep(delay)
                    continue

                # Committed to this provider from here on.
                slot.breaker.record_success()
                while chunk is not None:
                    yield _emit(chunk, run_manager)
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        chunk = None
                    except Exception as exc:
                        raise _interrupted(slot, exc) from exc
                info = run.success_info(index, slot, attempt, started)
                yield _metadata_chunk(info)
                return
        raise run.exhausted()

    # ----------------------------------------------------------------- async

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        run = _Run(self)
        for index, slot in run.available_slots():
            for attempt in range(1, self.max_attempts + 1):
                run.check_deadline()
                started = self.clock()
                try:
                    message = await slot.model.ainvoke(
                        messages,
                        stop=stop,
                        config=_child_config(run_manager, slot, attempt),
                        **kwargs,
                    )
                except Exception as exc:
                    delay = run.handle_failure(slot, attempt, exc)
                    if delay is None:
                        break
                    await self.async_sleep(delay)
                    continue
                slot.breaker.record_success()
                info = run.success_info(index, slot, attempt, started)
                return ChatResult(generations=[ChatGeneration(message=_annotate(message, info))])
        raise run.exhausted()

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        parent = run_manager or _STREAM_PARENT.get()
        run = _Run(self)
        for index, slot in run.available_slots():
            for attempt in range(1, self.max_attempts + 1):
                run.check_deadline()
                started = self.clock()
                iterator = aiter(
                    slot.model.astream(
                        messages,
                        stop=stop,
                        config=_child_config(parent, slot, attempt),
                        **kwargs,
                    )
                )
                try:
                    chunk: AIMessageChunk | None = await anext(iterator)
                except StopAsyncIteration:
                    chunk = None
                except Exception as exc:
                    delay = run.handle_failure(slot, attempt, exc)
                    if delay is None:
                        break
                    await self.async_sleep(delay)
                    continue

                slot.breaker.record_success()
                while chunk is not None:
                    gen_chunk = ChatGenerationChunk(message=chunk)
                    if run_manager:
                        await run_manager.on_llm_new_token(gen_chunk.text, chunk=gen_chunk)
                    yield gen_chunk
                    try:
                        chunk = await anext(iterator)
                    except StopAsyncIteration:
                        chunk = None
                    except Exception as exc:
                        raise _interrupted(slot, exc) from exc
                info = run.success_info(index, slot, attempt, started)
                yield _metadata_chunk(info)
                return
        raise run.exhausted()


class _Run:
    """Book-keeping for a single request across all its attempts."""

    def __init__(self, model: FailoverChatModel) -> None:
        self.model = model
        self.started = model.clock()
        self.failures: list[AttemptFailure] = []
        self.skipped: list[str] = []

    def available_slots(self) -> Iterator[tuple[int, ProviderSlot]]:
        for index, slot in enumerate(self.model.slots):
            if slot.breaker.allow_request():
                yield index, slot
            else:
                logger.warning("Skipping %s: circuit open", slot.label)
                self.skipped.append(slot.label)

    def remaining_s(self) -> float | None:
        if self.model.overall_deadline_s is None:
            return None
        return self.model.overall_deadline_s - (self.model.clock() - self.started)

    def check_deadline(self) -> None:
        remaining = self.remaining_s()
        if remaining is not None and remaining <= 0:
            raise AllProvidersFailedError(
                self.failures,
                self.skipped,
                reason=f"overall deadline of {self.model.overall_deadline_s}s exceeded",
            )

    def handle_failure(self, slot: ProviderSlot, attempt: int, exc: Exception) -> float | None:
        """Record a failed attempt.

        Raises ``exc`` again if it is fatal. Otherwise returns the delay before retrying the
        same provider, or ``None`` to move on to the next provider.
        """
        if classify_error(exc) is ErrorKind.FATAL:
            slot.breaker.release_probe()
            logger.error("Fatal error from %s, not failing over: %r", slot.label, exc)
            raise exc

        self.failures.append(AttemptFailure(slot.provider, slot.model_name, attempt, exc))
        slot.breaker.record_failure()

        if slot.breaker.state is CircuitState.OPEN:
            logger.warning("%s failed (%r); circuit opened, moving on", slot.label, exc)
            return None
        if attempt >= self.model.max_attempts:
            logger.warning("%s failed %d times (%r), moving on", slot.label, attempt, exc)
            return None

        delay = self.backoff(attempt)
        remaining = self.remaining_s()
        if remaining is not None and delay >= remaining:
            logger.warning("%s failed (%r); no time left to retry, moving on", slot.label, exc)
            return None
        logger.warning(
            "%s failed (%r); retry %d/%d in %.2fs",
            slot.label,
            exc,
            attempt + 1,
            self.model.max_attempts,
            delay,
        )
        return delay

    def backoff(self, attempt: int) -> float:
        """Exponential backoff with "equal jitter": between half and all of the capped delay."""
        cap = min(self.model.max_delay_s, self.model.base_delay_s * (2 ** (attempt - 1)))
        return cap / 2 + self.model.random() * cap / 2

    def success_info(
        self, index: int, slot: ProviderSlot, attempt: int, started: float
    ) -> dict[str, Any]:
        info = {
            "provider": slot.provider,
            "model": slot.model_name,
            "failed_over": index > 0,
            "attempts": len(self.failures) + 1,
            "latency_s": round(self.model.clock() - started, 3),
            "errors": [str(f) for f in self.failures],
            "skipped": list(self.skipped),
        }
        if info["failed_over"] or self.failures:
            logger.info("Answered by %s after %d attempt(s)", slot.label, info["attempts"])
        return info

    def exhausted(self) -> AllProvidersFailedError:
        reason = "every provider failed" if self.failures else "every provider's circuit is open"
        return AllProvidersFailedError(self.failures, self.skipped, reason=reason)


def _child_config(
    parent: CallbackManagerForLLMRun | AsyncCallbackManagerForLLMRun | _StreamParent | None,
    slot: ProviderSlot,
    attempt: int,
) -> RunnableConfig:
    """Nest each attempt under the failover run, so traces (e.g. LangSmith) show every try."""
    config: RunnableConfig = {"run_name": f"{slot.label} (attempt {attempt})"}
    if parent is not None:
        # LLM run managers have no get_child(); this mirrors ParentRunManager.get_child().
        is_async = (
            parent.is_async
            if isinstance(parent, _StreamParent)
            else isinstance(parent, AsyncCallbackManagerForLLMRun)
        )
        manager_cls = AsyncCallbackManager if is_async else CallbackManager
        child = manager_cls(handlers=[], parent_run_id=parent.run_id)
        child.set_handlers(parent.inheritable_handlers)
        child.add_tags(parent.inheritable_tags)
        child.add_metadata(parent.inheritable_metadata)
        config["callbacks"] = child
    return config


def _annotate(message: BaseMessage, info: dict[str, Any]) -> AIMessage:
    if not isinstance(message, AIMessage):
        message = AIMessage(content=message.content)
    message.response_metadata = {**message.response_metadata, FAILOVER_METADATA_KEY: info}
    return message


def _emit(
    chunk: AIMessageChunk, run_manager: CallbackManagerForLLMRun | None
) -> ChatGenerationChunk:
    gen_chunk = ChatGenerationChunk(message=chunk)
    if run_manager:
        run_manager.on_llm_new_token(gen_chunk.text, chunk=gen_chunk)
    return gen_chunk


def _metadata_chunk(info: dict[str, Any]) -> ChatGenerationChunk:
    """An empty final chunk carrying the failover metadata; it merges into the full message."""
    return ChatGenerationChunk(
        message=AIMessageChunk(content="", response_metadata={FAILOVER_METADATA_KEY: info})
    )


def _interrupted(slot: ProviderSlot, exc: Exception) -> StreamInterruptedError:
    if classify_error(exc) is ErrorKind.RETRYABLE:
        slot.breaker.record_failure()
    logger.error("Stream from %s failed mid-answer: %r", slot.label, exc)
    return StreamInterruptedError(slot.provider, slot.model_name, exc)
