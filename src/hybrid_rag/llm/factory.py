"""Build LangChain chat models and ``FailoverChatModel``s from the settings."""

from __future__ import annotations

import logging
import threading
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from hybrid_rag.config import ModelTarget, Settings
from hybrid_rag.llm.circuit_breaker import CircuitBreakerRegistry
from hybrid_rag.llm.errors import NoProvidersConfiguredError
from hybrid_rag.llm.failover import FailoverChatModel, ProviderSlot

logger = logging.getLogger(__name__)

Task = Literal["answer", "query_rewrite"]


def build_provider_model(target: ModelTarget, settings: Settings) -> BaseChatModel | None:
    """Create the chat model for one provider entry, or ``None`` if its API key is missing.

    The SDKs' own retries are turned off (``max_retries=0``) so that all retry decisions
    are made by ``FailoverChatModel``.
    """
    timeout = settings.llm.request_timeout_s

    if target.provider == "openai":
        if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
            return None
        from langchain_openai import ChatOpenAI

        extra = {"reasoning_effort": target.reasoning_effort} if target.reasoning_effort else {}
        return ChatOpenAI(
            model=target.model,
            api_key=settings.openai_api_key,
            timeout=timeout,
            max_retries=0,
            stream_usage=True,
            **extra,
        )

    if target.provider == "anthropic":
        if settings.anthropic_api_key is None or not settings.anthropic_api_key.get_secret_value():
            return None
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=target.model,
            api_key=settings.anthropic_api_key,
            timeout=timeout,
            max_retries=0,
        )

    if target.provider == "fake":
        # For local development and demos without API keys.
        return FakeListChatModel(responses=[f"[fake:{target.model}] This is a fake response."])

    raise ValueError(f"Unknown provider: {target.provider}")


class LLMFactory:
    """Builds one ``FailoverChatModel`` per task and shares circuit breakers between them.

    Create one factory per process (e.g. at API startup) so that every request sees the
    same circuit-breaker state.
    """

    def __init__(self, settings: Settings, registry: CircuitBreakerRegistry | None = None) -> None:
        self.settings = settings
        cb = settings.llm.circuit_breaker
        self.registry = registry or CircuitBreakerRegistry(
            failure_threshold=cb.failure_threshold, cooldown_s=cb.cooldown_s
        )
        self._models: dict[Task, FailoverChatModel] = {}
        self._lock = threading.Lock()

    def for_task(self, task: Task) -> FailoverChatModel:
        with self._lock:
            model = self._models.get(task)
            if model is None:
                model = self._build(task)
                self._models[task] = model
            return model

    def _build(self, task: Task) -> FailoverChatModel:
        targets: list[ModelTarget] = getattr(self.settings.llm, task)
        slots: list[ProviderSlot] = []
        missing_keys: list[str] = []

        for target in targets:
            model = build_provider_model(target, self.settings)
            label = f"{target.provider}/{target.model}"
            if model is None:
                missing_keys.append(label)
                continue
            slots.append(
                ProviderSlot(target.provider, target.model, model, self.registry.get(label))
            )

        if missing_keys:
            logger.info("Task %r: skipping %s (no API key)", task, ", ".join(missing_keys))
        if not slots:
            raise NoProvidersConfiguredError(
                f"No LLM provider for task {task!r} has an API key. "
                f"Configured: {', '.join(missing_keys)}. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
            )

        llm = self.settings.llm
        return FailoverChatModel(
            slots=slots,
            max_attempts=llm.retry.max_attempts,
            base_delay_s=llm.retry.base_delay_s,
            max_delay_s=llm.retry.max_delay_s,
            overall_deadline_s=llm.overall_deadline_s,
        )

    def health(self) -> list[dict[str, object]]:
        """Circuit-breaker state of every provider model built so far (for ``/health``)."""
        return self.registry.snapshot()
