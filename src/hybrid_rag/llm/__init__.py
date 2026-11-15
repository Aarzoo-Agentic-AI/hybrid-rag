"""LLM layer: LangChain chat model factory and FailoverChatModel."""

from hybrid_rag.llm.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState
from hybrid_rag.llm.errors import (
    AllProvidersFailedError,
    ErrorKind,
    LLMError,
    NoProvidersConfiguredError,
    StreamInterruptedError,
    classify_error,
)
from hybrid_rag.llm.factory import LLMFactory, Task, build_provider_model
from hybrid_rag.llm.failover import FAILOVER_METADATA_KEY, FailoverChatModel, ProviderSlot

__all__ = [
    "FAILOVER_METADATA_KEY",
    "AllProvidersFailedError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
    "ErrorKind",
    "FailoverChatModel",
    "LLMError",
    "LLMFactory",
    "NoProvidersConfiguredError",
    "ProviderSlot",
    "StreamInterruptedError",
    "Task",
    "build_provider_model",
    "classify_error",
]
