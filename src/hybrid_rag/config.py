"""Application settings.

Values are loaded in this order (later sources win):
1. Defaults defined below
2. ``config.yaml`` (or the file named by ``HYBRID_RAG_CONFIG``)
3. ``.env`` file
4. Environment variables (``HYBRID_RAG_`` prefix, ``__`` between nested levels)

API keys are read from their conventional names (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
``QDRANT_API_KEY``) and never from ``config.yaml``.

LangSmith settings (``LANGSMITH_*``) are read by LangChain directly from the process
environment, so ``get_settings()`` copies them from ``.env`` into ``os.environ``.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import AliasChoices, BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

CONFIG_ENV_VAR = "HYBRID_RAG_CONFIG"
DOTENV_PATH = Path(".env")
# Variables read straight from os.environ by LangChain/LangSmith rather than by Settings.
TRACING_ENV_PREFIXES = ("LANGSMITH_", "LANGCHAIN_")
DEFAULT_CONFIG_PATH = Path("config.yaml")

ProviderName = Literal["anthropic", "openai", "fake"]


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    collection: str = "documents"


class EmbeddingSettings(BaseModel):
    # "bge-m3": the real model (about 2.3 GB, downloaded on first use).
    # "hashing": a tiny deterministic stand-in for tests and offline demos; no download,
    # but no real understanding of meaning either.
    provider: Literal["bge-m3", "hashing"] = "bge-m3"
    model: str = "BAAI/bge-m3"
    batch_size: int = Field(default=16, gt=0)
    max_length: int = Field(default=1024, gt=0)  # tokens per chunk; longer text is truncated
    device: str = "cpu"
    cache_size: int = Field(default=4096, ge=0)  # encoded texts kept in memory


class RerankerSettings(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"


class ChunkingSettings(BaseModel):
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    contextual_headers: bool = True  # prepend "Document: ... / Location: ..." to each chunk

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class RetrievalSettings(BaseModel):
    dense_limit: int = Field(default=50, gt=0)
    sparse_limit: int = Field(default=50, gt=0)
    fused_limit: int = Field(default=20, gt=0)
    top_k: int = Field(default=6, gt=0)

    @model_validator(mode="after")
    def _top_k_within_fused(self) -> RetrievalSettings:
        if self.top_k > self.fused_limit:
            raise ValueError("top_k must not exceed fused_limit")
        return self


class ModelTarget(BaseModel):
    provider: ProviderName
    model: str


class RetrySettings(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    base_delay_s: float = Field(default=0.5, ge=0)
    max_delay_s: float = Field(default=8.0, ge=0)


class CircuitBreakerSettings(BaseModel):
    failure_threshold: int = Field(default=5, ge=1)
    cooldown_s: float = Field(default=30.0, ge=0)


class LLMSettings(BaseModel):
    answer: list[ModelTarget] = Field(
        default_factory=lambda: [ModelTarget(provider="openai", model="gpt-5")],
        min_length=1,
    )
    query_rewrite: list[ModelTarget] = Field(
        default_factory=lambda: [ModelTarget(provider="openai", model="gpt-5-mini")],
        min_length=1,
    )
    request_timeout_s: float = Field(default=60.0, gt=0)
    overall_deadline_s: float = Field(default=120.0, gt=0)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    circuit_breaker: CircuitBreakerSettings = Field(default_factory=CircuitBreakerSettings)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HYBRID_RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("QDRANT_API_KEY")
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)
        # Earlier sources take priority.
        return (init_settings, env_settings, dotenv_settings, yaml_settings, file_secret_settings)


def export_tracing_env(
    dotenv_path: Path = DOTENV_PATH, environ: MutableMapping[str, str] | None = None
) -> list[str]:
    """Copy ``LANGSMITH_*`` / ``LANGCHAIN_*`` values from ``.env`` into ``os.environ``.

    Variables already set in the environment win. Returns the names that were exported.
    ``environ`` exists for tests: LangSmith caches these variables, so tests must not
    change the real environment.
    """
    environ = os.environ if environ is None else environ
    if not dotenv_path.is_file():
        return []
    exported = []
    for name, value in dotenv_values(dotenv_path).items():
        if name.startswith(TRACING_ENV_PREFIXES) and value and name not in environ:
            environ[name] = value
            exported.append(name)
    return exported


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (cached). Also exports LangSmith settings from .env."""
    export_tracing_env()
    return Settings()
