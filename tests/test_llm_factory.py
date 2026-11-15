from pathlib import Path

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from hybrid_rag.config import CONFIG_ENV_VAR, Settings
from hybrid_rag.llm import LLMFactory, NoProvidersConfiguredError

TWO_PROVIDERS = """
llm:
  answer:
    - {provider: openai, model: gpt-test}
    - {provider: anthropic, model: claude-test}
  query_rewrite:
    - {provider: openai, model: gpt-test}
  request_timeout_s: 12
  retry: {max_attempts: 4, base_delay_s: 0.1, max_delay_s: 1}
"""


@pytest.fixture
def settings_from(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)  # ignore the developer's real .env
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", CONFIG_ENV_VAR):
        monkeypatch.delenv(var, raising=False)

    def build(yaml_text: str, **keys: str) -> Settings:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml_text, encoding="utf-8")
        monkeypatch.setenv(CONFIG_ENV_VAR, str(cfg))
        for name, value in keys.items():
            monkeypatch.setenv(name, value)
        return Settings()

    return build


def test_builds_both_providers_in_order(settings_from) -> None:
    settings = settings_from(TWO_PROVIDERS, OPENAI_API_KEY="sk-1", ANTHROPIC_API_KEY="sk-2")
    model = LLMFactory(settings).for_task("answer")

    assert [s.label for s in model.slots] == ["openai/gpt-test", "anthropic/claude-test"]
    assert isinstance(model.slots[0].model, ChatOpenAI)
    assert isinstance(model.slots[1].model, ChatAnthropic)
    assert model.max_attempts == 4


def test_sdk_retries_are_disabled_and_timeout_is_set(settings_from) -> None:
    settings = settings_from(TWO_PROVIDERS, OPENAI_API_KEY="sk-1", ANTHROPIC_API_KEY="sk-2")
    model = LLMFactory(settings).for_task("answer")

    openai_model, anthropic_model = (s.model for s in model.slots)
    assert openai_model.max_retries == 0
    assert openai_model.request_timeout == 12
    assert anthropic_model.max_retries == 0
    assert anthropic_model.default_request_timeout == 12


def test_provider_without_key_is_skipped(settings_from) -> None:
    settings = settings_from(TWO_PROVIDERS, OPENAI_API_KEY="sk-1")
    model = LLMFactory(settings).for_task("answer")

    assert [s.label for s in model.slots] == ["openai/gpt-test"]


def test_no_keys_at_all_raises(settings_from) -> None:
    settings = settings_from(TWO_PROVIDERS)

    with pytest.raises(NoProvidersConfiguredError):
        LLMFactory(settings).for_task("answer")


def test_tasks_share_circuit_breakers_for_the_same_model(settings_from) -> None:
    settings = settings_from(TWO_PROVIDERS, OPENAI_API_KEY="sk-1")
    factory = LLMFactory(settings)

    answer = factory.for_task("answer")
    rewrite = factory.for_task("query_rewrite")

    assert answer.slots[0].breaker is rewrite.slots[0].breaker
    assert factory.for_task("answer") is answer  # built once, then reused
    assert factory.health()[0]["state"] == "closed"


def test_fake_provider_needs_no_key(settings_from) -> None:
    settings = settings_from(
        "llm:\n  answer:\n    - {provider: fake, model: demo}\n"
        "  query_rewrite:\n    - {provider: fake, model: demo}\n"
    )
    model = LLMFactory(settings).for_task("answer")

    assert "fake:demo" in model.invoke("hi").content
