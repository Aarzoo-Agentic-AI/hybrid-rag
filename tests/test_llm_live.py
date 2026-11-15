"""Live calls to the real providers. Run with: uv run pytest -m integration"""

import pytest

from hybrid_rag.config import Settings
from hybrid_rag.llm import FAILOVER_METADATA_KEY, LLMFactory, NoProvidersConfiguredError

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def factory() -> LLMFactory:
    return LLMFactory(Settings())


@pytest.mark.parametrize("task", ["answer", "query_rewrite"])
def test_real_provider_answers(factory: LLMFactory, task: str) -> None:
    try:
        model = factory.for_task(task)  # type: ignore[arg-type]
    except NoProvidersConfiguredError:
        pytest.skip("no LLM API key configured")

    message = model.invoke("Reply with exactly one word: pong")

    assert "pong" in str(message.content).lower()
    assert message.response_metadata[FAILOVER_METADATA_KEY]["provider"] in {"openai", "anthropic"}
