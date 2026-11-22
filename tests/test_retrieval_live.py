"""The real rerankers. Run with: uv run pytest -m integration

The first run downloads the models (MiniLM about 90 MB, bge-reranker-v2-m3 about 2.3 GB).
"""

import pytest

from hybrid_rag.config import RerankerSettings
from hybrid_rag.retrieval import get_reranker

pytestmark = pytest.mark.integration

QUESTION = "Can I fly premium economy on a long flight?"
CHUNKS = {
    "answer": "Policy: Travel Policy\nSection: 5 Air travel\n\n| Flight length | Cabin class |\n"
    "| 6 hours or less | Economy |\n| Over 6 hours | Premium economy |",
    "same words, wrong topic": "Policy: Employee Benefits Guide\nSection: 2 Medical Plans\n\n"
    "Economy of choice: premium plans have a lower deductible.",
    "unrelated": "Policy: HR Policy Handbook\nSection: 16 Information Security\n\n"
    "Use multi-factor authentication and a password manager.",
}


@pytest.mark.parametrize(
    "settings",
    [
        RerankerSettings(),  # the default, ms-marco-MiniLM-L-6-v2
        RerankerSettings(model="BAAI/bge-reranker-v2-m3", max_length=1024),
    ],
    ids=["minilm-default", "bge-reranker-v2-m3"],
)
def test_reranker_prefers_the_chunk_that_answers(settings: RerankerSettings) -> None:
    reranker = get_reranker(settings)

    scores = dict(
        zip(CHUNKS, reranker.score([(QUESTION, t) for t in CHUNKS.values()]), strict=True)
    )

    assert scores["answer"] > max(scores["same words, wrong topic"], scores["unrelated"])
    assert get_reranker(settings) is reranker  # loaded once per process
