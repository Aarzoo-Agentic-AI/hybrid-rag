"""End to end on the real index with the real LLM (OpenAI). Run with: uv run pytest -m integration

Needs Qdrant running with the HR policies indexed (scripts/ingest.py) and OPENAI_API_KEY in
.env. Each test makes one LLM call (a few cents in total). Tracing stays off (conftest.py).
"""

import pytest

from hybrid_rag.access import Principal
from hybrid_rag.config import get_settings
from hybrid_rag.generation import AnswerService
from hybrid_rag.store import build_document_index

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def service() -> AnswerService:
    settings = get_settings()
    if settings.openai_api_key is None:
        pytest.skip("OPENAI_API_KEY is not set")
    return AnswerService.from_settings(settings, build_document_index(settings))


@pytest.fixture(scope="module")
def tenant() -> str:
    return get_settings().documents.tenant_id


def test_pto_answer_is_grounded_in_the_benefits_guide(service, tenant) -> None:
    answer = service.answer(
        "How many PTO days do I get after 4 years?", Principal(tenant, "u-1", "internal")
    )

    assert answer.status == "answered" and answer.grounded
    assert "20" in answer.text
    assert any(c.source == "Employee_Benefits_Guide_2027.pdf" for c in answer.citations)


def test_out_of_scope_question_is_not_answered(service, tenant) -> None:
    answer = service.answer("What is the capital of France?", Principal(tenant, "u-1", "internal"))

    assert answer.status in {"low_confidence", "not_found"} and not answer.citations


def test_restricted_policy_is_never_cited_for_an_employee(service, tenant) -> None:
    question = "What merit increase does a Meets rating get at a 0.95 compa-ratio?"

    employee = service.answer(question, Principal(tenant, "u-1", "internal"))
    manager = service.answer(
        question, Principal(tenant, "u-2", "confidential", roles={"people_manager"})
    )

    assert all("Merit" not in c.source for c in employee.citations)
    assert "2.5" in manager.text and "3.5" in manager.text  # the merit matrix row
    assert any(c.source.startswith("Annual_Review_Merit") for c in manager.citations)
