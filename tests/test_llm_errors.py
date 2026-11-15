import httpx
import openai
import pytest

from hybrid_rag.llm.errors import ErrorKind, classify_error
from tests.llm_fakes import (
    anthropic_overloaded,
    openai_auth_error,
    openai_bad_request,
    openai_rate_limit,
    openai_server_error,
    openai_timeout,
    status_error,
)


@pytest.mark.parametrize(
    "exc",
    [
        openai_rate_limit(),
        openai_server_error(),
        openai_timeout(),
        anthropic_overloaded(),
        status_error(openai.APIStatusError, 503),
        status_error(openai.APIStatusError, 408),
        openai.APIConnectionError(request=httpx.Request("GET", "https://x.invalid")),
        httpx.ConnectTimeout("timed out"),
        TimeoutError(),
        ConnectionResetError(),
    ],
    ids=lambda e: type(e).__name__,
)
def test_temporary_errors_are_retryable(exc: Exception) -> None:
    assert classify_error(exc) is ErrorKind.RETRYABLE


@pytest.mark.parametrize(
    "exc",
    [
        openai_auth_error(),
        openai_bad_request(),
        status_error(openai.NotFoundError, 404),
        status_error(openai.PermissionDeniedError, 403),
        ValueError("bug in our code"),
        KeyError("missing"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_caller_errors_and_bugs_are_fatal(exc: Exception) -> None:
    assert classify_error(exc) is ErrorKind.FATAL
