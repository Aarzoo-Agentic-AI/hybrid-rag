import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from hybrid_rag.llm import (
    FAILOVER_METADATA_KEY,
    AllProvidersFailedError,
    CircuitState,
    StreamInterruptedError,
)
from tests.llm_fakes import (
    FakeClock,
    MidStreamFailure,
    Script,
    anthropic_overloaded,
    make_failover,
    openai_auth_error,
    openai_bad_request,
    openai_rate_limit,
    openai_server_error,
    openai_timeout,
)


def info(message) -> dict:
    return message.response_metadata[FAILOVER_METADATA_KEY]


# ----------------------------------------------------------------- invoke


def test_primary_answers() -> None:
    p0, p1 = Script(["hello"]), Script([])
    model, _, clock = make_failover(p0, p1)

    result = model.invoke("hi")

    assert result.content == "hello"
    assert info(result)["provider"] == "p0"
    assert info(result)["failed_over"] is False
    assert info(result)["attempts"] == 1
    assert p1.calls == 0
    assert clock.sleeps == []


def test_retries_temporary_error_on_same_provider() -> None:
    p0 = Script([openai_rate_limit(), "ok"])
    model, _, clock = make_failover(p0, Script([]))

    result = model.invoke("hi")

    assert result.content == "ok"
    assert info(result)["provider"] == "p0"
    assert info(result)["attempts"] == 2
    assert len(info(result)["errors"]) == 1
    assert clock.sleeps == [0.5]


def test_fails_over_after_retries_are_used_up() -> None:
    p0 = Script([openai_server_error(), openai_timeout(), anthropic_overloaded()])
    p1 = Script(["from fallback"])
    model, _, _ = make_failover(p0, p1, max_attempts=3)

    result = model.invoke("hi")

    assert result.content == "from fallback"
    assert info(result)["provider"] == "p1"
    assert info(result)["failed_over"] is True
    assert info(result)["attempts"] == 4
    assert p0.calls == 3


def test_backoff_doubles_and_is_capped() -> None:
    p0 = Script([openai_rate_limit()] * 5 + ["ok"])
    model, _, clock = make_failover(p0, max_attempts=6, base_delay_s=1.0, max_delay_s=5.0)

    model.invoke("hi")

    assert clock.sleeps == [1.0, 2.0, 4.0, 5.0, 5.0]


@pytest.mark.parametrize(
    "error", [openai_auth_error(), openai_bad_request(), ValueError("bug")], ids=repr
)
def test_fatal_errors_raise_immediately_without_failover(error: Exception) -> None:
    p0, p1 = Script([error]), Script(["should not be used"])
    model, _, clock = make_failover(p0, p1)

    with pytest.raises(type(error)):
        model.invoke("hi")

    assert p0.calls == 1
    assert p1.calls == 0
    assert clock.sleeps == []


def test_all_providers_failing_raises_with_every_attempt() -> None:
    p0 = Script([openai_server_error()] * 2)
    p1 = Script([anthropic_overloaded()] * 2)
    model, _, _ = make_failover(p0, p1, max_attempts=2)

    with pytest.raises(AllProvidersFailedError) as exc_info:
        model.invoke("hi")

    assert len(exc_info.value.failures) == 4
    assert [f.provider for f in exc_info.value.failures] == ["p0", "p0", "p1", "p1"]


def test_overall_deadline_stops_retrying() -> None:
    p0 = Script([openai_rate_limit()] * 10)
    model, _, clock = make_failover(
        p0, max_attempts=10, base_delay_s=4.0, max_delay_s=4.0, overall_deadline_s=10.0
    )

    with pytest.raises(AllProvidersFailedError):
        model.invoke("hi")

    assert sum(clock.sleeps) < 10.0
    assert p0.calls < 10


# --------------------------------------------------------- circuit breaker


def test_open_circuit_skips_provider_on_later_requests() -> None:
    p0 = Script([openai_server_error()] * 2)
    p1 = Script(["first", "second"])
    model, breakers, clock = make_failover(p0, p1, failure_threshold=2, max_attempts=3)

    first = model.invoke("hi")
    assert breakers[0].state is CircuitState.OPEN
    assert p0.calls == 2  # stopped retrying as soon as the circuit opened
    assert info(first)["provider"] == "p1"

    second = model.invoke("hi again")
    assert p0.calls == 2  # not called at all
    assert info(second)["skipped"] == ["p0/m"]
    assert clock.sleeps == [0.5]


def test_provider_comes_back_after_cooldown() -> None:
    p0 = Script([openai_server_error(), "recovered"])
    p1 = Script(["fallback"])
    model, breakers, clock = make_failover(p0, p1, failure_threshold=1, cooldown_s=30)

    assert info(model.invoke("hi"))["provider"] == "p1"
    clock.advance(30)

    result = model.invoke("hi")
    assert info(result)["provider"] == "p0"
    assert breakers[0].state is CircuitState.CLOSED


def test_all_circuits_open_raises() -> None:
    model, breakers, _ = make_failover(Script([]), Script([]), failure_threshold=1)
    for b in breakers:
        b.record_failure()

    with pytest.raises(AllProvidersFailedError) as exc_info:
        model.invoke("hi")

    assert exc_info.value.skipped == ["p0/m", "p1/m"]
    assert "circuit" in exc_info.value.reason


# ---------------------------------------------------------------- stream


def collect(model, prompt: str = "hi") -> AIMessageChunk:
    chunks = list(model.stream(prompt))
    full = chunks[0]
    for c in chunks[1:]:
        full += c
    return full


def test_stream_success_includes_failover_metadata() -> None:
    model, _, _ = make_failover(Script(["streamed answer here"]))

    full = collect(model)

    assert full.content == "streamed answer here "
    assert info(full)["provider"] == "p0"


def test_stream_fails_over_before_first_token() -> None:
    p0 = Script([openai_server_error()])
    p1 = Script(["fallback stream"])
    model, _, _ = make_failover(p0, p1, max_attempts=1)

    full = collect(model)

    assert full.content == "fallback stream "
    assert info(full)["failed_over"] is True


def test_stream_failure_mid_answer_does_not_fail_over() -> None:
    p0 = Script([MidStreamFailure(["partial ", "answer "], openai_server_error())])
    p1 = Script(["must not be mixed in"])
    model, _, _ = make_failover(p0, p1)

    received: list[str] = []
    with pytest.raises(StreamInterruptedError) as exc_info:
        for chunk in model.stream("hi"):
            received.append(chunk.content)

    assert received == ["partial ", "answer "]
    assert exc_info.value.provider == "p0"
    assert p1.calls == 0


# ----------------------------------------------------------------- async


async def test_async_invoke_fails_over() -> None:
    p0 = Script([openai_rate_limit()])
    p1 = Script(["async fallback"])
    model, _, _ = make_failover(p0, p1, max_attempts=1)

    result = await model.ainvoke("hi")

    assert result.content == "async fallback"
    assert info(result)["provider"] == "p1"


async def test_async_retry_uses_async_sleep() -> None:
    p0 = Script([openai_rate_limit(), "ok"])
    model, _, clock = make_failover(p0)

    result = await model.ainvoke("hi")

    assert result.content == "ok"
    assert clock.sleeps == [0.5]


async def test_async_stream_fails_over_before_first_token() -> None:
    p0 = Script([openai_timeout()])
    p1 = Script(["async stream"])
    model, _, _ = make_failover(p0, p1, max_attempts=1)

    chunks = [c async for c in model.astream("hi")]
    full = chunks[0]
    for c in chunks[1:]:
        full += c  # metadata travels in a chunk and merges into the full message

    assert full.content == "async stream "
    assert info(full)["provider"] == "p1"


async def test_async_stream_failure_mid_answer_raises() -> None:
    p0 = Script([MidStreamFailure(["partial "], openai_server_error())])
    model, _, _ = make_failover(p0, Script(["unused"]))

    with pytest.raises(StreamInterruptedError):
        async for _ in model.astream("hi"):
            pass


# ------------------------------------------------------------------ LCEL


def test_works_inside_an_lcel_chain() -> None:
    p0 = Script([openai_server_error()])
    p1 = Script(["chain answer"])
    model, _, _ = make_failover(p0, p1, max_attempts=1)
    chain = (
        ChatPromptTemplate.from_messages([("human", "Q: {question}")]) | model | StrOutputParser()
    )

    assert chain.invoke({"question": "what?"}) == "chain answer"
    assert p1.seen[0][0].content == "Q: what?"


def test_streaming_lcel_chain_yields_text() -> None:
    model, _, _ = make_failover(Script(["a b c"]))
    chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | model | StrOutputParser()

    assert "".join(chain.stream({"q": "x"})) == "a b c "


def test_fake_clock_default_is_independent() -> None:
    # Guard against shared state between tests built with make_failover.
    assert FakeClock().sleeps == []


# --------------------------------------------------------------- tracing


def _attempt_names(run) -> list[str]:
    return [child.name for child in run.child_runs]


@pytest.mark.parametrize("mode", ["invoke", "stream", "ainvoke", "astream"])
async def test_provider_attempts_are_nested_under_the_failover_run(mode: str) -> None:
    from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

    model, _, _ = make_failover(Script([openai_rate_limit(), "a b"]))
    collector = RunCollectorCallbackHandler()
    config = {"callbacks": [collector]}

    if mode == "invoke":
        model.invoke("hi", config=config)
    elif mode == "stream":
        list(model.stream("hi", config=config))
    elif mode == "ainvoke":
        await model.ainvoke("hi", config=config)
    else:
        _ = [c async for c in model.astream("hi", config=config)]

    assert [r.name for r in collector.traced_runs] == ["FailoverChatModel"]
    assert _attempt_names(collector.traced_runs[0]) == ["p0/m (attempt 1)", "p0/m (attempt 2)"]


def test_streamed_chain_nests_failover_and_attempts() -> None:
    from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

    model, _, _ = make_failover(Script([openai_server_error()]), Script(["x y"]), max_attempts=1)
    chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | model | StrOutputParser()
    collector = RunCollectorCallbackHandler()

    list(chain.stream({"q": "hi"}, config={"callbacks": [collector]}))

    (root,) = collector.traced_runs
    failover = next(c for c in root.child_runs if c.name == "FailoverChatModel")
    assert _attempt_names(failover) == ["p0/m (attempt 1)", "p1/m (attempt 1)"]


def test_stream_parent_does_not_leak_after_streaming() -> None:
    from hybrid_rag.llm.failover import _STREAM_PARENT

    model, _, _ = make_failover(Script(["a b c"]))
    for _ in model.stream("hi"):
        assert _STREAM_PARENT.get() is None  # only set while a chunk is being produced
    assert _STREAM_PARENT.get() is None
