from hybrid_rag.llm.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState
from tests.llm_fakes import FakeClock


def make(threshold: int = 3, cooldown: float = 30.0) -> tuple[CircuitBreaker, FakeClock]:
    clock = FakeClock()
    return CircuitBreaker("p/m", threshold, cooldown, clock=clock), clock


def test_opens_after_consecutive_failures() -> None:
    breaker, _ = make(threshold=3)
    for _ in range(2):
        breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request()

    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_request()


def test_success_resets_failure_count() -> None:
    breaker, _ = make(threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


def test_half_open_after_cooldown_allows_one_probe() -> None:
    breaker, clock = make(threshold=1, cooldown=30)
    breaker.record_failure()
    clock.advance(29.9)
    assert not breaker.allow_request()

    clock.advance(0.1)
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow_request()  # the probe
    assert not breaker.allow_request()  # only one probe at a time


def test_probe_success_closes_circuit() -> None:
    breaker, clock = make(threshold=1)
    breaker.record_failure()
    clock.advance(30)
    assert breaker.allow_request()
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow_request()


def test_probe_failure_reopens_circuit() -> None:
    breaker, clock = make(threshold=5)
    for _ in range(5):
        breaker.record_failure()
    clock.advance(30)
    assert breaker.allow_request()
    breaker.record_failure()  # a single failure in half-open is enough
    assert breaker.state is CircuitState.OPEN
    assert not breaker.allow_request()


def test_abandoned_probe_expires() -> None:
    breaker, clock = make(threshold=1, cooldown=30)
    breaker.record_failure()
    clock.advance(30)
    assert breaker.allow_request()  # probe starts but never reports back
    assert not breaker.allow_request()

    clock.advance(30)
    assert breaker.allow_request()  # probe slot freed after another cooldown


def test_registry_shares_breakers_by_key() -> None:
    registry = CircuitBreakerRegistry(failure_threshold=1)
    assert registry.get("openai/gpt") is registry.get("openai/gpt")
    assert registry.get("openai/gpt") is not registry.get("anthropic/claude")

    registry.get("openai/gpt").record_failure()
    states = {s["name"]: s["state"] for s in registry.snapshot()}
    assert states == {"openai/gpt": "open", "anthropic/claude": "closed"}
