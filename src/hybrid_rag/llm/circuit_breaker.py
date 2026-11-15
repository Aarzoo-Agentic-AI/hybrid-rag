"""A small, thread-safe circuit breaker for each LLM provider.

States:
- CLOSED: requests flow normally. Consecutive failures are counted.
- OPEN: after ``failure_threshold`` consecutive failures, requests are refused until
  ``cooldown_s`` has passed.
- HALF_OPEN: after the cooldown, a single probe request is let through. Success closes
  the circuit; failure opens it again for another cooldown. A probe that never reports back
  (for example, the request was abandoned) expires after another cooldown, so the provider
  can't get stuck.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import Enum


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._probe_started_at = 0.0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._current_state()

    def _current_state(self) -> CircuitState:
        # Caller must hold the lock.
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self.cooldown_s:
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def allow_request(self) -> bool:
        """Return True if a request may be sent now. In HALF_OPEN, only one probe is allowed."""
        with self._lock:
            state = self._current_state()
            if state is CircuitState.CLOSED:
                return True
            if state is CircuitState.HALF_OPEN:
                probe_expired = self._clock() - self._probe_started_at >= self.cooldown_s
                if not self._probe_in_flight or probe_expired:
                    self._probe_in_flight = True
                    self._probe_started_at = self._clock()
                    return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            state = self._current_state()
            self._consecutive_failures += 1
            if (
                state is CircuitState.HALF_OPEN
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                self._probe_in_flight = False

    def release_probe(self) -> None:
        """Free the half-open probe slot without recording a result (e.g. on a fatal error)."""
        with self._lock:
            self._probe_in_flight = False

    def snapshot(self) -> dict[str, object]:
        """State for health checks and debugging."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._current_state().value,
                "consecutive_failures": self._consecutive_failures,
            }


class CircuitBreakerRegistry:
    """Hands out one shared breaker per key, so every task using a model sees the same state."""

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, key: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(key)
            if breaker is None:
                breaker = CircuitBreaker(
                    key, self.failure_threshold, self.cooldown_s, clock=self._clock
                )
                self._breakers[key] = breaker
            return breaker

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            breakers = list(self._breakers.values())
        return [b.snapshot() for b in breakers]
