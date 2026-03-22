from __future__ import annotations

import pytest
from structlog.testing import capture_logs

import tardigrade._circuit_breaker as circuit_breaker_module
from tardigrade import CircuitBreakerConfig, CircuitState
from tardigrade._circuit_breaker import CircuitBreaker


def test_circuit_breaker_starts_closed() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig())

    assert breaker.state is CircuitState.CLOSED


def test_circuit_breaker_stays_closed_below_threshold() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))

    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED


def test_circuit_breaker_trips_open_at_threshold() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
    breaker.bind("step")

    with capture_logs() as captured_logs:
        breaker.record_failure()
        breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert captured_logs[-1]["event"] == "circuit_opened"
    assert captured_logs[-1]["failure_count"] == 2


def test_circuit_breaker_success_resets_failure_count_in_closed() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=5))

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 3


def test_circuit_breaker_open_blocks_execution() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))

    breaker.record_failure()

    assert breaker.can_execute() is False


def test_circuit_breaker_open_transitions_to_half_open_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [0.0]
    monkeypatch.setattr(circuit_breaker_module.time, "monotonic", lambda: current[0])
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1, recovery_timeout=10.0))

    breaker.record_failure()
    current[0] = 10.1

    assert breaker.can_execute() is True
    assert breaker.state is CircuitState.HALF_OPEN


def test_circuit_breaker_half_open_limits_probe_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [0.0]
    monkeypatch.setattr(circuit_breaker_module.time, "monotonic", lambda: current[0])
    breaker = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=10.0,
            half_open_max_calls=1,
        )
    )

    breaker.record_failure()
    current[0] = 10.1

    assert breaker.can_execute() is True
    assert breaker.can_execute() is False


def test_circuit_breaker_half_open_success_closes_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [0.0]
    monkeypatch.setattr(circuit_breaker_module.time, "monotonic", lambda: current[0])
    breaker = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=10.0,
            success_threshold=2,
            half_open_max_calls=1,
        )
    )
    breaker.bind("step")

    breaker.record_failure()
    current[0] = 10.1
    assert breaker.can_execute() is True
    breaker.record_success()
    assert breaker.state is CircuitState.HALF_OPEN

    current[0] = 10.2
    assert breaker.can_execute() is True
    with capture_logs() as captured_logs:
        breaker.record_success()

    assert breaker.state is CircuitState.CLOSED
    assert captured_logs[-1]["event"] == "circuit_closed"
    assert captured_logs[-1]["success_count"] == 2


def test_circuit_breaker_half_open_failure_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [0.0]
    monkeypatch.setattr(circuit_breaker_module.time, "monotonic", lambda: current[0])
    breaker = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=10.0,
        )
    )
    breaker.bind("step")

    breaker.record_failure()
    current[0] = 10.1
    assert breaker.can_execute() is True

    current[0] = 10.2
    with capture_logs() as captured_logs:
        breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert captured_logs[-1]["event"] == "circuit_reopened"


def test_circuit_breaker_reset_returns_to_closed() -> None:
    breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))

    breaker.record_failure()
    breaker.reset()

    assert breaker.state is CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_config_holds_monitored_exceptions() -> None:
    config = CircuitBreakerConfig(monitored_exceptions=(ValueError,))
    breaker = CircuitBreaker(config)

    assert breaker.config.monitored_exceptions == (ValueError,)

