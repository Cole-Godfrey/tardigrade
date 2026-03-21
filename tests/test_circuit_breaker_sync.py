from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

import agentarmor._circuit_breaker as circuit_breaker_module
from agentarmor import (
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
    SQLiteCheckpointStore,
    Workflow,
    armor,
)
from agentarmor._types import AgentArmorCircuitOpenError


def test_sync_circuit_breaker_uses_fallback_when_open() -> None:
    fallback_calls = 0

    def fallback(prompt: str) -> str:
        nonlocal fallback_calls
        fallback_calls += 1
        return f"fallback:{prompt}"

    @armor(
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=2,
            fallback=fallback,
            monitored_exceptions=(ValueError,),
        )
    )
    def step(prompt: str) -> str:
        raise ValueError("primary failed")

    with pytest.raises(ValueError, match="primary failed"):
        step("hello")
    with pytest.raises(ValueError, match="primary failed"):
        step("hello")

    with capture_logs() as captured_logs:
        result = step("hello")

    assert result == "fallback:hello"
    assert fallback_calls == 1
    assert step._circuit_breaker.state is CircuitState.OPEN
    fallback_events = [entry for entry in captured_logs if entry["event"] == "circuit_fallback"]
    assert len(fallback_events) == 1
    assert fallback_events[0]["fallback_name"] == fallback.__qualname__


def test_sync_circuit_breaker_fallback_receives_original_arguments() -> None:
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fallback(*args: Any, **kwargs: Any) -> str:
        seen.append((args, kwargs))
        return "fallback"

    @armor(
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=1,
            fallback=fallback,
            monitored_exceptions=(ValueError,),
        )
    )
    def step(x: int, *, label: str) -> str:
        raise ValueError("primary failed")

    with pytest.raises(ValueError, match="primary failed"):
        step(3, label="demo")

    assert step(3, label="demo") == "fallback"
    assert seen == [((3,), {"label": "demo"})]


def test_sync_circuit_breaker_without_fallback_raises_open_error() -> None:
    @armor(
        name="primary_step",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=1,
            monitored_exceptions=(ValueError,),
        ),
    )
    def step() -> str:
        raise ValueError("primary failed")

    with pytest.raises(ValueError, match="primary failed"):
        step()

    with pytest.raises(AgentArmorCircuitOpenError) as exc_info:
        step()

    assert exc_info.value.function_name == "primary_step"
    assert exc_info.value.state is CircuitState.OPEN


def test_sync_circuit_breaker_recovers_after_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [0.0]
    monkeypatch.setattr(circuit_breaker_module.time, "monotonic", lambda: current[0])
    calls = 0

    @armor(
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=30.0,
            success_threshold=1,
            monitored_exceptions=(ValueError,),
        )
    )
    def step() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("primary failed")
        return "recovered"

    with pytest.raises(ValueError, match="primary failed"):
        step()

    current[0] = 30.1
    result = step()

    breaker = step._circuit_breaker
    assert result == "recovered"
    assert breaker.state is CircuitState.CLOSED
    assert calls == 2


def test_sync_circuit_breaker_counts_one_failure_per_retried_invocation() -> None:
    calls = 0

    @armor(
        retry=RetryConfig(max_attempts=3, base_delay=1.0, jitter=False),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            monitored_exceptions=(ValueError,),
        ),
    )
    def step() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("primary failed")

    breaker = step._circuit_breaker

    with patch("agentarmor._decorator.time.sleep"):
        with pytest.raises(ValueError, match="primary failed"):
            step()
        assert breaker.failure_count == 1

        with pytest.raises(ValueError, match="primary failed"):
            step()
        assert breaker.failure_count == 2

        with pytest.raises(ValueError, match="primary failed"):
            step()

    assert breaker.state is CircuitState.OPEN
    assert calls == 9


def test_sync_checkpoint_restore_bypasses_circuit_breaker(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = 0

    @armor(
        name="step",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=1,
            monitored_exceptions=(ValueError,),
        ),
    )
    def step() -> int:
        nonlocal calls
        calls += 1
        return 7

    breaker = step._circuit_breaker

    try:
        with Workflow("pipeline", run_id="run-1", store=store):
            assert step() == 7

        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN

        with capture_logs() as captured_logs:
            with Workflow("pipeline", run_id="run-1", store=store):
                assert step() == 7

        assert calls == 1
        assert "circuit_fallback" not in [entry["event"] for entry in captured_logs]
    finally:
        store.close()


def test_sync_fallback_result_is_checkpointed(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = 0
    fallback_calls = 0

    def fallback() -> str:
        nonlocal fallback_calls
        fallback_calls += 1
        return "fallback-result"

    @armor(
        name="step",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=1,
            fallback=fallback,
            monitored_exceptions=(ValueError,),
        ),
    )
    def step() -> str:
        nonlocal calls
        calls += 1
        return "primary-result"

    breaker = step._circuit_breaker

    try:
        breaker.record_failure()

        with Workflow("pipeline", run_id="run-1", store=store):
            assert step() == "fallback-result"

        with capture_logs() as captured_logs:
            with Workflow("pipeline", run_id="run-1", store=store):
                assert step() == "fallback-result"

        assert calls == 0
        assert fallback_calls == 1
        assert [entry["event"] for entry in captured_logs] == ["step_restored_from_checkpoint"]
    finally:
        store.close()


def test_sync_circuit_breaker_state_persists_across_calls() -> None:
    @armor(
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            monitored_exceptions=(ValueError,),
        )
    )
    def step() -> str:
        raise ValueError("primary failed")

    breaker = step._circuit_breaker

    with pytest.raises(ValueError, match="primary failed"):
        step()
    with pytest.raises(ValueError, match="primary failed"):
        step()

    assert step._circuit_breaker is breaker
    assert breaker.failure_count == 2


def test_sync_circuit_breaker_respects_monitored_exceptions() -> None:
    @armor(
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=1,
            monitored_exceptions=(ValueError,),
        )
    )
    def step() -> str:
        raise TypeError("not monitored")

    breaker = step._circuit_breaker

    with pytest.raises(TypeError, match="not monitored"):
        step()

    assert breaker.failure_count == 0
    assert breaker.state is CircuitState.CLOSED
