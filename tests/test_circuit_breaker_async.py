from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

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


@pytest.mark.asyncio
async def test_async_circuit_breaker_uses_awaited_fallback_when_open() -> None:
    fallback_calls = 0

    async def fallback(prompt: str) -> str:
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
    async def step(prompt: str) -> str:
        raise ValueError("primary failed")

    with pytest.raises(ValueError, match="primary failed"):
        await step("hello")
    with pytest.raises(ValueError, match="primary failed"):
        await step("hello")

    with capture_logs() as captured_logs:
        result = await step("hello")

    breaker = step._circuit_breaker
    assert result == "fallback:hello"
    assert fallback_calls == 1
    assert breaker.state is CircuitState.OPEN
    assert type(breaker._lock).__module__ == "_thread"
    fallback_events = [entry for entry in captured_logs if entry["event"] == "circuit_fallback"]
    assert len(fallback_events) == 1


@pytest.mark.asyncio
async def test_async_circuit_breaker_fallback_receives_original_arguments() -> None:
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def fallback(*args: Any, **kwargs: Any) -> str:
        seen.append((args, kwargs))
        return "fallback"

    @armor(
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=1,
            fallback=fallback,
            monitored_exceptions=(ValueError,),
        )
    )
    async def step(x: int, *, label: str) -> str:
        raise ValueError("primary failed")

    with pytest.raises(ValueError, match="primary failed"):
        await step(5, label="demo")

    assert await step(5, label="demo") == "fallback"
    assert seen == [((5,), {"label": "demo"})]


@pytest.mark.asyncio
async def test_async_circuit_breaker_recovers_after_timeout(
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
    async def step() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("primary failed")
        return "recovered"

    with pytest.raises(ValueError, match="primary failed"):
        await step()

    current[0] = 30.1
    result = await step()

    breaker = step._circuit_breaker
    assert result == "recovered"
    assert breaker.state is CircuitState.CLOSED
    assert calls == 2


@pytest.mark.asyncio
async def test_async_circuit_breaker_counts_one_failure_per_retried_invocation() -> None:
    calls = 0

    @armor(
        retry=RetryConfig(max_attempts=3, base_delay=1.0, jitter=False),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            monitored_exceptions=(ValueError,),
        ),
    )
    async def step() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("primary failed")

    breaker = step._circuit_breaker

    with patch("agentarmor._decorator.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ValueError, match="primary failed"):
            await step()
        assert breaker.failure_count == 1

        with pytest.raises(ValueError, match="primary failed"):
            await step()
        assert breaker.failure_count == 2

        with pytest.raises(ValueError, match="primary failed"):
            await step()

    assert breaker.state is CircuitState.OPEN
    assert calls == 9


@pytest.mark.asyncio
async def test_async_fallback_result_is_checkpointed(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = 0
    fallback_calls = 0

    async def fallback() -> str:
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
    async def step() -> str:
        nonlocal calls
        calls += 1
        return "primary-result"

    breaker = step._circuit_breaker

    try:
        breaker.record_failure()

        async with Workflow("pipeline", run_id="run-1", store=store):
            assert await step() == "fallback-result"

        with capture_logs() as captured_logs:
            async with Workflow("pipeline", run_id="run-1", store=store):
                assert await step() == "fallback-result"

        assert calls == 0
        assert fallback_calls == 1
        assert [entry["event"] for entry in captured_logs] == ["step_restored_from_checkpoint"]
    finally:
        await store.aclose()
