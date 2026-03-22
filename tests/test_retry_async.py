from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from structlog.testing import capture_logs

from tardigrade import RetryConfig, armor
from tardigrade._context import get_current_armor_context


@pytest.mark.asyncio
async def test_async_retry_fails_then_succeeds() -> None:
    calls = 0
    attempts_seen: list[int] = []

    @armor(retry=RetryConfig(max_attempts=4, base_delay=1.0, jitter=False))
    async def step() -> str:
        nonlocal calls
        calls += 1
        context = get_current_armor_context()
        assert context is not None
        attempts_seen.append(context.attempt)
        if calls < 3:
            raise ValueError("transient")
        return "ok"

    with patch("tardigrade._decorator.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with capture_logs() as captured_logs:
            assert await step() == "ok"

    retry_events = [entry for entry in captured_logs if entry["event"] == "step_retrying"]
    completed_events = [entry for entry in captured_logs if entry["event"] == "step_completed"]

    assert calls == 3
    assert attempts_seen == [1, 2, 3]
    assert len(retry_events) == 2
    assert completed_events[0]["attempt"] == 3
    assert sleep_mock.await_args_list == [((1.0,), {}), ((2.0,), {})]


@pytest.mark.asyncio
async def test_async_retry_exhausts_all_attempts_and_reraises_original_exception() -> None:
    calls = 0

    @armor(retry=RetryConfig(max_attempts=3, base_delay=1.0, jitter=False))
    async def step() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("still broken")

    with patch("tardigrade._decorator.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with capture_logs() as captured_logs:
            with pytest.raises(ValueError, match="still broken"):
                await step()

    exhausted_events = [
        entry for entry in captured_logs if entry["event"] == "step_failed_all_retries"
    ]

    assert calls == 3
    assert len(exhausted_events) == 1
    assert exhausted_events[0]["total_attempts"] == 3
    assert exhausted_events[0]["last_exception_type"] == "ValueError"
    assert sleep_mock.await_args_list == [((1.0,), {}), ((2.0,), {})]


@pytest.mark.asyncio
async def test_async_retry_non_retryable_exception_skips_retries() -> None:
    calls = 0

    @armor(
        retry=RetryConfig(
            max_attempts=3,
            jitter=False,
            retryable_exceptions=(ValueError,),
        )
    )
    async def step() -> None:
        nonlocal calls
        calls += 1
        raise TypeError("not retryable")

    with patch("tardigrade._decorator.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with capture_logs() as captured_logs:
            with pytest.raises(TypeError, match="not retryable"):
                await step()

    assert calls == 1
    assert sleep_mock.await_count == 0
    assert "step_retrying" not in [entry["event"] for entry in captured_logs]
    assert captured_logs[-1]["event"] == "step_failed"


@pytest.mark.asyncio
async def test_async_retry_uses_asyncio_sleep_not_time_sleep() -> None:
    @armor(retry=RetryConfig(max_attempts=2, base_delay=0.25, jitter=False))
    async def step() -> None:
        raise ValueError("boom")

    with patch("tardigrade._decorator.asyncio.sleep", new=AsyncMock()) as async_sleep_mock:
        with patch("tardigrade._decorator.time.sleep") as time_sleep_mock:
            with pytest.raises(ValueError, match="boom"):
                await step()

    assert async_sleep_mock.await_args_list == [((0.25,), {})]
    assert time_sleep_mock.call_count == 0
