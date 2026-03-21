from __future__ import annotations

from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from agentarmor import RetryConfig, armor
from agentarmor._context import get_current_armor_context


def test_sync_retry_success_on_first_try_has_no_retry_events() -> None:
    calls = 0

    @armor(retry=RetryConfig(max_attempts=3, jitter=False))
    def step() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    with capture_logs() as captured_logs:
        assert step() == "ok"

    assert calls == 1
    assert [entry["event"] for entry in captured_logs] == ["step_started", "step_completed"]
    assert captured_logs[1]["attempt"] == 1


def test_sync_retry_fails_then_succeeds() -> None:
    calls = 0
    attempts_seen: list[int] = []
    max_attempts_seen: list[int] = []

    @armor(retry=RetryConfig(max_attempts=4, base_delay=1.0, jitter=False))
    def step() -> str:
        nonlocal calls
        calls += 1
        context = get_current_armor_context()
        assert context is not None
        attempts_seen.append(context.attempt)
        max_attempts_seen.append(context.max_attempts)
        if calls < 3:
            raise ValueError("transient")
        return "ok"

    with patch("agentarmor._decorator.time.sleep") as sleep_mock:
        with capture_logs() as captured_logs:
            assert step() == "ok"

    retry_events = [entry for entry in captured_logs if entry["event"] == "step_retrying"]
    completed_events = [entry for entry in captured_logs if entry["event"] == "step_completed"]

    assert calls == 3
    assert attempts_seen == [1, 2, 3]
    assert max_attempts_seen == [4, 4, 4]
    assert len(retry_events) == 2
    assert completed_events[0]["attempt"] == 3
    assert sleep_mock.call_args_list == [((1.0,), {}), ((2.0,), {})]


def test_sync_retry_exhausts_all_attempts_and_reraises_original_exception() -> None:
    calls = 0

    @armor(retry=RetryConfig(max_attempts=3, base_delay=1.0, jitter=False))
    def step() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("still broken")

    with patch("agentarmor._decorator.time.sleep") as sleep_mock:
        with capture_logs() as captured_logs:
            with pytest.raises(ValueError, match="still broken"):
                step()

    exhausted_events = [
        entry for entry in captured_logs if entry["event"] == "step_failed_all_retries"
    ]

    assert calls == 3
    assert len(exhausted_events) == 1
    assert exhausted_events[0]["total_attempts"] == 3
    assert exhausted_events[0]["last_exception_type"] == "ValueError"
    assert exhausted_events[0]["last_exception_message"] == "still broken"
    assert sleep_mock.call_args_list == [((1.0,), {}), ((2.0,), {})]


def test_sync_retry_non_retryable_exception_skips_retries() -> None:
    calls = 0

    @armor(
        retry=RetryConfig(
            max_attempts=3,
            jitter=False,
            retryable_exceptions=(ValueError,),
        )
    )
    def step() -> None:
        nonlocal calls
        calls += 1
        raise TypeError("not retryable")

    with patch("agentarmor._decorator.time.sleep") as sleep_mock:
        with capture_logs() as captured_logs:
            with pytest.raises(TypeError, match="not retryable"):
                step()

    assert calls == 1
    assert sleep_mock.call_count == 0
    assert "step_retrying" not in [entry["event"] for entry in captured_logs]
    assert captured_logs[-1]["event"] == "step_failed"


def test_sync_retry_backoff_delay_increases_exponentially() -> None:
    @armor(
        retry=RetryConfig(
            max_attempts=4,
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )
    )
    def step() -> None:
        raise ValueError("boom")

    with patch("agentarmor._decorator.time.sleep") as sleep_mock:
        with pytest.raises(ValueError, match="boom"):
            step()

    assert sleep_mock.call_args_list == [((1.0,), {}), ((2.0,), {}), ((4.0,), {})]


def test_sync_retry_respects_max_delay() -> None:
    @armor(
        retry=RetryConfig(
            max_attempts=3,
            base_delay=10.0,
            max_delay=15.0,
            exponential_base=2.0,
            jitter=False,
        )
    )
    def step() -> None:
        raise ValueError("boom")

    with patch("agentarmor._decorator.time.sleep") as sleep_mock:
        with pytest.raises(ValueError, match="boom"):
            step()

    assert sleep_mock.call_args_list == [((10.0,), {}), ((15.0,), {})]


def test_sync_retry_jitter_changes_delay() -> None:
    @armor(retry=RetryConfig(max_attempts=2, base_delay=2.0, jitter=True))
    def step() -> None:
        raise ValueError("boom")

    with patch("agentarmor._types.random.uniform", return_value=1.25):
        with patch("agentarmor._decorator.time.sleep") as sleep_mock:
            with pytest.raises(ValueError, match="boom"):
                step()

    assert sleep_mock.call_args_list == [((2.5,), {})]


def test_sync_retry_true_uses_default_retry_config() -> None:
    calls = 0

    @armor(retry=True)
    def step() -> None:
        nonlocal calls
        calls += 1
        raise TimeoutError("rate limited")

    with patch("agentarmor._types.random.uniform", return_value=1.0):
        with patch("agentarmor._decorator.time.sleep") as sleep_mock:
            with pytest.raises(TimeoutError, match="rate limited"):
                step()

    assert calls == 3
    assert sleep_mock.call_args_list == [((1.0,), {}), ((2.0,), {})]


def test_sync_retry_absent_means_single_attempt_only() -> None:
    calls = 0

    @armor
    def step() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("boom")

    with capture_logs() as captured_logs:
        with pytest.raises(ValueError, match="boom"):
            step()

    assert calls == 1
    assert [entry["event"] for entry in captured_logs] == ["step_started", "step_failed"]
