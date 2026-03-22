from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from tardigrade import armor
from tardigrade._context import get_current_armor_context


@pytest.mark.asyncio
async def test_async_decorator_returns_value_and_logs_success() -> None:
    @armor
    async def multiply(x: int, y: int) -> int:
        context = get_current_armor_context()
        assert context is not None
        return x * y

    with capture_logs() as captured_logs:
        result = await multiply(3, 4)

    assert result == 12
    assert [entry["event"] for entry in captured_logs] == ["step_started", "step_completed"]
    assert captured_logs[1]["status"] == "success"
    assert captured_logs[1]["result"] == "12"
    assert captured_logs[1]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_async_decorator_propagates_errors_and_logs_failure() -> None:
    @armor
    async def explode() -> None:
        raise ValueError("boom")

    with capture_logs() as captured_logs:
        with pytest.raises(ValueError, match="boom"):
            await explode()

    assert [entry["event"] for entry in captured_logs] == ["step_started", "step_failed"]
    assert captured_logs[1]["status"] == "error"
    assert captured_logs[1]["exception_type"] == "ValueError"
    assert captured_logs[1]["exception_message"] == "boom"


@pytest.mark.asyncio
async def test_async_decorator_supports_optional_parentheses_and_custom_names() -> None:
    @armor
    async def default_name() -> str:
        return "default"

    @armor(name="custom_async_step")
    async def overridden_name() -> str:
        return "custom"

    with capture_logs() as captured_logs:
        assert await default_name() == "default"
        assert await overridden_name() == "custom"

    assert captured_logs[1]["function_name"] == (
        "test_async_decorator_supports_optional_parentheses_and_custom_names.<locals>.default_name"
    )
    assert captured_logs[3]["function_name"] == "custom_async_step"
