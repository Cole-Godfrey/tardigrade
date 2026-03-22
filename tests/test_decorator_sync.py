from __future__ import annotations

import inspect

import pytest
from structlog.testing import capture_logs

from tardigrade import armor
from tardigrade._context import get_current_armor_context


def test_sync_decorator_returns_value_and_logs_success() -> None:
    @armor
    def multiply(x: int, y: int) -> int:
        return x * y

    with capture_logs() as captured_logs:
        result = multiply(3, 4)

    assert result == 12
    assert [entry["event"] for entry in captured_logs] == ["step_started", "step_completed"]
    assert captured_logs[0]["status"] == "started"
    assert captured_logs[1]["status"] == "success"
    assert captured_logs[0]["function_name"] == (
        "test_sync_decorator_returns_value_and_logs_success.<locals>.multiply"
    )
    assert captured_logs[0]["call_id"] == captured_logs[1]["call_id"]
    assert captured_logs[1]["result"] == "12"
    assert captured_logs[1]["duration_ms"] >= 0


def test_sync_decorator_propagates_errors_and_logs_failure() -> None:
    @armor
    def explode() -> None:
        raise ValueError("boom")

    with capture_logs() as captured_logs:
        with pytest.raises(ValueError, match="boom"):
            explode()

    assert [entry["event"] for entry in captured_logs] == ["step_started", "step_failed"]
    assert captured_logs[1]["status"] == "error"
    assert captured_logs[1]["exception_type"] == "ValueError"
    assert captured_logs[1]["exception_message"] == "boom"


def test_sync_decorator_supports_optional_parentheses_and_custom_names() -> None:
    @armor
    def default_name() -> str:
        return "default"

    @armor(name="custom_step")
    def overridden_name() -> str:
        return "custom"

    with capture_logs() as captured_logs:
        assert default_name() == "default"
        assert overridden_name() == "custom"

    assert captured_logs[1]["function_name"] == (
        "test_sync_decorator_supports_optional_parentheses_and_custom_names.<locals>.default_name"
    )
    assert captured_logs[3]["function_name"] == "custom_step"


def test_sync_decorator_preserves_metadata_and_signature() -> None:
    def original(value: int, *, scale: int = 2) -> int:
        """Multiply a value by a scale."""

        return value * scale

    decorated = armor(original)

    assert decorated.__name__ == original.__name__
    assert decorated.__doc__ == original.__doc__
    assert inspect.signature(decorated) == inspect.signature(original)


def test_sync_decorator_sets_context_for_instance_methods() -> None:
    class Worker:
        @armor
        def run(self, value: int) -> str:
            context = get_current_armor_context()
            assert context is not None
            assert context.function_name.endswith("Worker.run")
            return f"value={value}"

    worker = Worker()

    assert worker.run(5) == "value=5"
