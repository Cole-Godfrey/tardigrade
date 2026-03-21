from __future__ import annotations

import importlib

import pytest

from agentarmor import armor, configure_logging
from agentarmor._event_bus import EventBus
from agentarmor.dashboard import Dashboard
from agentarmor.dashboard._widgets import (
    CircuitBreakerPanel,
    CostPanel,
    EventLogPanel,
    WorkflowPanel,
)


def test_workflow_panel_updates_internal_state_from_events() -> None:
    panel = WorkflowPanel()

    panel.update_from_event(
        {
            "event": "step_completed",
            "function_name": "my_step",
            "duration_ms": 150.0,
            "attempt": 1,
            "max_attempts": 3,
        }
    )

    assert panel.steps["my_step"].status == "completed"
    assert panel.steps["my_step"].duration_ms == pytest.approx(150.0)
    assert panel.steps["my_step"].attempt == 1


def test_circuit_breaker_panel_tracks_state_changes() -> None:
    panel = CircuitBreakerPanel()

    panel.update_from_event(
        {
            "event": "circuit_opened",
            "function_name": "llm_call",
            "failure_count": 5,
            "_timestamp": 1.0,
        }
    )
    panel.update_from_event(
        {
            "event": "circuit_closed",
            "function_name": "llm_call",
            "_timestamp": 2.0,
        }
    )

    assert panel.circuits["llm_call"].state == "CLOSED"
    assert panel.circuits["llm_call"].failure_count == 0


def test_cost_panel_accumulates_step_costs() -> None:
    panel = CostPanel()

    panel.update_from_event(
        {
            "event": "step_cost_recorded",
            "step_name": "fetch",
            "model": "gpt-4o-mini",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost_usd": 0.01,
            "cumulative_usd": 0.01,
        }
    )
    panel.update_from_event(
        {
            "event": "step_cost_recorded",
            "step_name": "analyze",
            "model": "gpt-4o",
            "input_tokens": 200,
            "output_tokens": 100,
            "cost_usd": 0.02,
            "cumulative_usd": 0.03,
        }
    )
    panel.update_from_event(
        {
            "event": "step_cost_recorded",
            "step_name": "summarize",
            "model": "o4-mini",
            "input_tokens": 300,
            "output_tokens": 150,
            "cost_usd": 0.04,
            "cumulative_usd": 0.07,
        }
    )

    assert panel.total_usd == pytest.approx(0.07)
    assert len(panel.step_costs) == 3


def test_event_log_panel_caps_buffer_at_500_lines() -> None:
    panel = EventLogPanel()

    for index in range(600):
        panel.update_from_event(
            {
                "event": "step_started",
                "function_name": f"step_{index}",
                "_timestamp": float(index),
                "level": "info",
            }
        )

    assert len(panel.lines) == 500


def test_dashboard_import_guard_raises_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == "textual":
            raise ImportError("missing textual")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="pip install agentarmor\\[dashboard\\]"):
        Dashboard()


def test_configured_dashboard_logging_publishes_events_to_bus() -> None:
    EventBus.reset()
    configure_logging(enable_dashboard=True)

    @armor(name="bus_step")
    def step() -> int:
        return 1

    try:
        assert step() == 1
        events = EventBus.get().poll(max_events=20)
    finally:
        configure_logging(enable_dashboard=False)
        EventBus.reset()

    assert any(event.get("function_name") == "bus_step" for event in events)
    assert any(event.get("event") == "step_started" for event in events)
