from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from agentarmor import (
    AgentArmorBudgetExceededError,
    BudgetConfig,
    BudgetPolicy,
    StepCostReport,
)
from agentarmor._cost import CostTracker


def test_cost_tracker_calculates_cost_for_known_model() -> None:
    tracker = CostTracker()
    report = StepCostReport(input_tokens=1000, output_tokens=500, model="gpt-5.4")

    cost = tracker.calculate_cost(report)

    assert cost == pytest.approx((1000 * 2.50 + 500 * 15.00) / 1_000_000)


def test_cost_tracker_uses_cost_override_when_provided() -> None:
    tracker = CostTracker()

    assert tracker.calculate_cost(StepCostReport(cost_usd=0.05)) == pytest.approx(0.05)


def test_cost_tracker_unknown_model_returns_zero_and_logs_warning() -> None:
    tracker = CostTracker()

    with capture_logs() as captured_logs:
        cost = tracker.calculate_cost(
            StepCostReport(input_tokens=1000, output_tokens=500, model="unknown-model-7b")
        )

    assert cost == 0.0
    warning_events = [
        entry for entry in captured_logs if entry["event"] == "unknown_model_pricing"
    ]
    assert len(warning_events) == 1
    assert warning_events[0]["model"] == "unknown-model-7b"


def test_cost_tracker_record_accumulates_history() -> None:
    tracker = CostTracker()

    tracker.record("step_1", StepCostReport(cost_usd=0.01, model="gpt-5-mini"))
    tracker.record("step_2", StepCostReport(cost_usd=0.02, model="gpt-5.4"))
    tracker.record("step_3", StepCostReport(cost_usd=0.03, model="o4-mini"))

    assert tracker.total_usd == pytest.approx(0.06)
    assert tracker.step_costs == [
        {
            "step_name": "step_1",
            "model": "gpt-5-mini",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.01,
            "cumulative_usd": 0.01,
        },
        {
            "step_name": "step_2",
            "model": "gpt-5.4",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.02,
            "cumulative_usd": 0.03,
        },
        {
            "step_name": "step_3",
            "model": "o4-mini",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.03,
            "cumulative_usd": 0.06,
        },
    ]


def test_cost_tracker_budget_hard_stop_raises() -> None:
    tracker = CostTracker(budget_config=BudgetConfig(max_budget_usd=0.10))
    tracker.record("step", StepCostReport(cost_usd=0.11))

    with capture_logs() as captured_logs:
        with pytest.raises(AgentArmorBudgetExceededError) as exc_info:
            tracker.check_budget("workflow-a")

    assert exc_info.value.workflow_id == "workflow-a"
    assert exc_info.value.current_spend == pytest.approx(0.11)
    assert exc_info.value.max_budget == pytest.approx(0.10)
    exceeded_events = [entry for entry in captured_logs if entry["event"] == "budget_exceeded"]
    assert len(exceeded_events) == 1


def test_cost_tracker_budget_warn_logs_and_continues() -> None:
    tracker = CostTracker(
        budget_config=BudgetConfig(max_budget_usd=0.10, policy=BudgetPolicy.WARN)
    )
    tracker.record("step", StepCostReport(cost_usd=0.11))

    with capture_logs() as captured_logs:
        tracker.check_budget("workflow-a")

    exceeded_events = [entry for entry in captured_logs if entry["event"] == "budget_exceeded"]
    assert len(exceeded_events) == 1
    assert exceeded_events[0]["workflow_id"] == "workflow-a"


def test_cost_tracker_budget_degrade_calls_callback() -> None:
    calls: list[tuple[float, float]] = []
    tracker = CostTracker(
        budget_config=BudgetConfig(
            max_budget_usd=0.10,
            policy=BudgetPolicy.DEGRADE,
            on_degrade=lambda current, maximum: calls.append((current, maximum)),
        )
    )
    tracker.record("step", StepCostReport(cost_usd=0.11))

    with capture_logs() as captured_logs:
        tracker.check_budget("workflow-a")

    assert calls == [(0.11, 0.10)]
    degrade_events = [
        entry for entry in captured_logs if entry["event"] == "budget_degradation_triggered"
    ]
    assert len(degrade_events) == 1


def test_cost_tracker_budget_warning_fires_once() -> None:
    tracker = CostTracker(
        budget_config=BudgetConfig(max_budget_usd=1.0, warn_at_fraction=0.8),
    )

    with capture_logs() as captured_logs:
        tracker.record("step_1", StepCostReport(cost_usd=0.85))
        tracker.check_budget("workflow-a")
        tracker.record("step_2", StepCostReport(cost_usd=0.05))
        tracker.check_budget("workflow-a")

    warning_events = [entry for entry in captured_logs if entry["event"] == "budget_warning"]
    assert len(warning_events) == 1
    assert warning_events[0]["fraction"] == pytest.approx(0.8)


def test_cost_tracker_update_pricing_applies_custom_model() -> None:
    tracker = CostTracker()
    tracker.update_pricing({"my-model": (1.00, 2.00)})

    cost = tracker.calculate_cost(
        StepCostReport(input_tokens=100, output_tokens=50, model="my-model")
    )

    assert cost == pytest.approx((100 * 1.00 + 50 * 2.00) / 1_000_000)


def test_cost_tracker_reset_clears_state() -> None:
    tracker = CostTracker(
        budget_config=BudgetConfig(max_budget_usd=1.0, warn_at_fraction=0.5),
    )
    tracker.record("step", StepCostReport(cost_usd=0.6))
    tracker.check_budget("workflow-a")

    tracker.reset()

    assert tracker.total_usd == 0.0
    assert tracker.step_costs == []

    with capture_logs() as captured_logs:
        tracker.record("step", StepCostReport(cost_usd=0.6))
        tracker.check_budget("workflow-a")

    warning_events = [entry for entry in captured_logs if entry["event"] == "budget_warning"]
    assert len(warning_events) == 1


def test_cost_tracker_remaining_budget_reports_expected_value() -> None:
    tracker = CostTracker(budget_config=BudgetConfig(max_budget_usd=1.0))
    tracker.record("step", StepCostReport(cost_usd=0.35))

    assert tracker.remaining_usd == pytest.approx(0.65)


def test_cost_tracker_remaining_budget_is_none_without_budget() -> None:
    tracker = CostTracker()

    assert tracker.remaining_usd is None
