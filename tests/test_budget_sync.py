from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from tardigrade import (
    BudgetConfig,
    BudgetPolicy,
    CircuitBreakerConfig,
    RetryConfig,
    SQLiteCheckpointStore,
    StepCostReport,
    TardigradeBudgetExceededError,
    Workflow,
    armor,
    report_cost,
)
from tardigrade._serializer import deserialize_result


def test_sync_budget_extracts_cost_report_from_return_value(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")

    @armor(name="step")
    def step() -> str:
        return "result", StepCostReport(cost_usd=0.05, model="gpt-5-mini")

    try:
        with Workflow("pipeline", run_id="run-1", store=store) as workflow:
            result = step()

        assert result == "result"
        assert workflow.cost_tracker.total_usd == pytest.approx(0.05)
        assert workflow.cost_tracker.step_costs[0]["step_name"] == "step"
    finally:
        store.close()


def test_sync_budget_records_context_report(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")

    @armor(name="step")
    def step() -> str:
        report_cost(StepCostReport(cost_usd=0.07, model="gpt-5.4"))
        return "done"

    try:
        with Workflow("pipeline", run_id="run-1", store=store) as workflow:
            assert step() == "done"

        assert workflow.cost_tracker.total_usd == pytest.approx(0.07)
        assert workflow.cost_tracker.step_costs[0]["model"] == "gpt-5.4"
    finally:
        store.close()


def test_sync_budget_exceeded_stops_workflow_after_checkpoint(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0}

    @armor(name="step_1")
    def step_1() -> int:
        calls["step_1"] += 1
        return 1, StepCostReport(cost_usd=0.02, model="gpt-5.4")

    @armor(name="step_2")
    def step_2() -> str:
        calls["step_2"] += 1
        return "should-not-run"

    try:
        with pytest.raises(TardigradeBudgetExceededError) as exc_info:
            with Workflow(
                "pipeline",
                run_id="run-1",
                store=store,
                budget=BudgetConfig(max_budget_usd=0.01, policy=BudgetPolicy.HARD_STOP),
            ):
                step_1()
                step_2()

        checkpoint = store.load("pipeline", "step_1", "run-1")
        assert exc_info.value.workflow_id == "pipeline"
        assert checkpoint is not None
        assert deserialize_result(checkpoint) == 1
        assert calls == {"step_1": 1, "step_2": 0}
    finally:
        store.close()


def test_sync_budget_degrade_calls_callback(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls: list[tuple[float, float]] = []

    @armor(name="step")
    def step() -> str:
        return "done", StepCostReport(cost_usd=0.02, model="gpt-5.4")

    try:
        with capture_logs() as captured_logs:
            with Workflow(
                "pipeline",
                run_id="run-1",
                store=store,
                budget=BudgetConfig(
                    max_budget_usd=0.01,
                    policy=BudgetPolicy.DEGRADE,
                    on_degrade=lambda current, maximum: calls.append((current, maximum)),
                ),
            ):
                assert step() == "done"

        assert calls == [(0.02, 0.01)]
        assert "budget_degradation_triggered" in [entry["event"] for entry in captured_logs]
    finally:
        store.close()


def test_sync_budget_warning_logs_at_fraction(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")

    @armor(name="step")
    def step() -> str:
        return "done", StepCostReport(cost_usd=0.60, model="gpt-5.4")

    try:
        with capture_logs() as captured_logs:
            with Workflow(
                "pipeline",
                run_id="run-1",
                store=store,
                budget=BudgetConfig(
                    max_budget_usd=1.0,
                    policy=BudgetPolicy.WARN,
                    warn_at_fraction=0.5,
                ),
            ):
                assert step() == "done"

        warning_events = [entry for entry in captured_logs if entry["event"] == "budget_warning"]
        assert len(warning_events) == 1
        assert warning_events[0]["fraction"] == pytest.approx(0.5)
    finally:
        store.close()


def test_sync_budget_extracts_cost_report_without_workflow() -> None:
    @armor(name="step")
    def step() -> str:
        return "value", StepCostReport(cost_usd=0.40, model="gpt-5.4")

    assert step() == "value"


def test_sync_budget_uses_custom_pricing_table(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")

    @armor(name="step")
    def step() -> str:
        return "done", StepCostReport(input_tokens=100, output_tokens=50, model="my-model")

    try:
        with Workflow(
            "pipeline",
            run_id="run-1",
            store=store,
            pricing={"my-model": (1.0, 2.0)},
        ) as workflow:
            assert step() == "done"

        assert workflow.cost_tracker.total_usd == pytest.approx((100 * 1.0 + 50 * 2.0) / 1_000_000)
    finally:
        store.close()


def test_sync_budget_tracks_cost_once_after_retries(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = 0

    @armor(name="step", retry=RetryConfig(max_attempts=3, base_delay=1.0, jitter=False))
    def step() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("retry me")
        return "done", StepCostReport(cost_usd=0.01, model="gpt-5-mini")

    try:
        with patch("tardigrade._decorator.time.sleep"):
            with Workflow("pipeline", run_id="run-1", store=store) as workflow:
                assert step() == "done"

        assert calls == 3
        assert workflow.cost_tracker.total_usd == pytest.approx(0.01)
        assert len(workflow.cost_tracker.step_costs) == 1
    finally:
        store.close()


def test_sync_budget_tracks_fallback_cost(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    fallback_calls = 0

    def fallback(prompt: str) -> tuple[str, StepCostReport]:
        nonlocal fallback_calls
        fallback_calls += 1
        return "fallback", StepCostReport(cost_usd=0.03, model="gpt-5-mini")

    @armor(
        name="step",
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=1,
            fallback=fallback,
            monitored_exceptions=(ValueError,),
        ),
    )
    def step(prompt: str) -> str:
        raise ValueError("primary failed")

    breaker = step._circuit_breaker

    try:
        breaker.record_failure()

        with Workflow("pipeline", run_id="run-1", store=store) as workflow:
            assert step("hello") == "fallback"

        assert fallback_calls == 1
        assert workflow.cost_tracker.total_usd == pytest.approx(0.03)
        assert workflow.cost_tracker.step_costs[0]["model"] == "gpt-5-mini"
    finally:
        store.close()


def test_sync_budget_exposes_workflow_cost_tracker_after_run(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")

    @armor(name="step")
    def step() -> str:
        return "done", StepCostReport(cost_usd=0.02, model="gpt-5-mini")

    try:
        workflow = Workflow("pipeline", run_id="run-1", store=store)

        with workflow:
            assert step() == "done"

        assert workflow.cost_tracker.total_usd > 0
        assert workflow.cost_tracker.step_costs == [
            {
                "step_name": "step",
                "model": "gpt-5-mini",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.02,
                "cumulative_usd": 0.02,
            }
        ]
    finally:
        store.close()


def test_sync_budget_replays_checkpointed_costs_on_resume(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0}

    @armor(name="step_1")
    def step_1() -> int:
        calls["step_1"] += 1
        return 1, StepCostReport(cost_usd=0.02, model="gpt-5-mini")

    @armor(name="step_2")
    def step_2(value: int) -> str:
        calls["step_2"] += 1
        if calls["step_2"] == 1:
            raise RuntimeError("step 2 failed")
        return f"done={value}", StepCostReport(cost_usd=0.03, model="gpt-5.4")

    try:
        with pytest.raises(RuntimeError, match="step 2 failed"):
            with Workflow("pipeline", run_id="run-1", store=store):
                step_2(step_1())

        with Workflow("pipeline", run_id="run-1", store=store) as workflow:
            result = step_2(step_1())

        assert result == "done=1"
        assert calls == {"step_1": 1, "step_2": 2}
        assert workflow.cost_tracker.total_usd == pytest.approx(0.05)
        assert workflow.cost_tracker.step_costs == [
            {
                "step_name": "step_1",
                "model": "gpt-5-mini",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.02,
                "cumulative_usd": 0.02,
                "restored_from_checkpoint": True,
            },
            {
                "step_name": "step_2",
                "model": "gpt-5.4",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.03,
                "cumulative_usd": 0.05,
            },
        ]
    finally:
        store.close()


def test_sync_budget_resume_enforces_restored_spend_before_new_work(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0}

    @armor(name="step_1")
    def step_1() -> int:
        calls["step_1"] += 1
        return 1, StepCostReport(cost_usd=0.011, model="gpt-5-mini")

    @armor(name="step_2")
    def step_2(value: int) -> str:
        calls["step_2"] += 1
        return f"done={value}"

    try:
        with pytest.raises(TardigradeBudgetExceededError):
            with Workflow(
                "pipeline",
                run_id="run-1",
                store=store,
                budget=BudgetConfig(max_budget_usd=0.01, policy=BudgetPolicy.HARD_STOP),
            ):
                step_1()

        with pytest.raises(TardigradeBudgetExceededError):
            with Workflow(
                "pipeline",
                run_id="run-1",
                store=store,
                budget=BudgetConfig(max_budget_usd=0.01, policy=BudgetPolicy.HARD_STOP),
            ):
                step_2(step_1())

        assert calls == {"step_1": 1, "step_2": 0}
    finally:
        store.close()
