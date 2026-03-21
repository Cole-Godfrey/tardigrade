from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from agentarmor import (
    BudgetConfig,
    CircuitBreakerConfig,
    DegradationConfig,
    DegradationPolicy,
    FailedStep,
    RetryConfig,
    SQLiteCheckpointStore,
    StepCostReport,
    StepStatus,
    Workflow,
    armor,
)


def _step_map(workflow: Workflow) -> dict[str, object]:
    result = workflow.result
    assert result is not None
    return {step.step_name: step for step in result.steps}


def test_sync_degradation_raise_policy_preserves_exceptions() -> None:
    @armor(name="step")
    def step() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        with Workflow(
            "pipeline",
            degradation=DegradationConfig(policy=DegradationPolicy.RAISE),
        ):
            step()


def test_sync_degradation_collect_catches_failure_and_continues() -> None:
    @armor(name="step_1")
    def step_1() -> int:
        return 1

    @armor(name="step_2")
    def step_2() -> str:
        raise ValueError("boom")

    @armor(name="step_3")
    def step_3() -> int:
        return 3

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        first = step_1()
        second = step_2()
        third = step_3()

    assert first == 1
    assert isinstance(second, FailedStep)
    assert third == 3
    assert workflow.result is not None
    assert workflow.result.status == "partial"
    assert [step.step_name for step in workflow.result.completed_steps] == ["step_1", "step_3"]
    assert [step.step_name for step in workflow.result.failed_steps] == ["step_2"]


def test_sync_degradation_failed_step_returns_sentinel() -> None:
    @armor(name="step")
    def step() -> str:
        raise ValueError("boom")

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ):
        result = step()

    assert isinstance(result, FailedStep)
    assert bool(result) is False


def test_sync_degradation_skip_dependent_steps() -> None:
    calls = {"step_a": 0, "step_b": 0, "step_c": 0}

    @armor(name="step_a")
    def step_a() -> int:
        calls["step_a"] += 1
        raise ValueError("boom")

    @armor(name="step_b")
    def step_b(value: object) -> str:
        calls["step_b"] += 1
        return f"unexpected:{value}"

    @armor(name="step_c")
    def step_c() -> str:
        calls["step_c"] += 1
        return "done"

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        value_a = step_a()
        value_b = step_b(value_a)
        value_c = step_c()

    steps = _step_map(workflow)
    assert isinstance(value_a, FailedStep)
    assert isinstance(value_b, FailedStep)
    assert value_c == "done"
    assert calls == {"step_a": 1, "step_b": 0, "step_c": 1}
    assert steps["step_a"].status is StepStatus.FAILED
    assert steps["step_b"].status is StepStatus.SKIPPED
    assert steps["step_c"].status is StepStatus.COMPLETED


def test_sync_degradation_skip_dependent_can_be_disabled() -> None:
    calls = {"step_a": 0, "step_b": 0}

    @armor(name="step_a")
    def step_a() -> int:
        calls["step_a"] += 1
        raise ValueError("boom")

    @armor(name="step_b")
    def step_b(value: object) -> str:
        calls["step_b"] += 1
        return f"handled:{isinstance(value, FailedStep)}"

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(
            policy=DegradationPolicy.COLLECT,
            skip_dependent=False,
        ),
    ) as workflow:
        value_a = step_a()
        value_b = step_b(value_a)

    steps = _step_map(workflow)
    assert isinstance(value_a, FailedStep)
    assert value_b == "handled:True"
    assert calls == {"step_a": 1, "step_b": 1}
    assert steps["step_b"].status is StepStatus.COMPLETED


def test_sync_degradation_collect_and_stop_skips_remaining_steps() -> None:
    calls = {"step_1": 0, "step_2": 0, "step_3": 0}

    @armor(name="step_1")
    def step_1() -> str:
        calls["step_1"] += 1
        raise ValueError("boom")

    @armor(name="step_2")
    def step_2() -> str:
        calls["step_2"] += 1
        return "step_2"

    @armor(name="step_3")
    def step_3() -> str:
        calls["step_3"] += 1
        return "step_3"

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT_AND_STOP),
    ) as workflow:
        first = step_1()
        second = step_2()
        third = step_3()

    steps = _step_map(workflow)
    assert isinstance(first, FailedStep)
    assert isinstance(second, FailedStep)
    assert isinstance(third, FailedStep)
    assert calls == {"step_1": 1, "step_2": 0, "step_3": 0}
    assert workflow.result is not None
    assert workflow.result.status == "failed"
    assert steps["step_1"].status is StepStatus.FAILED
    assert steps["step_2"].status is StepStatus.SKIPPED
    assert steps["step_3"].status is StepStatus.SKIPPED


def test_sync_degradation_max_failures_escalates_to_stop() -> None:
    calls = {"step_1": 0, "step_2": 0, "step_3": 0, "step_4": 0}

    @armor(name="step_1")
    def step_1() -> str:
        calls["step_1"] += 1
        return "ok"

    @armor(name="step_2")
    def step_2() -> str:
        calls["step_2"] += 1
        raise ValueError("boom-2")

    @armor(name="step_3")
    def step_3() -> str:
        calls["step_3"] += 1
        raise ValueError("boom-3")

    @armor(name="step_4")
    def step_4() -> str:
        calls["step_4"] += 1
        return "never"

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(
            policy=DegradationPolicy.COLLECT,
            max_failures=2,
        ),
    ) as workflow:
        step_1()
        step_2()
        step_3()
        step_4()

    steps = _step_map(workflow)
    assert calls == {"step_1": 1, "step_2": 1, "step_3": 1, "step_4": 0}
    assert steps["step_1"].status is StepStatus.COMPLETED
    assert steps["step_2"].status is StepStatus.FAILED
    assert steps["step_3"].status is StepStatus.FAILED
    assert steps["step_4"].status is StepStatus.SKIPPED


def test_sync_degradation_on_step_failure_callback_is_called() -> None:
    failures: list[tuple[str, str]] = []

    @armor(name="step")
    def step() -> str:
        raise ValueError("boom")

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(
            policy=DegradationPolicy.COLLECT,
            on_step_failure=lambda step_name, exc: failures.append((step_name, str(exc))),
        ),
    ):
        step()

    assert failures == [("step", "boom")]


def test_sync_degradation_workflow_result_available_after_block() -> None:
    @armor(name="step")
    def step() -> int:
        return 1

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        step()

    assert workflow.result is not None
    assert workflow.result.get("step") == 1


def test_sync_degradation_checkpoint_restores_partial_results(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0, "step_3": 0}

    @armor(name="step_1")
    def step_1() -> int:
        calls["step_1"] += 1
        return 1

    @armor(name="step_2")
    def step_2() -> int:
        calls["step_2"] += 1
        return 2

    @armor(name="step_3")
    def step_3() -> str:
        calls["step_3"] += 1
        if calls["step_3"] == 1:
            raise RuntimeError("boom")
        return "done"

    try:
        with Workflow(
            "pipeline",
            run_id="run-1",
            store=store,
            degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
        ) as workflow_1:
            step_1()
            step_2()
            step_3()

        with Workflow(
            "pipeline",
            run_id="run-1",
            store=store,
            degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
        ) as workflow_2:
            step_1()
            step_2()
            step_3()

        first_run = _step_map(workflow_1)
        second_run = _step_map(workflow_2)
        assert first_run["step_1"].status is StepStatus.COMPLETED
        assert first_run["step_2"].status is StepStatus.COMPLETED
        assert first_run["step_3"].status is StepStatus.FAILED
        assert second_run["step_1"].status is StepStatus.CHECKPOINT_RESTORED
        assert second_run["step_2"].status is StepStatus.CHECKPOINT_RESTORED
        assert second_run["step_3"].status is StepStatus.COMPLETED
        assert calls == {"step_1": 1, "step_2": 1, "step_3": 2}
    finally:
        store.close()


def test_sync_degradation_budget_exceeded_is_captured_as_workflow_error() -> None:
    @armor(name="step_1")
    def step_1() -> str:
        return "ok-1", StepCostReport(cost_usd=0.005, model="gpt-4o-mini")

    @armor(name="step_2")
    def step_2() -> str:
        return "ok-2", StepCostReport(cost_usd=0.020, model="gpt-4o")

    with Workflow(
        "pipeline",
        budget=BudgetConfig(max_budget_usd=0.01),
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        step_1()
        step_2()

    assert workflow.result is not None
    assert workflow.result.status == "partial"
    assert workflow.result.get("step_1") == "ok-1"
    assert workflow.result.get("step_2") == "ok-2"
    assert workflow.result.failed_steps[0].step_name == "_workflow_error"


def test_sync_degradation_circuit_breaker_fallback_is_recorded_as_completed() -> None:
    def fallback() -> str:
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
        raise ValueError("boom")

    step._circuit_breaker.record_failure()

    with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        result = step()

    steps = _step_map(workflow)
    assert result == "fallback-result"
    assert steps["step"].status is StepStatus.COMPLETED
    assert workflow.result is not None
    assert workflow.result.get("step") == "fallback-result"


def test_sync_degradation_retries_exhaust_then_fail_once(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0}

    @armor(name="step_1", retry=RetryConfig(max_attempts=3, base_delay=1.0, jitter=False))
    def step_1() -> str:
        calls["step_1"] += 1
        raise ValueError("boom")

    @armor(name="step_2")
    def step_2() -> str:
        calls["step_2"] += 1
        return "ok"

    try:
        with patch("agentarmor._decorator.time.sleep"):
            with Workflow(
                "pipeline",
                store=store,
                degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
            ) as workflow:
                first = step_1()
                second = step_2()

        steps = _step_map(workflow)
        assert isinstance(first, FailedStep)
        assert second == "ok"
        assert calls == {"step_1": 3, "step_2": 1}
        assert len(workflow.result.failed_steps) == 1  # type: ignore[union-attr]
        assert steps["step_1"].status is StepStatus.FAILED
        assert steps["step_2"].status is StepStatus.COMPLETED
    finally:
        store.close()
