from __future__ import annotations

import pytest

from agentarmor import FailedStep, StepResult, StepStatus, WorkflowResult


def test_workflow_result_completed_status_when_all_steps_complete() -> None:
    result = WorkflowResult(
        workflow_id="workflow",
        run_id="run-1",
        status="completed",
        steps=[
            StepResult("step_1", StepStatus.COMPLETED, value=1),
            StepResult("step_2", StepStatus.COMPLETED, value=2),
            StepResult("step_3", StepStatus.COMPLETED, value=3),
        ],
        total_cost_usd=0.0,
        total_duration_ms=10.0,
    )

    assert result.status == "completed"
    assert result.is_complete is True
    assert result.is_partial is False


def test_workflow_result_partial_status_when_mixed() -> None:
    result = WorkflowResult(
        workflow_id="workflow",
        run_id="run-1",
        status="partial",
        steps=[
            StepResult("step_1", StepStatus.COMPLETED, value=1),
            StepResult("step_2", StepStatus.COMPLETED, value=2),
            StepResult("step_3", StepStatus.FAILED, exception=ValueError("boom")),
        ],
        total_cost_usd=0.0,
        total_duration_ms=10.0,
    )

    assert result.status == "partial"
    assert result.is_partial is True
    assert len(result.completed_steps) == 2
    assert len(result.failed_steps) == 1


def test_workflow_result_failed_status_when_all_steps_fail() -> None:
    result = WorkflowResult(
        workflow_id="workflow",
        run_id="run-1",
        status="failed",
        steps=[
            StepResult("step_1", StepStatus.FAILED, exception=ValueError("boom")),
            StepResult("step_2", StepStatus.FAILED, exception=RuntimeError("boom")),
        ],
        total_cost_usd=0.0,
        total_duration_ms=10.0,
    )

    assert result.status == "failed"


def test_workflow_result_get_returns_completed_value() -> None:
    result = WorkflowResult(
        workflow_id="workflow",
        run_id="run-1",
        status="completed",
        steps=[
            StepResult("step_1", StepStatus.COMPLETED, value=1),
            StepResult("step_2", StepStatus.COMPLETED, value=2),
            StepResult("step_3", StepStatus.COMPLETED, value=3),
        ],
        total_cost_usd=0.0,
        total_duration_ms=10.0,
    )

    assert result.get("step_2") == 2


def test_workflow_result_get_raises_for_missing_step() -> None:
    result = WorkflowResult(
        workflow_id="workflow",
        run_id="run-1",
        status="completed",
        steps=[StepResult("step_1", StepStatus.COMPLETED, value=1)],
        total_cost_usd=0.0,
        total_duration_ms=10.0,
    )

    with pytest.raises(KeyError, match="nonexistent"):
        result.get("nonexistent")


def test_workflow_result_get_raises_for_failed_step() -> None:
    result = WorkflowResult(
        workflow_id="workflow",
        run_id="run-1",
        status="partial",
        steps=[StepResult("step_1", StepStatus.FAILED, exception=ValueError("boom"))],
        total_cost_usd=0.0,
        total_duration_ms=10.0,
    )

    with pytest.raises(KeyError, match="step_1"):
        result.get("step_1")


def test_failed_step_is_falsy() -> None:
    failed = FailedStep("step", ValueError("boom"))

    assert bool(failed) is False


def test_checkpoint_restored_step_counts_as_completed() -> None:
    result = WorkflowResult(
        workflow_id="workflow",
        run_id="run-1",
        status="completed",
        steps=[
            StepResult(
                "step_1",
                StepStatus.CHECKPOINT_RESTORED,
                value="cached",
                from_checkpoint=True,
            )
        ],
        total_cost_usd=0.0,
        total_duration_ms=10.0,
    )

    assert result.completed_steps[0].step_name == "step_1"
    assert result.get("step_1") == "cached"
