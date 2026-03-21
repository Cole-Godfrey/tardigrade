from __future__ import annotations

import pytest

from agentarmor import (
    DegradationConfig,
    DegradationPolicy,
    FailedStep,
    StepStatus,
    Workflow,
    armor,
)


def _step_map(workflow: Workflow) -> dict[str, object]:
    result = workflow.result
    assert result is not None
    return {step.step_name: step for step in result.steps}


@pytest.mark.asyncio
async def test_async_degradation_collect_catches_failure_and_continues() -> None:
    @armor(name="step_1")
    async def step_1() -> int:
        return 1

    @armor(name="step_2")
    async def step_2() -> str:
        raise ValueError("boom")

    @armor(name="step_3")
    async def step_3() -> int:
        return 3

    async with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        first = await step_1()
        second = await step_2()
        third = await step_3()

    assert first == 1
    assert isinstance(second, FailedStep)
    assert third == 3
    assert workflow.result is not None
    assert workflow.result.status == "partial"
    assert [step.step_name for step in workflow.result.completed_steps] == ["step_1", "step_3"]
    assert [step.step_name for step in workflow.result.failed_steps] == ["step_2"]


@pytest.mark.asyncio
async def test_async_degradation_failed_step_returns_sentinel() -> None:
    @armor(name="step")
    async def step() -> str:
        raise ValueError("boom")

    async with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ):
        result = await step()

    assert isinstance(result, FailedStep)
    assert bool(result) is False


@pytest.mark.asyncio
async def test_async_degradation_skip_dependent_steps() -> None:
    calls = {"step_a": 0, "step_b": 0, "step_c": 0}

    @armor(name="step_a")
    async def step_a() -> int:
        calls["step_a"] += 1
        raise ValueError("boom")

    @armor(name="step_b")
    async def step_b(value: object) -> str:
        calls["step_b"] += 1
        return f"unexpected:{value}"

    @armor(name="step_c")
    async def step_c() -> str:
        calls["step_c"] += 1
        return "done"

    async with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        value_a = await step_a()
        value_b = await step_b(value_a)
        value_c = await step_c()

    steps = _step_map(workflow)
    assert isinstance(value_a, FailedStep)
    assert isinstance(value_b, FailedStep)
    assert value_c == "done"
    assert calls == {"step_a": 1, "step_b": 0, "step_c": 1}
    assert steps["step_a"].status is StepStatus.FAILED
    assert steps["step_b"].status is StepStatus.SKIPPED
    assert steps["step_c"].status is StepStatus.COMPLETED


@pytest.mark.asyncio
async def test_async_degradation_collect_and_stop_skips_remaining_steps() -> None:
    calls = {"step_1": 0, "step_2": 0, "step_3": 0}

    @armor(name="step_1")
    async def step_1() -> str:
        calls["step_1"] += 1
        raise ValueError("boom")

    @armor(name="step_2")
    async def step_2() -> str:
        calls["step_2"] += 1
        return "step_2"

    @armor(name="step_3")
    async def step_3() -> str:
        calls["step_3"] += 1
        return "step_3"

    async with Workflow(
        "pipeline",
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT_AND_STOP),
    ) as workflow:
        first = await step_1()
        second = await step_2()
        third = await step_3()

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
