from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agentarmor import (
    AgentArmorBudgetExceededError,
    BudgetConfig,
    BudgetPolicy,
    RetryConfig,
    SQLiteCheckpointStore,
    StepCostReport,
    Workflow,
    armor,
    report_cost,
)
from agentarmor._serializer import deserialize_result


@pytest.mark.asyncio
async def test_async_budget_extracts_cost_report_from_return_value(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")

    @armor(name="step")
    async def step() -> str:
        return "result", StepCostReport(cost_usd=0.05, model="gpt-4o-mini")

    try:
        async with Workflow("pipeline", run_id="run-1", store=store) as workflow:
            result = await step()

        assert result == "result"
        assert workflow.cost_tracker.total_usd == pytest.approx(0.05)
        assert workflow.cost_tracker.step_costs[0]["step_name"] == "step"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_budget_records_context_report(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")

    @armor(name="step")
    async def step() -> str:
        report_cost(StepCostReport(cost_usd=0.07, model="gpt-4o"))
        return "done"

    try:
        async with Workflow("pipeline", run_id="run-1", store=store) as workflow:
            assert await step() == "done"

        assert workflow.cost_tracker.total_usd == pytest.approx(0.07)
        assert workflow.cost_tracker.step_costs[0]["model"] == "gpt-4o"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_budget_exceeded_stops_workflow_after_checkpoint(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0}

    @armor(name="step_1")
    async def step_1() -> int:
        calls["step_1"] += 1
        return 1, StepCostReport(cost_usd=0.02, model="gpt-4o")

    @armor(name="step_2")
    async def step_2() -> str:
        calls["step_2"] += 1
        return "should-not-run"

    try:
        with pytest.raises(AgentArmorBudgetExceededError) as exc_info:
            async with Workflow(
                "pipeline",
                run_id="run-1",
                store=store,
                budget=BudgetConfig(max_budget_usd=0.01, policy=BudgetPolicy.HARD_STOP),
            ):
                await step_1()
                await step_2()

        checkpoint = await store.aload("pipeline", "step_1", "run-1")
        assert exc_info.value.workflow_id == "pipeline"
        assert checkpoint is not None
        assert deserialize_result(checkpoint) == 1
        assert calls == {"step_1": 1, "step_2": 0}
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_budget_tracks_cost_once_after_retries(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = 0

    @armor(name="step", retry=RetryConfig(max_attempts=3, base_delay=1.0, jitter=False))
    async def step() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("retry me")
        return "done", StepCostReport(cost_usd=0.01, model="gpt-4o-mini")

    try:
        with patch("agentarmor._decorator.asyncio.sleep", new=AsyncMock()):
            async with Workflow("pipeline", run_id="run-1", store=store) as workflow:
                assert await step() == "done"

        assert calls == 3
        assert workflow.cost_tracker.total_usd == pytest.approx(0.01)
        assert len(workflow.cost_tracker.step_costs) == 1
    finally:
        await store.aclose()
