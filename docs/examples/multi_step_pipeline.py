from __future__ import annotations

import time

from agentarmor import (
    BudgetConfig,
    DegradationConfig,
    DegradationPolicy,
    StepCostReport,
    Workflow,
    armor,
)


@armor(name="fetch")
def fetch() -> dict[str, list[int]]:
    time.sleep(0.1)
    return {"items": [1, 2, 3]}, StepCostReport(cost_usd=0.01, model="gpt-4o-mini")


@armor(name="enrich")
def enrich(data: dict[str, list[int]]) -> dict[str, list[int]]:
    time.sleep(0.1)
    raise ConnectionError("enrichment provider unavailable")


@armor(name="summarize")
def summarize(data: dict[str, list[int]]) -> str:
    time.sleep(0.1)
    return "summary ready", StepCostReport(cost_usd=0.03, model="gpt-4o")


if __name__ == "__main__":
    with Workflow(
        "pipeline-demo",
        budget=BudgetConfig(max_budget_usd=0.10),
        degradation=DegradationConfig(policy=DegradationPolicy.COLLECT),
    ) as workflow:
        raw = fetch()
        enrich(raw)
        summarize(raw)

    assert workflow.result is not None
    print(workflow.result.status)
    print([step.step_name for step in workflow.result.completed_steps])
    print([step.step_name for step in workflow.result.failed_steps])
