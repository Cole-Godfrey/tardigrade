from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Literal


class StepStatus(enum.Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CHECKPOINT_RESTORED = "checkpoint_restored"


@dataclass(slots=True)
class StepResult:
    step_name: str
    status: StepStatus
    value: Any = None
    exception: BaseException | None = None
    duration_ms: float = 0.0
    attempt: int = 0
    cost_usd: float = 0.0
    from_checkpoint: bool = False


@dataclass(slots=True)
class WorkflowResult:
    workflow_id: str
    run_id: str
    status: Literal["completed", "partial", "failed"]
    steps: list[StepResult]
    total_cost_usd: float
    total_duration_ms: float

    @property
    def completed_steps(self) -> list[StepResult]:
        return [
            step
            for step in self.steps
            if step.status in {StepStatus.COMPLETED, StepStatus.CHECKPOINT_RESTORED}
        ]

    @property
    def failed_steps(self) -> list[StepResult]:
        return [step for step in self.steps if step.status is StepStatus.FAILED]

    @property
    def skipped_steps(self) -> list[StepResult]:
        return [step for step in self.steps if step.status is StepStatus.SKIPPED]

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"

    @property
    def is_partial(self) -> bool:
        return self.status == "partial"

    def get(self, step_name: str) -> Any:
        for step in self.steps:
            if step.step_name != step_name:
                continue
            if step.status in {StepStatus.COMPLETED, StepStatus.CHECKPOINT_RESTORED}:
                return step.value
            break

        raise KeyError(f"No completed result for step '{step_name}'")


class FailedStep:
    """Sentinel returned when a step fails in degradation mode."""

    def __init__(self, step_name: str, exception: BaseException) -> None:
        self.step_name = step_name
        self.exception = exception

    def __repr__(self) -> str:
        return f"FailedStep({self.step_name!r}, {self.exception!r})"

    def __bool__(self) -> bool:
        return False
