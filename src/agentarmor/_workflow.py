from __future__ import annotations

import time
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Literal
from uuid import uuid4

import structlog

from ._checkpoint import CheckpointStore, SQLiteCheckpointStore
from ._cost import CostTracker
from ._logging import configure_logging
from ._result import StepResult, StepStatus, WorkflowResult
from ._types import (
    BudgetConfig,
    DegradationConfig,
    DegradationPolicy,
)

configure_logging()
_logger = structlog.get_logger("agentarmor")


class Workflow:
    def __init__(
        self,
        workflow_id: str,
        run_id: str | None = None,
        store: CheckpointStore | None = None,
        budget: BudgetConfig | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
        degradation: DegradationConfig | None = None,
    ) -> None:
        self.workflow_id = workflow_id
        self.run_id = str(uuid4()) if run_id is None else run_id
        self.store: CheckpointStore = SQLiteCheckpointStore() if store is None else store
        self.cost_tracker = CostTracker(
            budget_config=budget,
            pricing=pricing,
        )
        self._degradation = degradation or DegradationConfig(
            policy=DegradationPolicy.RAISE,
        )
        self._step_results: list[StepResult] = []
        self._failure_count = 0
        self._start_time: float | None = None
        self._result: WorkflowResult | None = None
        self._stop_remaining = False
        self._token: Token[Workflow | None] | None = None

    def __enter__(self) -> Workflow:
        self._prepare_run()
        self._token = set_current_workflow(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        elapsed = self._elapsed_ms()

        if self._degradation.policy is DegradationPolicy.RAISE:
            self._build_result(elapsed)
            self._reset_context()
            return False

        if exc_type is not None and exc is not None:
            self.record_step_result(
                StepResult(
                    step_name="_workflow_error",
                    status=StepStatus.FAILED,
                    exception=exc,
                )
            )

        self._build_result(elapsed)
        self._reset_context()
        return exc_type is not None

    async def __aenter__(self) -> Workflow:
        self._prepare_run()
        self._token = set_current_workflow(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        elapsed = self._elapsed_ms()

        if self._degradation.policy is DegradationPolicy.RAISE:
            self._build_result(elapsed)
            self._reset_context()
            return False

        if exc_type is not None and exc is not None:
            self.record_step_result(
                StepResult(
                    step_name="_workflow_error",
                    status=StepStatus.FAILED,
                    exception=exc,
                )
            )

        self._build_result(elapsed)
        self._reset_context()
        return exc_type is not None

    def clear(self) -> None:
        self.store.clear_run(self.workflow_id, self.run_id)

    async def aclear(self) -> None:
        await self.store.aclear_run(self.workflow_id, self.run_id)

    @property
    def degradation_config(self) -> DegradationConfig:
        return self._degradation

    @property
    def result(self) -> WorkflowResult | None:
        return self._result

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def record_step_result(self, result: StepResult) -> None:
        self._step_results.append(result)

        if result.status is StepStatus.FAILED:
            self._failure_count += 1
            if self._degradation.on_step_failure is not None and result.exception is not None:
                self._degradation.on_step_failure(result.step_name, result.exception)
            if self._degradation.policy is DegradationPolicy.COLLECT_AND_STOP:
                self._stop_remaining = True
            if (
                self._degradation.max_failures is not None
                and self._failure_count >= self._degradation.max_failures
            ):
                self._stop_remaining = True

        if self._degradation.policy is not DegradationPolicy.RAISE:
            _logger.info(
                "step_result_recorded",
                step_name=result.step_name,
                status=result.status.value,
            )

    def should_skip(self) -> bool:
        if self._degradation.policy is DegradationPolicy.RAISE:
            return False
        return self._stop_remaining

    def should_degrade_failures(self) -> bool:
        return self._degradation.policy is not DegradationPolicy.RAISE

    def _prepare_run(self) -> None:
        self._step_results = []
        self._failure_count = 0
        self._result = None
        self._stop_remaining = False
        self._start_time = time.monotonic()
        self.cost_tracker.reset()

    def _elapsed_ms(self) -> float:
        if self._start_time is None:
            return 0.0
        return (time.monotonic() - self._start_time) * 1000

    def _build_result(self, elapsed: float) -> None:
        completed = [
            step
            for step in self._step_results
            if step.status in {StepStatus.COMPLETED, StepStatus.CHECKPOINT_RESTORED}
        ]
        failed = [step for step in self._step_results if step.status is StepStatus.FAILED]
        skipped = [step for step in self._step_results if step.status is StepStatus.SKIPPED]

        status: Literal["completed", "partial", "failed"]
        if not failed:
            status = "completed"
        elif not completed:
            status = "failed"
        else:
            status = "partial"

        self._result = WorkflowResult(
            workflow_id=self.workflow_id,
            run_id=self.run_id,
            status=status,
            steps=list(self._step_results),
            total_cost_usd=self.cost_tracker.total_usd,
            total_duration_ms=elapsed,
        )
        if self._degradation.policy is not DegradationPolicy.RAISE:
            _logger.info(
                "workflow_completed",
                workflow_id=self.workflow_id,
                run_id=self.run_id,
                status=status,
                completed_count=len(completed),
                failed_count=len(failed),
                skipped_count=len(skipped),
                total_cost_usd=self.cost_tracker.total_usd,
                total_duration_ms=elapsed,
            )

    def _reset_context(self) -> None:
        token = self._token
        if token is None:
            return
        reset_current_workflow(token)
        self._token = None


_current_workflow: ContextVar[Workflow | None] = ContextVar(
    "agentarmor_current_workflow",
    default=None,
)


def get_current_workflow() -> Workflow | None:
    return _current_workflow.get()


def set_current_workflow(workflow: Workflow) -> Token[Workflow | None]:
    return _current_workflow.set(workflow)


def reset_current_workflow(token: Token[Workflow | None]) -> None:
    _current_workflow.reset(token)
