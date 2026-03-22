from typing import Any

from ._checkpoint import CheckpointStore, SQLiteCheckpointStore
from ._context import ArmorContext, report_cost
from ._cost import DEFAULT_PRICING
from ._decorator import armor
from ._logging import configure_logging
from ._result import FailedStep, StepResult, StepStatus, WorkflowResult
from ._types import (
    BudgetConfig,
    BudgetPolicy,
    CheckpointConfig,
    CircuitBreakerConfig,
    CircuitState,
    DegradationConfig,
    DegradationPolicy,
    RetryConfig,
    StepCostReport,
    TardigradeBudgetExceededError,
)
from ._workflow import Workflow

__all__ = [
    "armor",
    "ArmorContext",
    "RetryConfig",
    "CheckpointConfig",
    "BudgetConfig",
    "BudgetPolicy",
    "DegradationConfig",
    "DegradationPolicy",
    "StepCostReport",
    "TardigradeBudgetExceededError",
    "CircuitBreakerConfig",
    "CircuitState",
    "FailedStep",
    "StepResult",
    "StepStatus",
    "WorkflowResult",
    "CheckpointStore",
    "SQLiteCheckpointStore",
    "DEFAULT_PRICING",
    "Dashboard",
    "configure_logging",
    "report_cost",
    "Workflow",
]


def __getattr__(name: str) -> Any:
    if name == "Dashboard":
        from .dashboard import Dashboard

        return Dashboard
    raise AttributeError(f"module 'tardigrade' has no attribute {name}")
