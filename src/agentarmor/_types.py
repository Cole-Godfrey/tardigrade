from __future__ import annotations

import enum
import random
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(slots=True)
class ArmorResult:
    value: Any
    duration_ms: float
    status: Literal["success", "error"]
    exception: BaseException | None
    timestamp: datetime


class AgentArmorSerializationError(Exception):
    """Raised when checkpoint serialization or deserialization fails."""


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AgentArmorCircuitOpenError(Exception):
    """Raised when circuit is open and no fallback is configured."""

    def __init__(self, function_name: str, state: CircuitState) -> None:
        self.function_name = function_name
        self.state = state
        super().__init__(
            f"Circuit breaker for '{function_name}' is {state.value} "
            f"and no fallback is configured"
        )


class BudgetPolicy(enum.Enum):
    HARD_STOP = "hard_stop"
    WARN = "warn"
    DEGRADE = "degrade"


class DegradationPolicy(enum.Enum):
    RAISE = "raise"
    COLLECT = "collect"
    COLLECT_AND_STOP = "collect_and_stop"


class AgentArmorBudgetExceededError(Exception):
    def __init__(
        self,
        workflow_id: str,
        current_spend: float,
        max_budget: float,
    ) -> None:
        self.workflow_id = workflow_id
        self.current_spend = current_spend
        self.max_budget = max_budget
        super().__init__(
            f"Budget exceeded for workflow '{workflow_id}': "
            f"${current_spend:.4f} >= ${max_budget:.4f}"
        )


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class StepCostReport:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_budget_usd: float
    policy: BudgetPolicy = BudgetPolicy.HARD_STOP
    warn_at_fraction: float = 0.8
    on_degrade: Callable[[float, float], None] | None = None

    def __post_init__(self) -> None:
        if self.max_budget_usd < 0:
            msg = "max_budget_usd must be greater than or equal to 0"
            raise ValueError(msg)
        if not 0 <= self.warn_at_fraction <= 1:
            msg = "warn_at_fraction must be between 0 and 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DegradationConfig:
    policy: DegradationPolicy = DegradationPolicy.COLLECT
    skip_dependent: bool = True
    on_step_failure: Callable[[str, BaseException], None] | None = None
    max_failures: int | None = None

    def __post_init__(self) -> None:
        if self.max_failures is not None and self.max_failures < 1:
            msg = "max_failures must be at least 1 when provided"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1
    success_threshold: int = 2
    fallback: Callable[..., Any] | None = None
    monitored_exceptions: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            msg = "failure_threshold must be at least 1"
            raise ValueError(msg)
        if self.recovery_timeout < 0:
            msg = "recovery_timeout must be greater than or equal to 0"
            raise ValueError(msg)
        if self.half_open_max_calls < 1:
            msg = "half_open_max_calls must be at least 1"
            raise ValueError(msg)
        if self.success_threshold < 1:
            msg = "success_threshold must be at least 1"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            msg = "max_attempts must be at least 1"
            raise ValueError(msg)
        if self.base_delay < 0:
            msg = "base_delay must be greater than or equal to 0"
            raise ValueError(msg)
        if self.max_delay < 0:
            msg = "max_delay must be greater than or equal to 0"
            raise ValueError(msg)
        if self.exponential_base <= 0:
            msg = "exponential_base must be greater than 0"
            raise ValueError(msg)

    def delay_for_attempt(self, attempt: int) -> float:
        if attempt < 1:
            msg = "attempt must be at least 1"
            raise ValueError(msg)

        delay = min(
            self.base_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay,
        )
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return delay
