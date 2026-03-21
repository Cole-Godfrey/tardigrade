"""Cost tracking utilities for workflow-scoped budget enforcement.

Model pricing changes frequently. Users should override prices via
CostTracker.update_pricing() or pass StepCostReport.cost_usd when they have
exact provider-side billing information available.
"""

from __future__ import annotations

import threading
from typing import Any

import structlog

from ._logging import configure_logging
from ._types import (
    AgentArmorBudgetExceededError,
    BudgetConfig,
    BudgetPolicy,
    StepCostReport,
)

DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-3-5": (0.80, 4.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    "llama-4-maverick": (0.20, 0.60),
    "llama-4-scout": (0.15, 0.40),
}

configure_logging()
_logger = structlog.get_logger("agentarmor")


class CostTracker:
    def __init__(
        self,
        budget_config: BudgetConfig | None = None,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._budget = budget_config
        self._pricing = {**DEFAULT_PRICING, **(pricing or {})}
        self._total_usd = 0.0
        self._step_costs: list[dict[str, Any]] = []
        self._warned = False
        self._lock = threading.Lock()

    def calculate_cost(self, report: StepCostReport) -> float:
        if report.cost_usd is not None:
            return report.cost_usd

        pricing = self._pricing.get(report.model)
        if pricing is None:
            _logger.warning(
                "unknown_model_pricing",
                model=report.model,
            )
            return 0.0

        input_price, output_price = pricing
        return (
            (report.input_tokens * input_price) + (report.output_tokens * output_price)
        ) / 1_000_000

    def record(
        self,
        step_name: str,
        report: StepCostReport,
        *,
        restored_from_checkpoint: bool = False,
    ) -> float:
        with self._lock:
            cost_usd = self.calculate_cost(report)
            self._total_usd += cost_usd
            entry = {
                "step_name": step_name,
                "model": report.model,
                "input_tokens": report.input_tokens,
                "output_tokens": report.output_tokens,
                "cost_usd": cost_usd,
                "cumulative_usd": self._total_usd,
            }
            if self._budget is not None:
                entry["max_budget_usd"] = self._budget.max_budget_usd
            if restored_from_checkpoint:
                entry["restored_from_checkpoint"] = True
            self._step_costs.append(entry)

        _logger.info(
            "step_cost_recorded",
            **entry,
        )
        return cost_usd

    def check_budget(self, workflow_id: str) -> None:
        budget = self._budget
        if budget is None:
            return

        with self._lock:
            total_usd = self._total_usd
            should_warn = (
                total_usd >= budget.max_budget_usd * budget.warn_at_fraction and not self._warned
            )
            if should_warn:
                self._warned = True

        if total_usd >= budget.max_budget_usd:
            _logger.warning(
                "budget_exceeded",
                workflow_id=workflow_id,
                spend_usd=total_usd,
                max_budget_usd=budget.max_budget_usd,
                current_spend=total_usd,
                max_budget=budget.max_budget_usd,
            )
            if budget.policy is BudgetPolicy.HARD_STOP:
                raise AgentArmorBudgetExceededError(
                    workflow_id=workflow_id,
                    current_spend=total_usd,
                    max_budget=budget.max_budget_usd,
                )
            if budget.policy is BudgetPolicy.WARN:
                return

            if budget.on_degrade is not None:
                budget.on_degrade(total_usd, budget.max_budget_usd)
            _logger.warning(
                "budget_degradation_triggered",
                workflow_id=workflow_id,
                spend_usd=total_usd,
                max_budget_usd=budget.max_budget_usd,
                current_spend=total_usd,
                max_budget=budget.max_budget_usd,
            )
            return

        if should_warn:
            _logger.warning(
                "budget_warning",
                workflow_id=workflow_id,
                spend_usd=total_usd,
                max_budget_usd=budget.max_budget_usd,
                fraction=budget.warn_at_fraction,
            )

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd

    @property
    def budget_config(self) -> BudgetConfig | None:
        return self._budget

    @property
    def step_costs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [entry.copy() for entry in self._step_costs]

    @property
    def remaining_usd(self) -> float | None:
        budget = self._budget
        if budget is None:
            return None
        with self._lock:
            return budget.max_budget_usd - self._total_usd

    def update_pricing(self, overrides: dict[str, tuple[float, float]]) -> None:
        with self._lock:
            self._pricing.update(overrides)

    def reset(self) -> None:
        with self._lock:
            self._total_usd = 0.0
            self._step_costs.clear()
            self._warned = False
