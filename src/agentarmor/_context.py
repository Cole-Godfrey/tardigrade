from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ._types import StepCostReport


@dataclass(slots=True)
class ArmorContext:
    function_name: str
    call_id: str
    attempt: int
    max_attempts: int
    start_time: datetime
    config: dict[str, Any]
    cost_report: StepCostReport | None = None


_current_armor_context: ContextVar[ArmorContext | None] = ContextVar(
    "agentarmor_current_context",
    default=None,
)


def get_current_armor_context() -> ArmorContext | None:
    return _current_armor_context.get()


def set_current_armor_context(context: ArmorContext) -> Token[ArmorContext | None]:
    return _current_armor_context.set(context)


def reset_current_armor_context(token: Token[ArmorContext | None]) -> None:
    _current_armor_context.reset(token)


def report_cost(report: StepCostReport) -> None:
    context = get_current_armor_context()
    if context is None:
        msg = "No active ArmorContext; report_cost() must be called inside @armor"
        raise RuntimeError(msg)
    context.cost_report = report
