from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    class _StaticBase:
        def __init__(self, *args: object, **kwargs: object) -> None:
            ...

        def update(self, renderable: object) -> None:
            ...

    class _RichLogBase(_StaticBase):
        def clear(self) -> None:
            ...

        def write(self, line: str, scroll_end: bool = True) -> None:
            ...
else:
    try:
        from textual.widgets import RichLog as _RichLogBase, Static as _StaticBase  # type: ignore[import-not-found]  # noqa: I001
    except ImportError:
        class _StaticBase:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.id = kwargs.get("id")
                self.renderable: object = ""

            def update(self, renderable: object) -> None:
                self.renderable = renderable

        class _RichLogBase(_StaticBase):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.rendered_lines: list[str] = []

            def clear(self) -> None:
                self.rendered_lines.clear()
                self.renderable = ""

            def write(self, line: str, scroll_end: bool = True) -> None:
                del scroll_end
                self.rendered_lines.append(line)
                self.renderable = "\n".join(self.rendered_lines)


def _format_duration(duration_ms: object) -> str:
    if isinstance(duration_ms, (int, float)):
        return f"{duration_ms:.1f}ms"
    return "-"


def _format_attempt(attempt: object, max_attempts: object) -> str:
    if isinstance(attempt, int) and isinstance(max_attempts, int):
        return f"{attempt}/{max_attempts}"
    if isinstance(attempt, int):
        return str(attempt)
    return "-"


def _format_event_time(event: dict[str, Any]) -> str:
    timestamp = event.get("timestamp")
    if isinstance(timestamp, str):
        return timestamp.split("T")[-1].replace("Z", "")

    fallback = event.get("_timestamp")
    if isinstance(fallback, (int, float)):
        return datetime.fromtimestamp(fallback).strftime("%H:%M:%S")
    return "--:--:--"


def _render_budget_bar(total_usd: float, budget_usd: float | None) -> str:
    if budget_usd is None or budget_usd <= 0:
        return "[--------------------]"

    fraction = min(max(total_usd / budget_usd, 0.0), 1.0)
    width = 20
    filled = round(fraction * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {fraction * 100:.0f}%"


def _format_usd(amount: float) -> str:
    precision = 5 if abs(amount) < 0.01 else 4
    quantum = Decimal("1").scaleb(-precision)
    rounded = Decimal(str(amount)).quantize(quantum, rounding=ROUND_HALF_UP)
    return f"{rounded:.{precision}f}"


@dataclass(slots=True)
class WorkflowStepState:
    name: str
    status: str = "pending"
    duration_ms: float | None = None
    attempt: int | None = None
    max_attempts: int | None = None


class WorkflowPanel(_StaticBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.steps: dict[str, WorkflowStepState] = {}
        self._refresh()

    def update_from_event(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        if event_name not in {
            "step_started",
            "step_completed",
            "step_retrying",
            "step_restored_from_checkpoint",
            "step_failed_all_retries",
            "step_failed",
        }:
            return

        step_name = str(event.get("step_name") or event.get("function_name") or "<unknown>")
        state = self.steps.get(step_name, WorkflowStepState(name=step_name))

        if event_name == "step_started":
            state.status = "running"
        elif event_name == "step_completed":
            state.status = "completed"
            duration_ms = event.get("duration_ms")
            state.duration_ms = (
                float(duration_ms) if isinstance(duration_ms, (int, float)) else None
            )
        elif event_name == "step_retrying":
            state.status = "retrying"
        elif event_name == "step_restored_from_checkpoint":
            state.status = "restored"
            state.duration_ms = 0.0
        else:
            state.status = "failed"
            duration_ms = event.get("duration_ms") or event.get("total_elapsed_ms")
            state.duration_ms = (
                float(duration_ms) if isinstance(duration_ms, (int, float)) else None
            )

        attempt = event.get("attempt") or event.get("total_attempts")
        max_attempts = event.get("max_attempts")
        state.attempt = attempt if isinstance(attempt, int) else state.attempt
        state.max_attempts = (
            max_attempts if isinstance(max_attempts, int) else state.max_attempts
        )
        self.steps[step_name] = state
        self._refresh()

    def _refresh(self) -> None:
        icon_map = {
            "pending": "·",
            "running": "⏳",
            "completed": "✓",
            "retrying": "↻",
            "restored": "⚡",
            "failed": "✗",
        }
        lines = ["Workflow Progress", "Step | Status | Duration | Attempt"]
        for state in self.steps.values():
            lines.append(
                f"{state.name} | "
                f"{icon_map.get(state.status, '?')} {state.status} | "
                f"{_format_duration(state.duration_ms)} | "
                f"{_format_attempt(state.attempt, state.max_attempts)}"
            )
        self.update("\n".join(lines))


@dataclass(slots=True)
class CircuitStatus:
    function_name: str
    state: str = "CLOSED"
    failure_count: int = 0
    last_failure: str = "-"


class CircuitBreakerPanel(_StaticBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.circuits: dict[str, CircuitStatus] = {}
        self._refresh()

    def update_from_event(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        function_name = event.get("function_name")
        if not isinstance(function_name, str):
            return

        if event_name == "step_started":
            existing = self.circuits.get(function_name)
            if existing is None or existing.state != "OPEN":
                return
            existing.state = "HALF_OPEN"
            self._refresh()
            return

        if event_name not in {
            "circuit_opened",
            "circuit_closed",
            "circuit_reopened",
            "circuit_fallback",
        }:
            return

        state = self.circuits.get(function_name, CircuitStatus(function_name=function_name))
        if event_name == "circuit_opened":
            state.state = "OPEN"
            failure_count = event.get("failure_count")
            state.failure_count = failure_count if isinstance(failure_count, int) else 0
            state.last_failure = _format_event_time(event)
        elif event_name == "circuit_closed":
            state.state = "CLOSED"
            state.failure_count = 0
        elif event_name == "circuit_reopened":
            state.state = "OPEN"
            state.last_failure = _format_event_time(event)
        else:
            circuit_state = event.get("circuit_state")
            if isinstance(circuit_state, str):
                state.state = circuit_state.upper()

        self.circuits[function_name] = state
        self._refresh()

    def _refresh(self) -> None:
        lines = ["Circuit Breakers", "Function | State | Failures | Last Failure"]
        for state in self.circuits.values():
            lines.append(
                f"{state.function_name} | {state.state} | "
                f"{state.failure_count} | {state.last_failure}"
            )
        self.update("\n".join(lines))


class CostPanel(_StaticBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.total_usd = 0.0
        self.budget_usd: float | None = None
        self.budget_state = "ok"
        self.step_costs: list[dict[str, Any]] = []
        self._step_indexes: dict[str, int] = {}
        self._workflow_id: str | None = None
        self._run_id: str | None = None
        self._refresh()

    def update_from_event(self, event: dict[str, Any]) -> None:
        workflow_id = event.get("workflow_id")
        run_id = event.get("run_id")
        if isinstance(workflow_id, str) and isinstance(run_id, str):
            self._sync_run(workflow_id, run_id)

        event_name = event.get("event")
        if event_name == "step_cost_recorded":
            cumulative = event.get("cumulative_usd")
            if isinstance(cumulative, (int, float)):
                self.total_usd = max(self.total_usd, float(cumulative))
            max_budget_usd = event.get("max_budget_usd")
            if isinstance(max_budget_usd, (int, float)):
                self.budget_usd = float(max_budget_usd)
            self._upsert_step_cost(
                {
                    "step_name": str(event.get("step_name", "<unknown>")),
                    "model": str(event.get("model", "")),
                    "input_tokens": int(event.get("input_tokens", 0)),
                    "output_tokens": int(event.get("output_tokens", 0)),
                    "cost_usd": float(event.get("cost_usd", 0.0)),
                    "restored_from_checkpoint": bool(event.get("restored_from_checkpoint", False)),
                }
            )
        elif event_name == "budget_warning":
            self.budget_state = "warning"
            max_budget_usd = event.get("max_budget_usd")
            if isinstance(max_budget_usd, (int, float)):
                self.budget_usd = float(max_budget_usd)
        elif event_name in {"budget_exceeded", "budget_degradation_triggered"}:
            self.budget_state = "exceeded" if event_name == "budget_exceeded" else "degrade"
            current_spend = event.get("current_spend", event.get("spend_usd"))
            if isinstance(current_spend, (int, float)):
                self.total_usd = float(current_spend)
            max_budget = event.get("max_budget", event.get("max_budget_usd"))
            if isinstance(max_budget, (int, float)):
                self.budget_usd = float(max_budget)
        else:
            return

        self._refresh()

    def _sync_run(self, workflow_id: str, run_id: str) -> None:
        if self._workflow_id is None and self._run_id is None:
            self._workflow_id = workflow_id
            self._run_id = run_id
            return

        if self._workflow_id == workflow_id and self._run_id == run_id:
            return

        self.total_usd = 0.0
        self.budget_usd = None
        self.budget_state = "ok"
        self.step_costs = []
        self._step_indexes = {}
        self._workflow_id = workflow_id
        self._run_id = run_id

    def _upsert_step_cost(self, entry: dict[str, Any]) -> None:
        step_name = str(entry["step_name"])
        index = self._step_indexes.get(step_name)
        if index is None:
            self.step_costs.append(entry)
            self._step_indexes[step_name] = len(self.step_costs) - 1
        else:
            existing = self.step_costs[index]
            if (
                bool(entry.get("restored_from_checkpoint"))
                and not bool(existing.get("restored_from_checkpoint"))
            ):
                # Preserve the original live execution record for this run so replayed
                # checkpoint costs do not look like a second billable execution.
                return

            merged = {**existing, **entry}
            self.step_costs[index] = merged

        if len(self.step_costs) > 25:
            self.step_costs = self.step_costs[-25:]
            self._step_indexes = {
                str(step["step_name"]): idx for idx, step in enumerate(self.step_costs)
            }

    def _refresh(self) -> None:
        lines = [
            "Cost Tracker",
            f"Total: ${_format_usd(self.total_usd)}",
            f"Budget: {_render_budget_bar(self.total_usd, self.budget_usd)}",
            f"Budget State: {self.budget_state}",
            "Step | Model | Tokens | Cost | Source",
        ]
        for entry in self.step_costs:
            tokens = entry["input_tokens"] + entry["output_tokens"]
            source = "checkpoint" if entry.get("restored_from_checkpoint") else "live"
            lines.append(
                f"{entry['step_name']} | {entry['model']} | {tokens} | "
                f"${_format_usd(entry['cost_usd'])} | {source}"
            )
        self.update("\n".join(lines))


class EventLogPanel(_StaticBase):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.lines: list[str] = []

    def update_from_event(self, event: dict[str, Any]) -> None:
        level = str(event.get("level", "info")).upper()
        timestamp = _format_event_time(event)
        event_name = str(event.get("event", "event"))
        parts = []
        for key, value in event.items():
            if key in {"event", "level", "timestamp", "_timestamp"}:
                continue
            parts.append(f"{key}={value}")
        line = f"[{timestamp}] [{level}] {event_name} {' '.join(parts)}".rstrip()
        self.lines.append(line)
        self.lines = self.lines[-500:]
        self.update("\n".join(self.lines))
