from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from tardigrade._event_bus import EventBus
from tardigrade.dashboard._widgets import (
    CircuitBreakerPanel,
    CostPanel,
    EventLogPanel,
    WorkflowPanel,
)

if TYPE_CHECKING:
    class _AppBase:
        CSS_PATH: str
        TITLE: str

        def run(self) -> None:
            ...

        def set_interval(self, interval: float, callback: Any) -> None:
            ...

        def query_one(self, selector: str, expect_type: type[Any]) -> Any:
            ...

    ComposeResult = Iterable[object]

    class _Header:
        pass

    class _Footer:
        pass
else:
    from textual.app import ComposeResult, App as _AppBase  # type: ignore[import-not-found]  # noqa: I001
    from textual.widgets import Footer as _Footer, Header as _Header  # type: ignore[import-not-found]


class TardigradeDashboard(_AppBase):
    CSS_PATH = "_styles.tcss"
    TITLE = "Tardigrade Dashboard"

    def compose(self) -> ComposeResult:
        yield _Header()
        yield WorkflowPanel(id="workflow-panel")
        yield CircuitBreakerPanel(id="circuit-panel")
        yield CostPanel(id="cost-panel")
        yield EventLogPanel(id="event-log")
        yield _Footer()

    def on_mount(self) -> None:
        self.set_interval(0.1, self._poll_events)

    def _poll_events(self) -> None:
        events = EventBus.get().poll(max_events=50)
        if not events:
            return

        workflow_panel = self.query_one("#workflow-panel", WorkflowPanel)
        circuit_panel = self.query_one("#circuit-panel", CircuitBreakerPanel)
        cost_panel = self.query_one("#cost-panel", CostPanel)
        event_log = self.query_one("#event-log", EventLogPanel)

        for event in events:
            workflow_panel.update_from_event(event)
            circuit_panel.update_from_event(event)
            cost_panel.update_from_event(event)
            event_log.update_from_event(event)
