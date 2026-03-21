"""Thread-safe event bus for dashboard consumption.

The dashboard consumes the structured events emitted across AgentArmor's
resilience features. Current event vocabulary:

- step_started
- step_completed
- step_retrying
- step_failed_all_retries
- step_restored_from_checkpoint
- step_checkpointed
- circuit_opened
- circuit_fallback
- circuit_closed
- circuit_reopened
- step_cost_recorded
- budget_warning
- budget_exceeded
- budget_degradation_triggered
"""

from __future__ import annotations

import queue
import threading
from typing import Any, ClassVar


class EventBus:
    _instance: ClassVar[EventBus | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, maxsize: int = 10_000) -> None:
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)

    @classmethod
    def get(cls) -> EventBus:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def publish(self, event: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            return

    def poll(self, max_events: int = 100) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for _ in range(max_events):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events
