from __future__ import annotations

import threading
import time

import structlog

from ._logging import configure_logging
from ._types import CircuitBreakerConfig, CircuitState

configure_logging()
_logger = structlog.get_logger("tardigrade")


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: float | None = None
        self._opened_at: float | None = None
        self._function_name = "<unbound>"
        self._lock = threading.Lock()

    @property
    def config(self) -> CircuitBreakerConfig:
        return self._config

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def bind(self, function_name: str) -> None:
        with self._lock:
            self._function_name = function_name

    def can_execute(self) -> bool:
        now = time.monotonic()

        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True

            if self._state is CircuitState.OPEN:
                if self._last_failure_time is None:
                    self._transition_to_half_open_locked()
                    self._half_open_calls = 1
                    return True

                elapsed = now - self._last_failure_time
                if elapsed >= self._config.recovery_timeout:
                    self._transition_to_half_open_locked()
                    self._half_open_calls = 1
                    return True
                return False

            if self._half_open_calls < self._config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        log_success_count: int | None = None
        recovery_duration_ms = 0.0

        with self._lock:
            if self._state is CircuitState.CLOSED:
                self._failure_count = 0
                return

            if self._state is CircuitState.OPEN:
                self._reset_locked()
                return

            self._half_open_calls = max(0, self._half_open_calls - 1)
            self._success_count += 1
            if self._success_count >= self._config.success_threshold:
                log_success_count = self._success_count
                if self._opened_at is not None:
                    recovery_duration_ms = (time.monotonic() - self._opened_at) * 1000
                self._reset_locked()

        if log_success_count is not None:
            _logger.info(
                "circuit_closed",
                function_name=self._function_name,
                success_count=log_success_count,
                recovery_duration_ms=recovery_duration_ms,
            )

    def record_failure(self) -> None:
        log_event: str | None = None
        failure_count: int | None = None
        threshold: int | None = None
        now = time.monotonic()

        with self._lock:
            if self._state is CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._success_count = 0
                    self._half_open_calls = 0
                    self._last_failure_time = now
                    self._opened_at = now
                    log_event = "circuit_opened"
                    failure_count = self._failure_count
                    threshold = self._config.failure_threshold
            elif self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._failure_count = self._config.failure_threshold
                self._success_count = 0
                self._half_open_calls = 0
                self._last_failure_time = now
                self._opened_at = now
                log_event = "circuit_reopened"
            else:
                self._last_failure_time = now
                if self._opened_at is None:
                    self._opened_at = now

        if log_event == "circuit_opened":
            _logger.info(
                "circuit_opened",
                function_name=self._function_name,
                failure_count=failure_count,
                threshold=threshold,
            )
        elif log_event == "circuit_reopened":
            _logger.info(
                "circuit_reopened",
                function_name=self._function_name,
            )

    def release_probe(self) -> None:
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_calls = max(0, self._half_open_calls - 1)

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def _transition_to_half_open_locked(self) -> None:
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0
        self._half_open_calls = 0

    def _reset_locked(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None
        self._opened_at = None
