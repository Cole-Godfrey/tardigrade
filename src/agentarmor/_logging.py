from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime
from typing import Any, TextIO, cast

import structlog

from ._event_bus import EventBus


class _NullWriter(io.TextIOBase):
    def write(self, s: str) -> int:
        return len(s)


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return repr(value)


def _json_serializer(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("default", _json_default)
    return json.dumps(obj, **kwargs)


def dashboard_processor(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    del logger, method_name
    EventBus.get().publish({**event_dict, "_timestamp": time.time()})
    return event_dict


def configure_logging(enable_dashboard: bool = False) -> None:
    structlog.reset_defaults()

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if enable_dashboard:
        processors.append(dashboard_processor)
    processors.append(structlog.processors.JSONRenderer(serializer=_json_serializer))

    logger_factory: Any
    if enable_dashboard:
        logger_factory = structlog.WriteLoggerFactory(file=cast(TextIO, _NullWriter()))
    else:
        logger_factory = structlog.PrintLoggerFactory()

    structlog.configure(
        processors=processors,
        logger_factory=logger_factory,
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        cache_logger_on_first_use=False,
    )
