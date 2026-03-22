"""Checkpoint serialization for local, same-process persistence.

Tardigrade uses pickle because checkpoints are written and read by the same
application. A JSON serializer can be introduced later for safer or more
portable checkpoint formats.
"""

from __future__ import annotations

import pickle
from typing import Any

from ._types import TardigradeSerializationError


def serialize_result(value: Any, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
    try:
        return pickle.dumps(value, protocol=protocol)
    except Exception as exc:
        msg = f"Failed to serialize checkpoint result for type {type(value).__name__}"
        raise TardigradeSerializationError(msg) from exc


def deserialize_result(data: bytes) -> Any:
    try:
        return pickle.loads(data)
    except Exception as exc:
        msg = "Failed to deserialize checkpoint result"
        raise TardigradeSerializationError(msg) from exc
