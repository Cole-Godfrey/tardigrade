from __future__ import annotations

from pathlib import Path

import pytest

from tardigrade import SQLiteCheckpointStore
from tardigrade._serializer import deserialize_result, serialize_result


@pytest.mark.asyncio
async def test_async_checkpoint_store_save_and_load(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    try:
        await store.asave("workflow", "step", "run-1", serialize_result({"value": 1}))

        loaded = await store.aload("workflow", "step", "run-1")

        assert loaded is not None
        assert deserialize_result(loaded) == {"value": 1}
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_checkpoint_store_load_missing_returns_none(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    try:
        assert await store.aload("workflow", "missing-step", "run-1") is None
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_checkpoint_store_upsert_overwrites_existing_value(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    try:
        await store.asave("workflow", "step", "run-1", serialize_result(1))
        await store.asave("workflow", "step", "run-1", serialize_result(2))

        loaded = await store.aload("workflow", "step", "run-1")

        assert loaded is not None
        assert deserialize_result(loaded) == 2
    finally:
        await store.aclose()
