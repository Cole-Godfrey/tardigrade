from __future__ import annotations

import fakeredis.aioredis
import pytest

from tardigrade import RedisCheckpointStore
from tardigrade._serializer import deserialize_result, serialize_result
from tardigrade._types import RedisCheckpointConfig


def _create_store() -> RedisCheckpointStore:
    return RedisCheckpointStore(RedisCheckpointConfig(key_prefix="test-async"))


@pytest.mark.asyncio
async def test_async_redis_checkpoint_store_save_and_load() -> None:
    store = _create_store()
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    store._build_async_client = lambda: client  # type: ignore[method-assign]
    try:
        await store.asave("workflow", "step", "run-1", serialize_result({"value": 1}))
        loaded = await store.aload("workflow", "step", "run-1")
        assert loaded is not None
        assert deserialize_result(loaded) == {"value": 1}
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_redis_checkpoint_store_load_missing_returns_none() -> None:
    store = _create_store()
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    store._build_async_client = lambda: client  # type: ignore[method-assign]
    try:
        assert await store.aload("workflow", "missing-step", "run-1") is None
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_redis_checkpoint_store_upsert_overwrites_existing_value() -> None:
    store = _create_store()
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    store._build_async_client = lambda: client  # type: ignore[method-assign]
    try:
        await store.asave("workflow", "step", "run-1", serialize_result(1))
        await store.asave("workflow", "step", "run-1", serialize_result(2))
        loaded = await store.aload("workflow", "step", "run-1")
        assert loaded is not None
        assert deserialize_result(loaded) == 2
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_redis_checkpoint_store_clear_run_only_removes_that_run() -> None:
    store = _create_store()
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    store._build_async_client = lambda: client  # type: ignore[method-assign]
    try:
        await store.asave("workflow", "step", "run-1", serialize_result("a"))
        await store.asave("workflow", "step", "run-2", serialize_result("b"))
        await store.asave_metadata("workflow", "step", "run-1", b"meta-a")
        await store.asave_metadata("workflow", "step", "run-2", b"meta-b")

        await store.aclear_run("workflow", "run-1")

        assert await store.aload("workflow", "step", "run-1") is None
        loaded = await store.aload("workflow", "step", "run-2")
        assert loaded is not None
        assert deserialize_result(loaded) == "b"
        assert await store.aload_metadata("workflow", "step", "run-1") is None
        assert await store.aload_metadata("workflow", "step", "run-2") == b"meta-b"
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_redis_checkpoint_store_clear_workflow_removes_everything() -> None:
    store = _create_store()
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    store._build_async_client = lambda: client  # type: ignore[method-assign]
    try:
        await store.asave("workflow", "step-1", "run-1", serialize_result("a"))
        await store.asave("workflow", "step-2", "run-2", serialize_result("b"))
        await store.asave_metadata("workflow", "step-1", "run-1", b"meta-a")
        await store.asave_metadata("workflow", "step-2", "run-2", b"meta-b")

        await store.aclear_workflow("workflow")

        assert await store.aload("workflow", "step-1", "run-1") is None
        assert await store.aload("workflow", "step-2", "run-2") is None
        assert await store.aload_metadata("workflow", "step-1", "run-1") is None
        assert await store.aload_metadata("workflow", "step-2", "run-2") is None
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_redis_checkpoint_store_metadata_save_load_and_delete() -> None:
    store = _create_store()
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    store._build_async_client = lambda: client  # type: ignore[method-assign]
    try:
        await store.asave_metadata("workflow", "step", "run-1", b"meta")
        assert await store.aload_metadata("workflow", "step", "run-1") == b"meta"

        await store.asave_metadata("workflow", "step", "run-1", None)
        assert await store.aload_metadata("workflow", "step", "run-1") is None
    finally:
        await store.aclose()


@pytest.mark.asyncio
async def test_async_redis_checkpoint_store_close_without_prior_usage() -> None:
    store = _create_store()
    await store.aclose()
