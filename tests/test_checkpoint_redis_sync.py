from __future__ import annotations

import fakeredis

from tardigrade import RedisCheckpointStore
from tardigrade._serializer import deserialize_result, serialize_result
from tardigrade._types import RedisCheckpointConfig


def _create_store() -> RedisCheckpointStore:
    return RedisCheckpointStore(RedisCheckpointConfig(key_prefix="test-sync"))


def test_redis_checkpoint_store_save_and_load() -> None:
    store = _create_store()
    client = fakeredis.FakeRedis(decode_responses=False)
    store._build_sync_client = lambda: client  # type: ignore[method-assign]
    try:
        store.save("workflow", "step", "run-1", serialize_result({"value": 1}))
        loaded = store.load("workflow", "step", "run-1")
        assert loaded is not None
        assert deserialize_result(loaded) == {"value": 1}
    finally:
        store.close()


def test_redis_checkpoint_store_load_missing_returns_none() -> None:
    store = _create_store()
    client = fakeredis.FakeRedis(decode_responses=False)
    store._build_sync_client = lambda: client  # type: ignore[method-assign]
    try:
        assert store.load("workflow", "missing-step", "run-1") is None
    finally:
        store.close()


def test_redis_checkpoint_store_upsert_overwrites_existing_value() -> None:
    store = _create_store()
    client = fakeredis.FakeRedis(decode_responses=False)
    store._build_sync_client = lambda: client  # type: ignore[method-assign]
    try:
        store.save("workflow", "step", "run-1", serialize_result(1))
        store.save("workflow", "step", "run-1", serialize_result(2))
        loaded = store.load("workflow", "step", "run-1")
        assert loaded is not None
        assert deserialize_result(loaded) == 2
    finally:
        store.close()


def test_redis_checkpoint_store_clear_run_only_removes_that_run() -> None:
    store = _create_store()
    client = fakeredis.FakeRedis(decode_responses=False)
    store._build_sync_client = lambda: client  # type: ignore[method-assign]
    try:
        store.save("workflow", "step", "run-1", serialize_result("a"))
        store.save("workflow", "step", "run-2", serialize_result("b"))
        store.save_metadata("workflow", "step", "run-1", b"meta-a")
        store.save_metadata("workflow", "step", "run-2", b"meta-b")

        store.clear_run("workflow", "run-1")

        assert store.load("workflow", "step", "run-1") is None
        loaded = store.load("workflow", "step", "run-2")
        assert loaded is not None
        assert deserialize_result(loaded) == "b"
        assert store.load_metadata("workflow", "step", "run-1") is None
        assert store.load_metadata("workflow", "step", "run-2") == b"meta-b"
    finally:
        store.close()


def test_redis_checkpoint_store_clear_workflow_removes_everything() -> None:
    store = _create_store()
    client = fakeredis.FakeRedis(decode_responses=False)
    store._build_sync_client = lambda: client  # type: ignore[method-assign]
    try:
        store.save("workflow", "step-1", "run-1", serialize_result("a"))
        store.save("workflow", "step-2", "run-2", serialize_result("b"))
        store.save_metadata("workflow", "step-1", "run-1", b"meta-a")
        store.save_metadata("workflow", "step-2", "run-2", b"meta-b")

        store.clear_workflow("workflow")

        assert store.load("workflow", "step-1", "run-1") is None
        assert store.load("workflow", "step-2", "run-2") is None
        assert store.load_metadata("workflow", "step-1", "run-1") is None
        assert store.load_metadata("workflow", "step-2", "run-2") is None
    finally:
        store.close()


def test_redis_checkpoint_store_metadata_save_load_and_delete() -> None:
    store = _create_store()
    client = fakeredis.FakeRedis(decode_responses=False)
    store._build_sync_client = lambda: client  # type: ignore[method-assign]
    try:
        store.save_metadata("workflow", "step", "run-1", b"meta")
        assert store.load_metadata("workflow", "step", "run-1") == b"meta"

        store.save_metadata("workflow", "step", "run-1", None)
        assert store.load_metadata("workflow", "step", "run-1") is None
    finally:
        store.close()


def test_redis_checkpoint_store_config_and_build_clients() -> None:
    config = RedisCheckpointConfig(
        host="localhost",
        port=6379,
        db=1,
        key_prefix="test-build",
    )
    store = RedisCheckpointStore(config)
    sync_client = store._build_sync_client()
    async_client = store._build_async_client()
    try:
        assert store.config is config
        assert sync_client is not None
        assert async_client is not None
    finally:
        sync_client.close()


def test_redis_checkpoint_store_close_without_prior_usage() -> None:
    store = _create_store()
    store.close()
