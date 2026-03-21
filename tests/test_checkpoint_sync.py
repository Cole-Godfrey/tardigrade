from __future__ import annotations

from pathlib import Path

import pytest

from agentarmor import SQLiteCheckpointStore
from agentarmor._serializer import deserialize_result, serialize_result
from agentarmor._types import AgentArmorSerializationError


def test_checkpoint_store_save_and_load(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    try:
        store.save("workflow", "step", "run-1", serialize_result({"value": 1}))

        loaded = store.load("workflow", "step", "run-1")

        assert loaded is not None
        assert deserialize_result(loaded) == {"value": 1}
    finally:
        store.close()


def test_checkpoint_store_load_missing_returns_none(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    try:
        assert store.load("workflow", "missing-step", "run-1") is None
    finally:
        store.close()


def test_checkpoint_store_upsert_overwrites_existing_value(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    try:
        store.save("workflow", "step", "run-1", serialize_result(1))
        store.save("workflow", "step", "run-1", serialize_result(2))

        loaded = store.load("workflow", "step", "run-1")

        assert loaded is not None
        assert deserialize_result(loaded) == 2
    finally:
        store.close()


def test_checkpoint_store_clear_run_only_removes_that_run(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    try:
        store.save("workflow", "step", "run-1", serialize_result("a"))
        store.save("workflow", "step", "run-2", serialize_result("b"))

        store.clear_run("workflow", "run-1")

        assert store.load("workflow", "step", "run-1") is None
        loaded = store.load("workflow", "step", "run-2")
        assert loaded is not None
        assert deserialize_result(loaded) == "b"
    finally:
        store.close()


def test_checkpoint_store_clear_workflow_removes_everything(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    try:
        store.save("workflow", "step-1", "run-1", serialize_result("a"))
        store.save("workflow", "step-2", "run-2", serialize_result("b"))

        store.clear_workflow("workflow")

        assert store.load("workflow", "step-1", "run-1") is None
        assert store.load("workflow", "step-2", "run-2") is None
    finally:
        store.close()


def test_checkpoint_serialization_error_is_wrapped() -> None:
    with pytest.raises(AgentArmorSerializationError):
        serialize_result(lambda: None)
