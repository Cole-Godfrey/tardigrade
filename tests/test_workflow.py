from __future__ import annotations

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from tardigrade import SQLiteCheckpointStore, Workflow, armor
from tardigrade._serializer import deserialize_result


def test_workflow_executes_all_steps_on_first_run(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0, "step_3": 0}

    @armor(name="step_1")
    def step_1() -> int:
        calls["step_1"] += 1
        return 1

    @armor(name="step_2")
    def step_2(value: int) -> int:
        calls["step_2"] += 1
        return value + 1

    @armor(name="step_3")
    def step_3(value: int) -> str:
        calls["step_3"] += 1
        return f"result={value}"

    try:
        with Workflow("pipeline", run_id="run-1", store=store):
            value_1 = step_1()
            value_2 = step_2(value_1)
            value_3 = step_3(value_2)

        assert value_1 == 1
        assert value_2 == 2
        assert value_3 == "result=2"
        assert calls == {"step_1": 1, "step_2": 1, "step_3": 1}
        assert store.db_path.exists()

        checkpoint = store.load("pipeline", "step_2", "run-1")
        assert checkpoint is not None
        assert deserialize_result(checkpoint) == 2
    finally:
        store.close()


def test_workflow_resumes_from_first_missing_checkpoint(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0, "step_3": 0}

    @armor(name="step_1")
    def step_1() -> int:
        calls["step_1"] += 1
        return 1

    @armor(name="step_2")
    def step_2(value: int) -> int:
        calls["step_2"] += 1
        return value + 1

    @armor(name="step_3")
    def step_3(value: int) -> str:
        calls["step_3"] += 1
        if calls["step_3"] == 1:
            raise RuntimeError("step 3 failed")
        return f"done={value}"

    try:
        with pytest.raises(RuntimeError, match="step 3 failed"):
            with Workflow("pipeline", run_id="run-1", store=store):
                value_1 = step_1()
                value_2 = step_2(value_1)
                step_3(value_2)

        assert calls == {"step_1": 1, "step_2": 1, "step_3": 1}

        with Workflow("pipeline", run_id="run-1", store=store):
            value_1 = step_1()
            value_2 = step_2(value_1)
            result = step_3(value_2)

        assert value_1 == 1
        assert value_2 == 2
        assert result == "done=2"
        assert calls == {"step_1": 1, "step_2": 1, "step_3": 2}
    finally:
        store.close()


def test_workflow_logs_checkpoint_restore_events(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0, "step_3": 0}

    @armor(name="step_1")
    def step_1() -> int:
        calls["step_1"] += 1
        return 1

    @armor(name="step_2")
    def step_2(value: int) -> int:
        calls["step_2"] += 1
        return value + 1

    @armor(name="step_3")
    def step_3(value: int) -> str:
        calls["step_3"] += 1
        if calls["step_3"] == 1:
            raise RuntimeError("step 3 failed")
        return f"done={value}"

    try:
        with pytest.raises(RuntimeError, match="step 3 failed"):
            with Workflow("pipeline", run_id="run-1", store=store):
                step_3(step_2(step_1()))

        with capture_logs() as captured_logs:
            with Workflow("pipeline", run_id="run-1", store=store):
                step_3(step_2(step_1()))

        restore_events = [
            entry for entry in captured_logs if entry["event"] == "step_restored_from_checkpoint"
        ]

        assert [entry["step_name"] for entry in restore_events] == ["step_1", "step_2"]
        assert all(entry["workflow_id"] == "pipeline" for entry in restore_events)
        assert all(entry["run_id"] == "run-1" for entry in restore_events)
    finally:
        store.close()


def test_workflow_absent_means_no_checkpoint_file_is_created(tmp_path: Path) -> None:
    db_path = tmp_path / "checkpoints.db"
    store = SQLiteCheckpointStore(db_path)

    @armor(name="step")
    def step() -> str:
        return "ok"

    try:
        assert step() == "ok"
        assert not db_path.exists()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_async_workflow_resumes_from_first_missing_checkpoint(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0, "step_3": 0}

    @armor(name="step_1")
    async def step_1() -> int:
        calls["step_1"] += 1
        return 1

    @armor(name="step_2")
    async def step_2(value: int) -> int:
        calls["step_2"] += 1
        return value + 1

    @armor(name="step_3")
    async def step_3(value: int) -> str:
        calls["step_3"] += 1
        if calls["step_3"] == 1:
            raise RuntimeError("step 3 failed")
        return f"done={value}"

    try:
        with pytest.raises(RuntimeError, match="step 3 failed"):
            async with Workflow("pipeline", run_id="run-1", store=store):
                value_1 = await step_1()
                value_2 = await step_2(value_1)
                await step_3(value_2)

        async with Workflow("pipeline", run_id="run-1", store=store):
            value_1 = await step_1()
            value_2 = await step_2(value_1)
            result = await step_3(value_2)

        assert value_1 == 1
        assert value_2 == 2
        assert result == "done=2"
        assert calls == {"step_1": 1, "step_2": 1, "step_3": 2}
    finally:
        await store.aclose()


def test_workflow_different_run_ids_execute_fresh_steps(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step_1": 0, "step_2": 0}

    @armor(name="step_1")
    def step_1() -> int:
        calls["step_1"] += 1
        return 1

    @armor(name="step_2")
    def step_2(value: int) -> int:
        calls["step_2"] += 1
        return value + 1

    try:
        with Workflow("pipeline", run_id="run-a", store=store):
            assert step_2(step_1()) == 2

        with Workflow("pipeline", run_id="run-b", store=store):
            assert step_2(step_1()) == 2

        assert calls == {"step_1": 2, "step_2": 2}
    finally:
        store.close()


def test_workflow_clear_forces_reexecution(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")
    calls = {"step": 0}

    @armor(name="step")
    def step() -> int:
        calls["step"] += 1
        return calls["step"]

    workflow = Workflow("pipeline", run_id="run-1", store=store)

    try:
        with workflow:
            assert step() == 1

        workflow.clear()

        with Workflow("pipeline", run_id="run-1", store=store):
            assert step() == 2

        assert calls["step"] == 2
    finally:
        store.close()


def test_workflow_uses_decorator_name_or_function_qualname_for_keys(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.db")

    @armor(name="custom")
    def custom_step() -> str:
        return "custom"

    @armor
    def default_named_step() -> str:
        return "default"

    try:
        with Workflow("pipeline", run_id="run-1", store=store):
            assert custom_step() == "custom"
            assert default_named_step() == "default"

        assert store.load("pipeline", "custom", "run-1") is not None
        assert store.load("pipeline", default_named_step.__qualname__, "run-1") is not None
    finally:
        store.close()
