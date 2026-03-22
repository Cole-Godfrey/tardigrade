from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Protocol, cast

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    workflow_id TEXT NOT NULL,
    step_name   TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    result_blob BLOB NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (workflow_id, step_name, run_id)
)
"""

_METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoint_metadata (
    workflow_id   TEXT NOT NULL,
    step_name     TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    metadata_blob BLOB NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (workflow_id, step_name, run_id)
)
"""


class CheckpointStore(Protocol):
    def save(self, workflow_id: str, step_name: str, run_id: str, result: bytes) -> None: ...

    def load(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None: ...

    def clear_run(self, workflow_id: str, run_id: str) -> None: ...

    def clear_workflow(self, workflow_id: str) -> None: ...

    async def asave(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        result: bytes,
    ) -> None: ...

    async def aload(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None: ...

    async def aclear_run(self, workflow_id: str, run_id: str) -> None: ...

    async def aclear_workflow(self, workflow_id: str) -> None: ...


class CheckpointMetadataStore(Protocol):
    def save_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        metadata: bytes | None,
    ) -> None: ...

    def load_metadata(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None: ...

    async def asave_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        metadata: bytes | None,
    ) -> None: ...

    async def aload_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
    ) -> bytes | None: ...


class SQLiteCheckpointStore:
    def __init__(self, db_path: str | Path = ".tardigrade/checkpoints.db") -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._sync_connection: sqlite3.Connection | None = None
        self._async_connection: aiosqlite.Connection | None = None
        self._async_init_lock = asyncio.Lock()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _ensure_directory(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_sync_connection(self) -> sqlite3.Connection:
        if self._sync_connection is None:
            self._ensure_directory()
            connection = sqlite3.connect(str(self._db_path), check_same_thread=False)
            connection.execute(_SCHEMA)
            connection.execute(_METADATA_SCHEMA)
            connection.commit()
            self._sync_connection = connection
        return self._sync_connection

    async def _ensure_async_connection(self) -> aiosqlite.Connection:
        async with self._async_init_lock:
            if self._async_connection is None:
                self._ensure_directory()
                connection = await aiosqlite.connect(str(self._db_path))
                await connection.execute(_SCHEMA)
                await connection.execute(_METADATA_SCHEMA)
                await connection.commit()
                self._async_connection = connection

        connection = self._async_connection
        if connection is None:
            msg = "Async checkpoint connection failed to initialize"
            raise RuntimeError(msg)
        return connection

    def save(self, workflow_id: str, step_name: str, run_id: str, result: bytes) -> None:
        with self._lock:
            connection = self._ensure_sync_connection()
            connection.execute(
                """
                INSERT OR REPLACE INTO checkpoints (
                    workflow_id,
                    step_name,
                    run_id,
                    result_blob
                ) VALUES (?, ?, ?, ?)
                """,
                (workflow_id, step_name, run_id, result),
            )
            connection.commit()

    def load(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None:
        with self._lock:
            connection = self._ensure_sync_connection()
            cursor = connection.execute(
                """
                SELECT result_blob
                FROM checkpoints
                WHERE workflow_id = ? AND step_name = ? AND run_id = ?
                """,
                (workflow_id, step_name, run_id),
            )
            row = cursor.fetchone()
            cursor.close()

        if row is None:
            return None
        return cast(bytes, row[0])

    def clear_run(self, workflow_id: str, run_id: str) -> None:
        with self._lock:
            connection = self._ensure_sync_connection()
            connection.execute(
                """
                DELETE FROM checkpoints
                WHERE workflow_id = ? AND run_id = ?
                """,
                (workflow_id, run_id),
            )
            connection.execute(
                """
                DELETE FROM checkpoint_metadata
                WHERE workflow_id = ? AND run_id = ?
                """,
                (workflow_id, run_id),
            )
            connection.commit()

    def clear_workflow(self, workflow_id: str) -> None:
        with self._lock:
            connection = self._ensure_sync_connection()
            connection.execute(
                """
                DELETE FROM checkpoints
                WHERE workflow_id = ?
                """,
                (workflow_id,),
            )
            connection.execute(
                """
                DELETE FROM checkpoint_metadata
                WHERE workflow_id = ?
                """,
                (workflow_id,),
            )
            connection.commit()

    def save_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        metadata: bytes | None,
    ) -> None:
        with self._lock:
            connection = self._ensure_sync_connection()
            if metadata is None:
                connection.execute(
                    """
                    DELETE FROM checkpoint_metadata
                    WHERE workflow_id = ? AND step_name = ? AND run_id = ?
                    """,
                    (workflow_id, step_name, run_id),
                )
            else:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO checkpoint_metadata (
                        workflow_id,
                        step_name,
                        run_id,
                        metadata_blob
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (workflow_id, step_name, run_id, metadata),
                )
            connection.commit()

    def load_metadata(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None:
        with self._lock:
            connection = self._ensure_sync_connection()
            cursor = connection.execute(
                """
                SELECT metadata_blob
                FROM checkpoint_metadata
                WHERE workflow_id = ? AND step_name = ? AND run_id = ?
                """,
                (workflow_id, step_name, run_id),
            )
            row = cursor.fetchone()
            cursor.close()

        if row is None:
            return None
        return cast(bytes, row[0])

    async def asave(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        result: bytes,
    ) -> None:
        connection = await self._ensure_async_connection()
        await connection.execute(
            """
            INSERT OR REPLACE INTO checkpoints (
                workflow_id,
                step_name,
                run_id,
                result_blob
            ) VALUES (?, ?, ?, ?)
            """,
            (workflow_id, step_name, run_id, result),
        )
        await connection.commit()

    async def aload(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None:
        connection = await self._ensure_async_connection()
        cursor = await connection.execute(
            """
            SELECT result_blob
            FROM checkpoints
            WHERE workflow_id = ? AND step_name = ? AND run_id = ?
            """,
            (workflow_id, step_name, run_id),
        )
        row = await cursor.fetchone()
        await cursor.close()

        if row is None:
            return None
        return cast(bytes, row[0])

    async def aclear_run(self, workflow_id: str, run_id: str) -> None:
        connection = await self._ensure_async_connection()
        await connection.execute(
            """
            DELETE FROM checkpoints
            WHERE workflow_id = ? AND run_id = ?
            """,
            (workflow_id, run_id),
        )
        await connection.execute(
            """
            DELETE FROM checkpoint_metadata
            WHERE workflow_id = ? AND run_id = ?
            """,
            (workflow_id, run_id),
        )
        await connection.commit()

    async def asave_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        metadata: bytes | None,
    ) -> None:
        connection = await self._ensure_async_connection()
        if metadata is None:
            await connection.execute(
                """
                DELETE FROM checkpoint_metadata
                WHERE workflow_id = ? AND step_name = ? AND run_id = ?
                """,
                (workflow_id, step_name, run_id),
            )
        else:
            await connection.execute(
                """
                INSERT OR REPLACE INTO checkpoint_metadata (
                    workflow_id,
                    step_name,
                    run_id,
                    metadata_blob
                ) VALUES (?, ?, ?, ?)
                """,
                (workflow_id, step_name, run_id, metadata),
            )
        await connection.commit()

    async def aload_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
    ) -> bytes | None:
        connection = await self._ensure_async_connection()
        cursor = await connection.execute(
            """
            SELECT metadata_blob
            FROM checkpoint_metadata
            WHERE workflow_id = ? AND step_name = ? AND run_id = ?
            """,
            (workflow_id, step_name, run_id),
        )
        row = await cursor.fetchone()
        await cursor.close()

        if row is None:
            return None
        return cast(bytes, row[0])

    async def aclear_workflow(self, workflow_id: str) -> None:
        connection = await self._ensure_async_connection()
        await connection.execute(
            """
            DELETE FROM checkpoints
            WHERE workflow_id = ?
            """,
            (workflow_id,),
        )
        await connection.execute(
            """
            DELETE FROM checkpoint_metadata
            WHERE workflow_id = ?
            """,
            (workflow_id,),
        )
        await connection.commit()

    def close(self) -> None:
        with self._lock:
            connection = self._sync_connection
            self._sync_connection = None
            if connection is not None:
                connection.close()

    async def aclose(self) -> None:
        async with self._async_init_lock:
            connection = self._async_connection
            self._async_connection = None
            if connection is not None:
                await connection.close()
