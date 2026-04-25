from __future__ import annotations

import asyncio
import sqlite3
import threading
from inspect import isawaitable
from pathlib import Path
from typing import Protocol, cast

import aiosqlite
from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from ._types import RedisCheckpointConfig

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


class RedisCheckpointStore:
    def __init__(self, config: RedisCheckpointConfig | None = None) -> None:
        self._config = RedisCheckpointConfig() if config is None else config
        self._lock = threading.Lock()
        self._sync_client: Redis | None = None
        self._async_client: AsyncRedis | None = None
        self._async_init_lock = asyncio.Lock()

    @property
    def config(self) -> RedisCheckpointConfig:
        return self._config

    def _build_sync_client(self) -> Redis:
        return Redis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            username=self._config.username,
            password=self._config.password,
            socket_timeout=self._config.socket_timeout,
            socket_connect_timeout=self._config.socket_connect_timeout,
            decode_responses=self._config.decode_responses,
        )

    def _build_async_client(self) -> AsyncRedis:
        return AsyncRedis(
            host=self._config.host,
            port=self._config.port,
            db=self._config.db,
            username=self._config.username,
            password=self._config.password,
            socket_timeout=self._config.socket_timeout,
            socket_connect_timeout=self._config.socket_connect_timeout,
            decode_responses=self._config.decode_responses,
        )

    def _ensure_sync_client(self) -> Redis:
        if self._sync_client is None:
            client = self._build_sync_client()
            client.ping()
            self._sync_client = client
        return self._sync_client

    async def _ensure_async_client(self) -> AsyncRedis:
        async with self._async_init_lock:
            if self._async_client is None:
                client = self._build_async_client()
                ping_result = client.ping()
                if isawaitable(ping_result):
                    await ping_result
                self._async_client = client

        client = self._async_client
        if client is None:
            msg = "Async checkpoint Redis client failed to initialize"
            raise RuntimeError(msg)
        return client

    def _result_key(self, workflow_id: str, step_name: str, run_id: str) -> str:
        return f"{self._config.key_prefix}:checkpoint:{workflow_id}:{step_name}:{run_id}"

    def _metadata_key(self, workflow_id: str, step_name: str, run_id: str) -> str:
        return f"{self._config.key_prefix}:metadata:{workflow_id}:{step_name}:{run_id}"

    def _run_patterns(self, workflow_id: str, run_id: str) -> tuple[str, str]:
        return (
            f"{self._config.key_prefix}:checkpoint:{workflow_id}:*:{run_id}",
            f"{self._config.key_prefix}:metadata:{workflow_id}:*:{run_id}",
        )

    def _workflow_patterns(self, workflow_id: str) -> tuple[str, str]:
        return (
            f"{self._config.key_prefix}:checkpoint:{workflow_id}:*",
            f"{self._config.key_prefix}:metadata:{workflow_id}:*",
        )

    def _delete_by_patterns(self, client: Redis, patterns: tuple[str, ...]) -> None:
        for pattern in patterns:
            keys = list(client.scan_iter(match=pattern))
            if keys:
                client.delete(*keys)

    async def _adelete_by_patterns(
        self,
        client: AsyncRedis,
        patterns: tuple[str, ...],
    ) -> None:
        for pattern in patterns:
            keys = [key async for key in client.scan_iter(match=pattern)]
            if keys:
                await client.delete(*keys)

    def save(self, workflow_id: str, step_name: str, run_id: str, result: bytes) -> None:
        with self._lock:
            client = self._ensure_sync_client()
            client.set(self._result_key(workflow_id, step_name, run_id), result)

    def load(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None:
        with self._lock:
            client = self._ensure_sync_client()
            payload = client.get(self._result_key(workflow_id, step_name, run_id))
        if payload is None:
            return None
        return cast(bytes, payload)

    def clear_run(self, workflow_id: str, run_id: str) -> None:
        with self._lock:
            client = self._ensure_sync_client()
            self._delete_by_patterns(client, self._run_patterns(workflow_id, run_id))

    def clear_workflow(self, workflow_id: str) -> None:
        with self._lock:
            client = self._ensure_sync_client()
            self._delete_by_patterns(client, self._workflow_patterns(workflow_id))

    def save_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        metadata: bytes | None,
    ) -> None:
        with self._lock:
            client = self._ensure_sync_client()
            key = self._metadata_key(workflow_id, step_name, run_id)
            if metadata is None:
                client.delete(key)
            else:
                client.set(key, metadata)

    def load_metadata(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None:
        with self._lock:
            client = self._ensure_sync_client()
            payload = client.get(self._metadata_key(workflow_id, step_name, run_id))
        if payload is None:
            return None
        return cast(bytes, payload)

    async def asave(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        result: bytes,
    ) -> None:
        client = await self._ensure_async_client()
        await client.set(self._result_key(workflow_id, step_name, run_id), result)

    async def aload(self, workflow_id: str, step_name: str, run_id: str) -> bytes | None:
        client = await self._ensure_async_client()
        payload = await client.get(self._result_key(workflow_id, step_name, run_id))
        if payload is None:
            return None
        return cast(bytes, payload)

    async def aclear_run(self, workflow_id: str, run_id: str) -> None:
        client = await self._ensure_async_client()
        await self._adelete_by_patterns(client, self._run_patterns(workflow_id, run_id))

    async def asave_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
        metadata: bytes | None,
    ) -> None:
        client = await self._ensure_async_client()
        key = self._metadata_key(workflow_id, step_name, run_id)
        if metadata is None:
            await client.delete(key)
        else:
            await client.set(key, metadata)

    async def aload_metadata(
        self,
        workflow_id: str,
        step_name: str,
        run_id: str,
    ) -> bytes | None:
        client = await self._ensure_async_client()
        payload = await client.get(self._metadata_key(workflow_id, step_name, run_id))
        if payload is None:
            return None
        return cast(bytes, payload)

    async def aclear_workflow(self, workflow_id: str) -> None:
        client = await self._ensure_async_client()
        await self._adelete_by_patterns(client, self._workflow_patterns(workflow_id))

    def close(self) -> None:
        with self._lock:
            client = self._sync_client
            self._sync_client = None
            if client is not None:
                client.close()

    async def aclose(self) -> None:
        async with self._async_init_lock:
            client = self._async_client
            self._async_client = None
            if client is not None:
                await client.aclose()
