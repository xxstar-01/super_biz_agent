"""
L2 会话记忆 - SqliteSaver

基于 BaseCheckpointSaver 的 SQLite 持久化实现,
替代 langgraph 默认的 MemorySaver (InMemorySaver)。

参考 langgraph 官方 SqliteSaver 设计, 适配当前版本 API。
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

logger = logging.getLogger(__name__)

# SQLite schema
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id       TEXT NOT NULL,
    checkpoint_ns   TEXT NOT NULL DEFAULT '',
    checkpoint_id   TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    checkpoint_type TEXT NOT NULL,
    checkpoint_data BLOB NOT NULL,
    metadata_type   TEXT NOT NULL,
    metadata_data   BLOB NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
    ON checkpoints(thread_id, checkpoint_ns);

CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id       TEXT NOT NULL,
    checkpoint_ns   TEXT NOT NULL DEFAULT '',
    checkpoint_id   TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    idx             INTEGER NOT NULL,
    channel         TEXT NOT NULL,
    value_type      TEXT NOT NULL,
    value_data      BLOB NOT NULL,
    task_path       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

CREATE TABLE IF NOT EXISTS checkpoint_blobs (
    thread_id       TEXT NOT NULL,
    checkpoint_ns   TEXT NOT NULL DEFAULT '',
    channel         TEXT NOT NULL,
    version         TEXT NOT NULL,
    blob_type       TEXT NOT NULL,
    blob_data       BLOB NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- 会话元数据: 记录最后活跃时间, 用于自动清理
CREATE TABLE IF NOT EXISTS session_meta (
    thread_id       TEXT PRIMARY KEY,
    last_active_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class SqliteSaver(BaseCheckpointSaver[str]):
    """SQLite 持久化 CheckpointSaver

    替代 MemorySaver, 实现 L2 会话记忆持久化。
    进程重启后会话状态不丢失。

    使用示例:
        saver = SqliteSaver(db_path="data/memory.db")
        graph = workflow.compile(checkpointer=saver)
    """

    def __init__(
        self,
        db_path: str = "data/memory.db",
        *,
        serde: JsonPlusSerializer | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 同步连接 (用于同步方法, 每个线程独立)
        self._local = threading.local()

        # 初始化数据库表
        self._init_sync_db()

        logger.info(f"SqliteSaver 初始化完成, db={self.db_path}")

    # ------------------------------------------------------------------
    # 同步 SQLite 连接
    # ------------------------------------------------------------------

    def _sync_conn(self) -> sqlite3.Connection:
        """获取当前线程的同步连接 (thread-local)"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return self._local.conn

    def _init_sync_db(self) -> None:
        """初始化同步数据库表"""
        conn = self._sync_conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    # ------------------------------------------------------------------
    # Sync: get_tuple / get
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """同步获取 checkpoint tuple"""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        conn = self._sync_conn()

        if checkpoint_id := get_checkpoint_id(config):
            # 获取指定 checkpoint
            row = conn.execute(
                """SELECT * FROM checkpoints
                   WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?""",
                (thread_id, checkpoint_ns, checkpoint_id),
            ).fetchone()
        else:
            # 获取最新 checkpoint
            row = conn.execute(
                """SELECT * FROM checkpoints
                   WHERE thread_id=? AND checkpoint_ns=?
                   ORDER BY checkpoint_id DESC LIMIT 1""",
                (thread_id, checkpoint_ns),
            ).fetchone()

        if not row:
            return None

        return self._row_to_tuple(thread_id, checkpoint_ns, row, conn, config)

    def _row_to_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        row: sqlite3.Row,
        conn: sqlite3.Connection,
        config: RunnableConfig,
    ) -> CheckpointTuple:
        """将数据库行转为 CheckpointTuple"""
        checkpoint_id = row["checkpoint_id"]
        parent_checkpoint_id = row["parent_checkpoint_id"]

        # 反序列化 checkpoint
        checkpoint = self.serde.loads_typed(
            (row["checkpoint_type"], bytes(row["checkpoint_data"]))
        )
        metadata = self.serde.loads_typed(
            (row["metadata_type"], bytes(row["metadata_data"]))
        )

        # 按 channel_versions 加载 blobs
        channel_versions = checkpoint.get("channel_versions", {})
        channel_values: dict[str, Any] = {}
        for ch, ver in channel_versions.items():
            br = conn.execute(
                """SELECT blob_type, blob_data FROM checkpoint_blobs
                   WHERE thread_id=? AND checkpoint_ns=? AND channel=? AND version=?""",
                (thread_id, checkpoint_ns, ch, ver),
            ).fetchone()
            if br:
                blob_type = br["blob_type"]
                blob_data = bytes(br["blob_data"])
                if blob_type != "empty":
                    channel_values[ch] = self.serde.loads_typed(
                        (blob_type, blob_data)
                    )

        # 加载 pending writes
        write_rows = conn.execute(
            """SELECT task_id, channel, value_type, value_data
               FROM checkpoint_writes
               WHERE thread_id=? AND checkpoint_ns=? AND checkpoint_id=?
               ORDER BY task_id, idx""",
            (thread_id, checkpoint_ns, checkpoint_id),
        ).fetchall()

        pending_writes = []
        for wr in write_rows:
            try:
                value = self.serde.loads_typed(
                    (wr["value_type"], bytes(wr["value_data"]))
                )
                pending_writes.append((wr["task_id"], wr["channel"], value))
            except Exception:
                pass

        # 构建返回的 config
        result_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        # 构建 parent_config
        parent_config: RunnableConfig | None = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        return CheckpointTuple(
            config=result_config,
            checkpoint={
                **checkpoint,
                "channel_values": channel_values,
            },
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes if pending_writes else None,
        )

    # ------------------------------------------------------------------
    # Sync: list
    # ------------------------------------------------------------------

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """同步列出 checkpoints"""
        conn = self._sync_conn()

        if config:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        else:
            # 列出所有线程
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            ).fetchall()
            thread_ids = [r["thread_id"] for r in rows]
            for tid in thread_ids:
                for ns_row in conn.execute(
                    "SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id=?",
                    (tid,),
                ).fetchall():
                    yield from self._list_checkpoints(
                        tid, ns_row["checkpoint_ns"], conn, filter, before, limit
                    )
            return

        yield from self._list_checkpoints(
            thread_id, checkpoint_ns, conn, filter, before, limit
        )

    def _list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str,
        conn: sqlite3.Connection,
        filter: dict[str, Any] | None,
        before: RunnableConfig | None,
        limit: int | None,
    ) -> Iterator[CheckpointTuple]:
        """列出指定线程的 checkpoints"""
        before_checkpoint_id = get_checkpoint_id(before) if before else None

        query = """SELECT * FROM checkpoints
                   WHERE thread_id=? AND checkpoint_ns=?
                   ORDER BY checkpoint_id DESC"""
        params: list = [thread_id, checkpoint_ns]

        rows = conn.execute(query, params).fetchall()

        count = 0
        for row in rows:
            if before_checkpoint_id and row["checkpoint_id"] >= before_checkpoint_id:
                continue

            if limit is not None and count >= limit:
                break

            # 对 metadata 进行过滤
            if filter:
                metadata = self.serde.loads_typed(
                    (row["metadata_type"], bytes(row["metadata_data"]))
                )
                if not all(
                    query_value == metadata.get(query_key)
                    for query_key, query_value in filter.items()
                ):
                    continue

            count += 1
            yield self._row_to_tuple(
                thread_id,
                checkpoint_ns,
                row,
                conn,
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": row["checkpoint_id"],
                    }
                },
            )

    # ------------------------------------------------------------------
    # Sync: put / put_writes
    # ------------------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """同步保存 checkpoint"""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        conn = self._sync_conn()

        # checkpoint 的副本: 移除 channel_values (单独存储在 blobs 中)
        c = dict(checkpoint)
        channel_values: dict[str, Any] = c.pop("channel_values", {})

        # 序列化
        checkpoint_serialized = self.serde.dumps_typed(c)
        metadata_serialized = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )

        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # 插入/更新 checkpoints 表
        conn.execute(
            """INSERT OR REPLACE INTO checkpoints
               (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                checkpoint_type, checkpoint_data, metadata_type, metadata_data, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                thread_id,
                checkpoint_ns,
                c["id"],
                parent_checkpoint_id,
                checkpoint_serialized[0],
                checkpoint_serialized[1],
                metadata_serialized[0],
                metadata_serialized[1],
            ),
        )

        # 保存 channel blobs
        for k, v in new_versions.items():
            if k in channel_values:
                blob_serialized = self.serde.dumps_typed(channel_values[k])
            else:
                blob_serialized = ("empty", b"")
            conn.execute(
                """INSERT OR REPLACE INTO checkpoint_blobs
                   (thread_id, checkpoint_ns, channel, version, blob_type, blob_data)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    checkpoint_ns,
                    k,
                    v,
                    blob_serialized[0],
                    blob_serialized[1],
                ),
            )

        # 更新会话活跃时间
        conn.execute(
            """INSERT OR REPLACE INTO session_meta (thread_id, last_active_at)
               VALUES (?, datetime('now'))""",
            (thread_id,),
        )

        conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": c["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """同步保存中间写入"""
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id: str = config["configurable"]["checkpoint_id"]
        conn = self._sync_conn()

        for idx, (channel, value) in enumerate(writes):
            mapped_idx = WRITES_IDX_MAP.get(channel, idx)
            serialized = self.serde.dumps_typed(value)
            conn.execute(
                """INSERT OR IGNORE INTO checkpoint_writes
                   (thread_id, checkpoint_ns, checkpoint_id, task_id, idx,
                    channel, value_type, value_data, task_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    mapped_idx,
                    channel,
                    serialized[0],
                    serialized[1],
                    task_path,
                ),
            )

        conn.commit()

    # ------------------------------------------------------------------
    # Sync: delete_thread
    # ------------------------------------------------------------------

    def delete_thread(self, thread_id: str) -> None:
        """同步删除线程的所有数据"""
        conn = self._sync_conn()
        conn.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
        conn.execute("DELETE FROM checkpoint_writes WHERE thread_id=?", (thread_id,))
        conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id=?", (thread_id,))
        conn.execute("DELETE FROM session_meta WHERE thread_id=?", (thread_id,))
        conn.commit()
        logger.info(f"已删除线程: {thread_id}")

    # ------------------------------------------------------------------
    # Async: aget_tuple
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """异步获取 checkpoint tuple"""
        # 将同步操作跑在线程池中以避免阻塞事件循环
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_tuple, config)

    # ------------------------------------------------------------------
    # Async: alist
    # ------------------------------------------------------------------

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """异步列出 checkpoints"""
        loop = asyncio.get_running_loop()
        for item in await loop.run_in_executor(
            None,
            lambda: list(self.list(config, filter=filter, before=before, limit=limit)),
        ):
            yield item

    # ------------------------------------------------------------------
    # Async: aput / aput_writes
    # ------------------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """异步保存 checkpoint"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """异步保存中间写入"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self.put_writes, config, writes, task_id, task_path
        )

    # ------------------------------------------------------------------
    # Async: adelete_thread
    # ------------------------------------------------------------------

    async def adelete_thread(self, thread_id: str) -> None:
        """异步删除线程"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.delete_thread, thread_id)

    # ------------------------------------------------------------------
    # 会话清理
    # ------------------------------------------------------------------

    def cleanup_expired_sessions(self, max_age_days: int = 7) -> int:
        """清理过期会话

        Args:
            max_age_days: 超过此天数的会话将被删除

        Returns:
            int: 清理的会话数量
        """
        conn = self._sync_conn()

        rows = conn.execute(
            """SELECT thread_id FROM session_meta
               WHERE datetime(last_active_at) < datetime('now', ?)""",
            (f"-{max_age_days} days",),
        ).fetchall()

        expired_ids = [r["thread_id"] for r in rows]
        for tid in expired_ids:
            self.delete_thread(tid)

        if expired_ids:
            logger.info(f"清理了 {len(expired_ids)} 个过期会话 (>{max_age_days}天)")
        return len(expired_ids)

    async def acleanup_expired_sessions(self, max_age_days: int = 7) -> int:
        """异步清理过期会话"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.cleanup_expired_sessions, max_age_days
        )

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取存储统计"""
        conn = self._sync_conn()
        checkpoint_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM checkpoints"
        ).fetchone()["cnt"]
        session_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM session_meta"
        ).fetchone()["cnt"]
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "checkpoint_count": checkpoint_count,
            "session_count": session_count,
        }

    # ------------------------------------------------------------------
    # 资源清理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭同步连接"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
            logger.info("SqliteSaver 同步连接已关闭")

    async def aclose(self) -> None:
        """异步关闭 (关闭线程池中的同步连接)"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close)
