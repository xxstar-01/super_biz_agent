"""
L3 情景记忆 - EpisodeStore

记录每次任务的完整执行历史:
- 任务描述、执行计划、每步结果、最终响应
- 成功/失败状态、耗时
- 向量检索 (Milvus): 任务完成后自动索引 task_input, 检索时用向量相似度代替关键词匹配
- 回退: Milvus 不可用时自动回退到 SQL LIKE 关键词匹配
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 常量: Milvus 中 episode 向量的 ID 前缀和 metadata type 标记
# ------------------------------------------------------------------
EPISODE_ID_PREFIX = "ep_"
EPISODE_META_TYPE = "episode"


EPISODE_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    task_input      TEXT NOT NULL,
    task_type       TEXT NOT NULL DEFAULT 'general',
    plan_json       TEXT NOT NULL DEFAULT '[]',
    past_steps_json TEXT NOT NULL DEFAULT '[]',
    response        TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    duration_ms     INTEGER,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_episodes_session
    ON episodes(session_id);

CREATE INDEX IF NOT EXISTS idx_episodes_task_type
    ON episodes(task_type);

CREATE INDEX IF NOT EXISTS idx_episodes_status
    ON episodes(status);

CREATE INDEX IF NOT EXISTS idx_episodes_created_at
    ON episodes(created_at);
"""


@dataclass
class EpisodeRecord:
    """单次任务记录"""
    session_id: str
    task_input: str
    task_type: str = "general"
    plan: list[str] = field(default_factory=list)
    past_steps: list[tuple[str, str]] = field(default_factory=list)
    response: str = ""
    status: str = "pending"  # pending | running | completed | failed
    error_message: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    episode_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "task_input": self.task_input,
            "task_type": self.task_type,
            "plan": self.plan,
            "past_steps": [
                {"step": s, "result": r[:500]} for s, r in self.past_steps
            ],
            "response": self.response[:1000],
            "status": self.status,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }


class EpisodeStore:
    """情景记忆存储

    每次 Agent 任务完成后, 写入一条 episode 记录。
    支持向量检索 (Milvus) 和关键词回退两种模式。

    使用示例:
        store = EpisodeStore(db_path="data/memory.db")
        store.start_episode("session-1", "分析 CPU 使用率", task_type="aiops")
        ...
        store.complete_episode("session-1", plan, past_steps, response)
    """

    def __init__(
        self,
        db_path: str = "data/memory.db",
        *,
        vector_search_enabled: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_search_enabled = vector_search_enabled
        self._local = threading.local()
        self._init_db()
        logger.info(
            f"EpisodeStore 初始化完成, db={self.db_path}, "
            f"vector_search={vector_search_enabled}"
        )

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(EPISODE_SCHEMA)
        conn.commit()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def start_episode(
        self,
        session_id: str,
        task_input: str,
        task_type: str = "general",
    ) -> int:
        """开始记录一个任务, 返回 episode_id"""
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """INSERT INTO episodes
               (session_id, task_input, task_type, status, started_at)
               VALUES (?, ?, ?, 'running', ?)""",
            (session_id, task_input, task_type, now),
        )
        conn.commit()
        episode_id = cursor.lastrowid
        logger.debug(f"Episode {episode_id} 已开始: {task_input[:80]}...")
        return episode_id

    def update_plan(self, episode_id: int, plan: list[str]) -> None:
        """更新执行计划"""
        conn = self._conn()
        conn.execute(
            "UPDATE episodes SET plan_json=? WHERE id=?",
            (json.dumps(plan, ensure_ascii=False), episode_id),
        )
        conn.commit()

    def update_past_steps(
        self, episode_id: int, past_steps: list[tuple[str, str]]
    ) -> None:
        """更新已执行步骤"""
        conn = self._conn()
        conn.execute(
            "UPDATE episodes SET past_steps_json=? WHERE id=?",
            (json.dumps(past_steps, ensure_ascii=False), episode_id),
        )
        conn.commit()

    def complete_episode(
        self,
        episode_id: int,
        plan: list[str],
        past_steps: list[tuple[str, str]],
        response: str,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """标记任务完成, 并自动索引到 Milvus"""
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()

        # 计算耗时
        row = conn.execute(
            "SELECT started_at FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()
        duration_ms = None
        if row and row["started_at"]:
            try:
                started = datetime.fromisoformat(str(row["started_at"]))
                completed = datetime.now(timezone.utc)
                duration_ms = int(
                    (completed - started.replace(tzinfo=timezone.utc)).total_seconds()
                    * 1000
                )
            except Exception:
                pass

        conn.execute(
            """UPDATE episodes SET
               plan_json=?, past_steps_json=?, response=?, status=?,
               error_message=?, completed_at=?, duration_ms=?
               WHERE id=?""",
            (
                json.dumps(plan, ensure_ascii=False),
                json.dumps(past_steps, ensure_ascii=False),
                response,
                status,
                error_message,
                now,
                duration_ms,
                episode_id,
            ),
        )
        conn.commit()
        logger.info(
            f"Episode {episode_id} {status}, 耗时: {duration_ms}ms"
        )

        # 索引到 Milvus (不阻塞, 失败不影响主流程)
        if self.vector_search_enabled:
            try:
                self._index_episode_to_milvus(episode_id)
            except Exception as e:
                logger.warning(f"Episode {episode_id} Milvus 索引失败 (非致命): {e}")

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_episode(self, episode_id: int) -> EpisodeRecord | None:
        """获取单条记录"""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def get_recent_episodes(
        self,
        limit: int = 10,
        task_type: str | None = None,
        status: str | None = None,
    ) -> list[EpisodeRecord]:
        """获取最近的任务记录"""
        conn = self._conn()
        query = "SELECT * FROM episodes WHERE 1=1"
        params: list = []

        if task_type:
            query += " AND task_type=?"
            params.append(task_type)
        if status:
            query += " AND status=?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_similar_episodes(
        self,
        task_input: str,
        limit: int = 5,
    ) -> list[EpisodeRecord]:
        """根据任务描述搜索相似的历史任务

        优先使用 Milvus 向量检索, 不可用时回退到 SQL LIKE 关键词匹配。
        """
        if self.vector_search_enabled:
            try:
                return self._vector_search_similar(task_input, limit)
            except Exception as e:
                logger.warning(f"向量检索失败, 回退到关键词匹配: {e}")

        return self._keyword_search_similar(task_input, limit)

    def get_session_episodes(self, session_id: str) -> list[EpisodeRecord]:
        """获取某个会话的所有任务记录"""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM episodes WHERE session_id=? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # 向量检索 (Milvus) — 6.1
    # ------------------------------------------------------------------

    def _get_embedding_service(self):
        """延迟导入 embedding 服务, 避免循环依赖"""
        from app.services.vector_embedding_service import vector_embedding_service
        return vector_embedding_service

    def _get_milvus_collection(self):
        """延迟导入 Milvus collection, 避免连接问题"""
        from app.core.milvus_client import milvus_manager
        milvus_manager.connect()
        return milvus_manager.get_collection()

    def _index_episode_to_milvus(self, episode_id: int) -> None:
        """将 episode 的 task_input 向量化后写入 Milvus

        使用 metadata {"type": "episode", "episode_id": <id>} 标记,
        与文档知识库隔离。
        """
        conn = self._conn()
        row = conn.execute(
            "SELECT task_input FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()
        if not row:
            return

        task_input = row["task_input"]
        if not task_input or not task_input.strip():
            return

        embedder = self._get_embedding_service()
        vector = embedder.embed_query(task_input)

        collection = self._get_milvus_collection()
        doc_id = f"{EPISODE_ID_PREFIX}{episode_id}"

        # 先删除旧向量 (幂等)
        try:
            collection.delete(f'id == "{doc_id}"')
        except Exception:
            pass

        # 插入新向量
        collection.insert([
            [doc_id],                         # id
            [vector],                         # vector
            [task_input[:8000]],              # content (max 8000 chars per schema)
            [json.dumps({                     # metadata
                "type": EPISODE_META_TYPE,
                "episode_id": episode_id,
            })],
        ])
        collection.flush()
        logger.debug(f"Episode {episode_id} 已索引到 Milvus, id={doc_id}")

    def _vector_search_similar(
        self, task_input: str, limit: int = 5
    ) -> list[EpisodeRecord]:
        """用向量相似度搜索历史 episode

        流程: embed query → Milvus search with type filter → SQLite lookup
        """
        # embedding 不需要 Milvus, 先做
        embedder = self._get_embedding_service()
        query_vector = embedder.embed_query(task_input)

        # Milvus 搜索
        collection = self._get_milvus_collection()
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 16},
        }

        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=limit * 2,  # 多取一些, 因为可能有些 episode 已被删除
            expr=f'metadata["type"] == "{EPISODE_META_TYPE}"',
            output_fields=["metadata"],
        )

        if not results or not results[0]:
            return []

        # 提取 episode IDs, 保留顺序
        episode_ids = []
        for hit in results[0]:
            try:
                meta = hit.entity.get("metadata")
                if isinstance(meta, str):
                    meta = json.loads(meta)
                ep_id = meta.get("episode_id")
                if ep_id is not None:
                    episode_ids.append(ep_id)
            except Exception:
                continue

        if not episode_ids:
            return []

        # 从 SQLite 批量加载 episode 记录
        conn = self._conn()
        placeholders = ",".join(["?" for _ in episode_ids])
        rows = conn.execute(
            f"""SELECT * FROM episodes
                WHERE id IN ({placeholders})
                  AND status='completed'
                ORDER BY created_at DESC""",
            episode_ids,
        ).fetchall()

        # 按向量相似度顺序排列 (Milvus 结果顺序)
        id_order = {ep_id: idx for idx, ep_id in enumerate(episode_ids)}
        records = [self._row_to_record(r) for r in rows]
        records.sort(key=lambda r: id_order.get(r.episode_id, 9999))

        return records[:limit]

    # ------------------------------------------------------------------
    # 关键词回退 (SQL LIKE)
    # ------------------------------------------------------------------

    def _keyword_search_similar(
        self, task_input: str, limit: int = 5
    ) -> list[EpisodeRecord]:
        """关键词匹配搜索 (Milvus 不可用时的回退方案)"""
        conn = self._conn()
        keywords = [
            w.strip() for w in task_input.replace("，", ",").split(",") if w.strip()
        ]
        if not keywords:
            keywords = task_input.split()[:5]

        conditions = " OR ".join(["task_input LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]

        rows = conn.execute(
            f"""SELECT * FROM episodes
                WHERE status='completed' AND ({conditions})
                ORDER BY created_at DESC LIMIT ?""",
            params + [limit],
        ).fetchall()

        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------------
    # 删除 episode (含 Milvus 清理)
    # ------------------------------------------------------------------

    def delete_episode(self, episode_id: int) -> bool:
        """删除 episode 及对应的 Milvus 向量"""
        conn = self._conn()
        cursor = conn.execute(
            "DELETE FROM episodes WHERE id=?", (episode_id,)
        )
        conn.commit()
        deleted = cursor.rowcount > 0

        if deleted and self.vector_search_enabled:
            try:
                collection = self._get_milvus_collection()
                doc_id = f"{EPISODE_ID_PREFIX}{episode_id}"
                collection.delete(f'id == "{doc_id}"')
                collection.flush()
            except Exception as e:
                logger.warning(f"清理 Milvus 向量失败: episode={episode_id}, {e}")

        return deleted

    # ------------------------------------------------------------------
    # 格式化: 将历史记录转为 Planner 可用的经验文本
    # ------------------------------------------------------------------

    def format_experience_context(
        self,
        task_input: str,
        limit: int = 3,
    ) -> str:
        """将相似历史任务格式化为经验上下文文本

        可直接注入 Planner 的 system prompt。
        """
        episodes = self.get_similar_episodes(task_input, limit=limit)
        if not episodes:
            return ""

        lines = ["## 历史相似任务经验\n"]
        for i, ep in enumerate(episodes, 1):
            lines.append(f"### 经验 {i}: {ep.task_input[:100]}")
            if ep.plan:
                lines.append(f"- 执行计划: {' → '.join(ep.plan[:5])}")
            lines.append(f"- 状态: {ep.status}")
            if ep.duration_ms:
                lines.append(f"- 耗时: {ep.duration_ms // 1000}秒")
            if ep.past_steps:
                last_step, last_result = ep.past_steps[-1]
                lines.append(f"- 最后步骤: {last_step}")
                lines.append(f"- 结果摘要: {last_result[:200]}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) as cnt FROM episodes").fetchone()["cnt"]
        completed = conn.execute(
            "SELECT COUNT(*) as cnt FROM episodes WHERE status='completed'"
        ).fetchone()["cnt"]
        failed = conn.execute(
            "SELECT COUNT(*) as cnt FROM episodes WHERE status='failed'"
        ).fetchone()["cnt"]

        avg_duration = conn.execute(
            "SELECT AVG(duration_ms) as avg FROM episodes WHERE duration_ms IS NOT NULL"
        ).fetchone()["avg"]

        return {
            "total_episodes": total,
            "completed": completed,
            "failed": failed,
            "success_rate": round(completed / total * 100, 1) if total > 0 else 0,
            "avg_duration_ms": round(avg_duration, 0) if avg_duration else None,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _row_to_record(self, row: sqlite3.Row) -> EpisodeRecord:
        """数据库行 → EpisodeRecord"""
        plan = json.loads(row["plan_json"]) if row["plan_json"] else []
        past_steps_raw = (
            json.loads(row["past_steps_json"]) if row["past_steps_json"] else []
        )
        past_steps = [tuple(s) for s in past_steps_raw]

        return EpisodeRecord(
            episode_id=row["id"],
            session_id=row["session_id"],
            task_input=row["task_input"],
            task_type=row["task_type"],
            plan=plan,
            past_steps=past_steps,
            response=row["response"],
            status=row["status"],
            error_message=row["error_message"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_ms=row["duration_ms"],
        )

    def close(self) -> None:
        """关闭连接"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
