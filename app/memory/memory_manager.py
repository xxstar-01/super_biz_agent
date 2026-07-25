"""
统一记忆管理器 (MemoryManager)

协调多层记忆的读写:
  L1 工作记忆 - LangGraph State (不管理, 由 graph 自行流转)
  L2 会话记忆 - SqliteSaver (checkpoint 持久化)
  L3 情景记忆 - EpisodeStore (任务历史记录 + 向量检索)
  L4 语义记忆 - SemanticExtractor (成功任务 → LLM 提取经验 → Milvus)

职责:
  - 任务开始时: 创建 episode 记录
  - 任务执行中: 更新 plan 和 past_steps
  - 任务完成后: 标记 episode 完成, 触发 L4 经验提取 (后台线程)
  - Planner 集成: 提供历史经验上下文 (L3 + L4 合并)
  - 定时清理: 过期会话清理
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .sqlite_saver import SqliteSaver
from .episode_store import EpisodeStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """统一记忆管理器

    使用示例:
        manager = MemoryManager(
            db_path="data/memory.db",
            enable_semantic=True,
        )

        # 任务开始
        ep_id = manager.start_episode("session-1", "分析CPU", task_type="aiops")

        # 获取历史经验 (供 Planner 使用)
        experience = manager.get_experience_context("分析CPU")

        # 任务完成 → 自动触发 L4 经验提取
        manager.complete_episode(ep_id, plan, past_steps, response)
    """

    def __init__(
        self,
        db_path: str = "data/memory.db",
        *,
        session_expiry_days: int = 7,
        enable_episodic: bool = True,
        enable_vector_search: bool = True,
        enable_semantic: bool = True,
        semantic_model: str = "qwen-plus",
    ) -> None:
        """
        Args:
            db_path: SQLite 数据库路径
            session_expiry_days: 会话过期天数
            enable_episodic: 是否启用 L3 情景记忆
            enable_vector_search: L3 是否使用向量检索 (Milvus), 关闭时回退关键词
            enable_semantic: 是否启用 L4 语义记忆 (成功后自动提取经验)
            semantic_model: L4 经验提取使用的 LLM 模型
        """
        self.db_path = db_path
        self.session_expiry_days = session_expiry_days
        self.enable_episodic = enable_episodic
        self.enable_semantic = enable_semantic
        self.semantic_model = semantic_model

        # L2: 会话记忆
        self.session_store = SqliteSaver(db_path=db_path)

        # L3: 情景记忆
        self.episode_store = (
            EpisodeStore(
                db_path=db_path,
                vector_search_enabled=enable_vector_search,
            )
            if enable_episodic
            else None
        )

        # L4: 语义提取器 (延迟加载)
        self._semantic_extractor = None

        # 运行中的 episode 缓存: {session_id: episode_id}
        self._active_episodes: dict[str, int] = {}

        logger.info(
            f"MemoryManager 初始化完成: db={db_path}, "
            f"episodic={enable_episodic}, vector_search={enable_vector_search}, "
            f"semantic={enable_semantic}, expiry={session_expiry_days}d"
        )

    @property
    def semantic_extractor(self):
        """延迟获取语义提取器 (避免未配置 API Key 时启动失败)"""
        if self._semantic_extractor is None and self.enable_semantic:
            from .semantic_extractor import get_semantic_extractor
            self._semantic_extractor = get_semantic_extractor(self.semantic_model)
        return self._semantic_extractor

    # ------------------------------------------------------------------
    # L3: Episode 生命周期
    # ------------------------------------------------------------------

    def start_episode(
        self,
        session_id: str,
        task_input: str,
        task_type: str = "general",
    ) -> int | None:
        """开始记录一个任务

        Returns:
            episode_id, 如果未启用 episodic 则返回 None
        """
        if not self.episode_store:
            return None

        ep_id = self.episode_store.start_episode(
            session_id=session_id,
            task_input=task_input,
            task_type=task_type,
        )
        self._active_episodes[session_id] = ep_id
        return ep_id

    def update_episode_plan(self, session_id: str, plan: list[str]) -> None:
        """更新执行计划"""
        if not self.episode_store:
            return
        ep_id = self._active_episodes.get(session_id)
        if ep_id is not None:
            self.episode_store.update_plan(ep_id, plan)

    def update_episode_steps(
        self, session_id: str, past_steps: list[tuple[str, str]]
    ) -> None:
        """更新已执行步骤"""
        if not self.episode_store:
            return
        ep_id = self._active_episodes.get(session_id)
        if ep_id is not None:
            self.episode_store.update_past_steps(ep_id, past_steps)

    def complete_episode(
        self,
        session_id: str,
        plan: list[str],
        past_steps: list[tuple[str, str]],
        response: str,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """标记任务完成

        - 写入 L3 情景记忆 (SQLite + Milvus 向量索引)
        - 如果成功, 触发 L4 语义经验提取 (后台线程, 不阻塞)
        """
        if not self.episode_store:
            return
        ep_id = self._active_episodes.pop(session_id, None)
        if ep_id is None:
            # 尝试从数据库查询最近 running 状态的 episode
            episodes = self.episode_store.get_session_episodes(session_id)
            for ep in episodes:
                if ep.status == "running":
                    ep_id = ep.episode_id
                    break

        if ep_id is not None:
            # 获取 task_input 用于后续 L4 提取
            ep_record = self.episode_store.get_episode(ep_id)
            task_input = ep_record.task_input if ep_record else ""

            # L3: 写入 episode (含 Milvus 向量索引)
            self.episode_store.complete_episode(
                episode_id=ep_id,
                plan=plan,
                past_steps=past_steps,
                response=response,
                status=status,
                error_message=error_message,
            )

            logger.info(
                f"Episode {ep_id} 完成: session={session_id}, status={status}"
            )

            # L4: 成功后后台提取语义经验 (6.2)
            if status == "completed" and self.enable_semantic and self.semantic_extractor:
                try:
                    self.semantic_extractor.extract_and_store_async(
                        task_input=task_input,
                        plan=plan,
                        past_steps=past_steps,
                        response=response,
                        episode_id=ep_id,
                    )
                except Exception as e:
                    logger.warning(f"L4 经验提取提交失败 (非致命): {e}")
        else:
            logger.warning(f"未找到活跃 episode: session={session_id}")

    # ------------------------------------------------------------------
    # 经验检索 (供 Planner 使用, L3 + L4 合并)
    # ------------------------------------------------------------------

    def get_experience_context(self, task_input: str, limit: int = 3) -> str:
        """获取历史经验上下文文本

        从 L3 情景记忆查询相似任务 + L4 语义经验,
        合并后格式化为 Planner 可用的文本。
        """
        parts = []

        # L3: 情景记忆经验
        if self.episode_store:
            l3_text = self.episode_store.format_experience_context(task_input, limit)
            if l3_text:
                parts.append(l3_text)

        # L4: 语义记忆经验 (向量检索)
        if self.enable_semantic:
            try:
                l4_text = self._get_semantic_context(task_input, limit)
                if l4_text:
                    parts.append(l4_text)
            except Exception as e:
                logger.warning(f"L4 经验检索失败: {e}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def _get_semantic_context(self, query: str, limit: int = 3) -> str:
        """从 L4 语义记忆中检索相关经验"""
        try:
            from app.services.vector_embedding_service import vector_embedding_service
            from app.core.milvus_client import milvus_manager

            milvus_manager.connect()
            collection = milvus_manager.get_collection()

            query_vector = vector_embedding_service.embed_query(query)

            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 16},
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=f'metadata["type"] == "semantic_experience"',
                output_fields=["metadata", "content"],
            )

            if not results or not results[0]:
                return ""

            lines = ["## 语义经验 (L4 知识库)\n"]
            for hit in results[0]:
                try:
                    meta = hit.entity.get("metadata")
                    if isinstance(meta, str):
                        meta = json.loads(meta)
                    title = meta.get("title", "未知")
                    problem = meta.get("problem", "")
                    solution = meta.get("solution", "")
                    tags = meta.get("tags", [])

                    lines.append(f"### 💡 {title}")
                    lines.append(f"- **问题**: {problem}")
                    if solution:
                        lines.append(f"- **方案**: {solution}")
                    if tags:
                        lines.append(f"- **标签**: {', '.join(tags)}")
                    lines.append("")
                except Exception:
                    continue

            return "\n".join(lines) if len(lines) > 1 else ""

        except Exception as e:
            logger.warning(f"L4 语义经验检索失败: {e}")
            return ""

    # ------------------------------------------------------------------
    # L2: 会话清理
    # ------------------------------------------------------------------

    def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        count = self.session_store.cleanup_expired_sessions(
            max_age_days=self.session_expiry_days
        )
        if count > 0:
            logger.info(f"会话清理完成: 删除 {count} 个过期会话")
        return count

    async def acleanup_expired_sessions(self) -> int:
        """异步清理过期会话"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.cleanup_expired_sessions
        )

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取完整统计"""
        stats: dict[str, Any] = {
            "session": self.session_store.get_stats(),
        }
        if self.episode_store:
            stats["episode"] = self.episode_store.get_stats()
        return stats

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭所有连接"""
        self.session_store.close()
        if self.episode_store:
            self.episode_store.close()
        logger.info("MemoryManager 已关闭")

    async def aclose(self) -> None:
        """异步关闭"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close)
