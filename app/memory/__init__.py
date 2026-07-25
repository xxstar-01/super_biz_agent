"""
多层记忆系统 (Multi-Layer Memory System)

架构:
  L1 工作记忆 (Working)  - LangGraph State, 当前任务内存中流转
  L2 会话记忆 (Session)  - SqliteSaver, 会话状态持久化到 SQLite
  L3 情景记忆 (Episodic) - EpisodeStore, 任务历史记录 + 向量检索 (Milvus)
  L4 语义记忆 (Semantic) - SemanticExtractor, LLM 提取经验 → Milvus 向量库
"""

from .sqlite_saver import SqliteSaver
from .episode_store import EpisodeStore
from .memory_manager import MemoryManager
from .semantic_extractor import SemanticExtractor, get_semantic_extractor

__all__ = [
    "SqliteSaver",
    "EpisodeStore",
    "MemoryManager",
    "SemanticExtractor",
    "get_semantic_extractor",
    "get_memory_manager",
]

_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """获取全局 MemoryManager 单例

    延迟初始化, 首次调用时根据 config 创建。
    """
    global _memory_manager
    if _memory_manager is None:
        from app.config import config

        _memory_manager = MemoryManager(
            db_path=config.memory_db_path,
            session_expiry_days=config.memory_session_expiry_days,
            enable_episodic=config.memory_episodic_enabled,
            enable_vector_search=config.memory_vector_search_enabled,
            enable_semantic=config.memory_semantic_enabled,
            semantic_model=config.memory_semantic_model,
        )
    return _memory_manager
