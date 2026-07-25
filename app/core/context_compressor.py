"""上下文自动压缩中间件

当对话历史的 token 数超过上下文窗口的指定阈值（默认 70%）时，
自动使用大模型对历史对话进行摘要压缩，减少 token 消耗。

原理：
- 通过 AgentMiddleware.abefore_model 钩子，在每次模型调用前检查消息总 token 数
- 超过阈值时：分离旧消息 → 调用 LLM 生成摘要 → 用摘要替换旧消息
- 摘要消息作为 SystemMessage 插入，保留最近 N 轮对话不压缩

前端通知：
- 每次压缩完成后，将摘要存入 _events dict（按 thread_id 索引）
- 调用方通过 pop_event() 获取并消费压缩事件
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from loguru import logger

from app.config import config

# ---------------------------------------------------------------------------
# 用于生成摘要的系统提示词
# ---------------------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要生成器。请将以下对话历史总结为简洁的摘要。

要求：
1. 保留用户的核心问题和需求
2. 保留重要的结论、决策和行动项
3. 保留关键的数据、数字和事实
4. 忽略重复的、无关紧要的细节
5. 使用中文输出
6. 摘要要简洁，不要超过 500 字"""

# ---------------------------------------------------------------------------
# 前端展示的压缩分隔线文案
# ---------------------------------------------------------------------------
COMPRESSION_SEPARATOR = "---------上下文已自动压缩---------"


class ContextCompressorMiddleware(AgentMiddleware):
    """上下文自动压缩中间件

    在每次模型调用前自动检查并压缩过长的对话历史。

    使用方式：
        middleware = ContextCompressorMiddleware(compression_llm)
        agent = create_agent(model, tools, middleware=[middleware])

    获取压缩事件（用于前端通知）：
        event = middleware.pop_event(thread_id)
        if event:
            # event 包含摘要文本
    """

    def __init__(self, compression_llm):
        """初始化压缩中间件

        Args:
            compression_llm: 用于生成摘要的 LLM 实例（建议用非流式模型）
        """
        super().__init__()
        self.compression_llm = compression_llm
        # 按 thread_id 存储压缩事件，供前端获取
        self._events: dict[str, str] = {}

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _count_tokens(self, messages: list[BaseMessage]) -> int:
        """计算消息列表的总 token 数

        优先使用 LLM 的精确计数方法，失败时回退到近似计数。
        """
        try:
            # ChatQwen / ChatOpenAI 都提供此方法
            if hasattr(self.compression_llm, "get_num_tokens_from_messages"):
                return self.compression_llm.get_num_tokens_from_messages(messages)
        except Exception:
            pass
        return count_tokens_approximately(messages)

    def _should_compress(self, messages: list[BaseMessage]) -> bool:
        """判断是否需要触发压缩"""
        if not config.context_compression_enabled:
            return False
        if len(messages) <= config.context_compression_keep_recent * 2 + 2:
            # 消息太少，不需要压缩
            return False

        token_count = self._count_tokens(messages)
        limit = int(
            config.context_compression_window_size
            * config.context_compression_threshold
        )
        if token_count > limit:
            logger.info(
                f"上下文压缩触发: token 数 {token_count} > 阈值 {limit} "
                f"({config.context_compression_threshold:.0%} × {config.context_compression_window_size})"
            )
            return True
        return False

    def _find_existing_summary(self, messages: list[BaseMessage]) -> str | None:
        """查找消息列表中是否已有历史摘要"""
        for msg in messages:
            if isinstance(msg, SystemMessage) and "[对话历史摘要]" in str(msg.content):
                return str(msg.content)
        return None

    def _split_messages(
        self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], list[BaseMessage]]:
        """拆分消息为「待压缩的旧消息」和「保留的最近消息」

        保留策略：
        - 第一条 SystemMessage（系统提示词）始终保留
        - 最近 N 轮对话（= keep_recent × 2 条消息）不压缩
        - 如果消息中存在旧的摘要消息，它也会被归入旧消息中
        - 保证不拆散 tool_calls ↔ ToolMessage 配对：
          如果切分点落在 AIMessage(tool_calls=[...]) 和 ToolMessage 之间，
          就把切分点前移到包含那个 AIMessage，迭代处理直到所有配对完整

        Returns:
            (old_messages, recent_messages)
        """
        keep_count = config.context_compression_keep_recent * 2

        # 找到系统消息（第一条，不包含摘要）
        sys_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
        body = list(messages[1:]) if sys_msg else list(messages)

        if len(body) <= keep_count:
            return [], body

        # 初步按数量切分
        split_idx = len(body) - keep_count

        # ================================================================
        # 修复：ToolMessage 必须在对应的 AIMessage(tool_calls=[...]) 之后
        #
        # 如果近期消息中有 ToolMessage，但它的工具调用方 AIMessage
        # 被分到了旧消息中，API 会报错：
        #   "messages with role 'tool' must be a response to a
        #    preceeding message with 'tool_calls'"
        #
        # 这里循环扩大 recent 范围，直到所有 ToolMessage 的父 AIMessage
        # 都在 recent 中为止
        # ================================================================
        while True:
            recent = body[split_idx:]

            # 收集 recent 中所有 ToolMessage 引用的 tool_call_id
            orphaned_ids: set[str] = set()
            for msg in recent:
                if isinstance(msg, ToolMessage):
                    orphaned_ids.add(msg.tool_call_id)

            if not orphaned_ids:
                break  # 没有 ToolMessage，安全

            # 在 old 中从后往前找匹配的 AIMessage
            earliest_match = split_idx
            for i in range(split_idx - 1, -1, -1):
                msg = body[i]
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                        if tc_id in orphaned_ids:
                            earliest_match = min(earliest_match, i)
                            orphaned_ids.discard(tc_id)
                            break

            if earliest_match == split_idx:
                break  # 没有找到匹配，切分点安全

            # 扩大 recent 范围，包含找到的 AIMessage
            split_idx = earliest_match
            # 继续循环：新纳入 recent 的消息中可能又有 ToolMessage，
            # 需要进一步回溯（工具链式调用场景）

        old = body[:split_idx]
        recent = body[split_idx:]
        return old, recent

    async def _generate_summary(
        self, old_messages: list[BaseMessage], existing_summary: str | None = None
    ) -> str:
        """用 LLM 生成对话摘要

        Args:
            old_messages: 待压缩的旧消息
            existing_summary: 已有的摘要文本（合并用）

        Returns:
            摘要文本
        """
        # 构建对话文本
        lines: list[str] = []
        for msg in old_messages:
            role = {
                HumanMessage: "用户",
                AIMessage: "AI助手",
                SystemMessage: "系统",
            }.get(type(msg), type(msg).__name__)
            content = msg.content
            if isinstance(content, str) and len(content) > 800:
                content = content[:800] + "..."
            lines.append(f"[{role}]: {content}")

        conversation_text = "\n".join(lines)

        if existing_summary:
            # 清理已有摘要的前缀标签
            clean_summary = existing_summary.replace("[对话历史摘要]\n", "")
            prompt = (
                f"已有的历史摘要：\n{clean_summary}\n\n"
                f"新增的对话内容：\n{conversation_text}\n\n"
                f"请将以上内容整合为一份完整、连贯的对话摘要。"
            )
        else:
            prompt = (
                f"对话历史：\n{conversation_text}\n\n"
                f"请将以上对话总结为简洁的摘要。"
            )

        try:
            response = await self.compression_llm.ainvoke([
                SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            summary = str(response.content) if hasattr(response, "content") else str(response)
            logger.info(f"摘要生成完成，长度: {len(summary)} 字符")
            return summary

        except Exception as exc:
            logger.error(f"生成摘要失败: {exc}，回退到简单截断")
            return (
                f"对话历史包含 {len(old_messages)} 条消息。"
                f"主要内容涉及用户与AI助手的多轮对话。（自动摘要生成失败，已保留最近对话）"
            )

    # ------------------------------------------------------------------
    # AgentMiddleware 钩子
    # ------------------------------------------------------------------

    async def abefore_model(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        """模型调用前的钩子 —— 在此检查并压缩上下文

        当消息 token 数超过阈值时：
        1. 拆分旧消息和最近消息
        2. 用 LLM 生成摘要
        3. 存储压缩事件（供前端展示）
        4. 返回压缩后的消息列表，替换原有消息

        Args:
            state: Agent 状态，包含 messages 字段
            runtime: 运行时上下文（本方法通过 get_config() 获取 thread_id）

        Returns:
            状态更新字典，或 None（表示无需修改）
        """
        messages: list[BaseMessage] = state.get("messages", [])

        if not self._should_compress(messages):
            return None

        # Runtime 没有 configurable 属性，必须用 get_config() 获取
        config_dict = get_config()
        thread_id = config_dict.get("configurable", {}).get("thread_id", "unknown")
        logger.info(f"[会话 {thread_id}] 开始上下文自动压缩...")

        # 查找已有摘要（用于增量合并）
        existing_summary = self._find_existing_summary(messages)

        # 拆分消息
        old_msgs, recent_msgs = self._split_messages(messages)
        if not old_msgs:
            return None

        # 生成摘要
        summary = await self._generate_summary(old_msgs, existing_summary)

        # 存储压缩事件，供前端展示
        self._events[thread_id] = summary

        # 构建压缩后的消息列表
        compressed: list[BaseMessage] = []

        # 保留原始系统提示（第一条 SystemMessage，不含摘要）
        if messages and isinstance(messages[0], SystemMessage) and "[对话历史摘要]" not in str(messages[0].content):
            compressed.append(messages[0])

        # 插入摘要消息
        compressed.append(
            SystemMessage(content=f"[对话历史摘要]\n{summary}")
        )

        # 追加最近消息
        compressed.extend(recent_msgs)

        logger.info(
            f"[会话 {thread_id}] 上下文压缩完成: "
            f"{len(messages)} 条消息 → {len(compressed)} 条消息 "
            f"(摘要 {len(summary)} 字符)"
        )

        # 返回状态更新：先删除全部消息，再写入压缩后的消息
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *compressed,
            ]
        }

    # ------------------------------------------------------------------
    # 前端通知接口
    # ------------------------------------------------------------------

    def pop_event(self, thread_id: str) -> str | None:
        """获取并消费压缩事件

        每次压缩事件只能被消费一次，消费后即从内部字典中移除。

        Args:
            thread_id: 会话/线程 ID

        Returns:
            摘要文本，如果没有待消费的事件则返回 None
        """
        return self._events.pop(thread_id, None)

    def has_event(self, thread_id: str) -> bool:
        """检查是否有待消费的压缩事件（不消费）"""
        return thread_id in self._events
