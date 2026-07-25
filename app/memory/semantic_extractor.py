"""
L4 语义记忆 - SemanticExtractor

从成功完成的任务中提取可复用的经验和最佳实践,
写入 Milvus 向量库, 供后续 Planner 检索使用。

提取流程:
  1. Episode 完成 → 收集 plan + past_steps + response
  2. LLM 提取: 标题、问题、根因、方案、标签
  3. 向量化 → 写入 Milvus biz collection
  4. 标记 metadata["type"] = "semantic_experience"

兼容性:
  - Milvus 不可用时静默跳过 (非致命)
  - 提取工作在后台线程执行, 不阻塞主流程
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from textwrap import dedent
from typing import Any

logger = logging.getLogger(__name__)

# Milvus 中语义经验的 ID 前缀和 metadata type 标记
EXPERIENCE_ID_PREFIX = "l4exp_"
EXPERIENCE_META_TYPE = "semantic_experience"

# LLM 提取 prompt
EXTRACTION_PROMPT = dedent("""
你是一个经验总结专家。请根据以下任务执行记录, 提取可复用的经验和最佳实践。

## 原始任务
{task_input}

## 执行计划
{plan}

## 执行步骤与结果
{past_steps}

## 最终响应
{response_summary}

---

请以 JSON 格式输出提取的经验, 包含以下字段:
- title: 经验标题 (20字以内, 简洁概括)
- problem: 问题的本质是什么 (50字以内)
- root_cause: 根本原因 (如适用, 否则填 "无")
- solution: 解决方案/处理方法 (100字以内)
- how_to_apply: 如何应用这条经验 (50字以内, 给未来任务的建议)
- tags: 3-5个标签 (如: "CPU", "告警", "性能分析")

**重要**:
- 只提取有通用价值的经验, 如果任务过于简单或没有可复用经验, 返回 null
- 不要编造信息, 所有内容基于实际执行记录
- 输出必须是有效的 JSON 格式

输出格式:
```json
{{
    "title": "...",
    "problem": "...",
    "root_cause": "...",
    "solution": "...",
    "how_to_apply": "...",
    "tags": ["...", "..."]
}}
```

如果任务没有可提取的经验, 请输出:
```json
null
```
""").strip()


class SemanticExtractor:
    """L4 语义记忆提取器

    使用 LLM 从成功的任务执行记录中提取可复用的经验,
    向量化后写入 Milvus。

    使用示例:
        extractor = SemanticExtractor(model_name="qwen-plus")
        extractor.extract_and_store(
            task_input="分析CPU高负载",
            plan=["步骤1", "步骤2"],
            past_steps=[("步骤1", "CPU 95%"), ("步骤2", "OOM Kill")],
            response="# 分析报告...",
            episode_id=42,
        )
    """

    def __init__(
        self,
        model_name: str = "qwen-plus",
        max_workers: int = 2,
    ) -> None:
        """
        Args:
            model_name: 用于提取经验的 LLM 模型
            max_workers: 后台线程池大小
        """
        self.model_name = model_name
        self._extraction_lock = threading.Lock()
        logger.info(f"SemanticExtractor 初始化完成, model={model_name}")

    # ------------------------------------------------------------------
    # 公开接口: 异步提取 + 写入
    # ------------------------------------------------------------------

    def extract_and_store_async(
        self,
        task_input: str,
        plan: list[str],
        past_steps: list[tuple[str, str]],
        response: str,
        episode_id: int,
    ) -> None:
        """在后台线程中执行经验提取和存储

        不阻塞主流程, 失败静默跳过。
        """
        thread = threading.Thread(
            target=self._extract_and_store,
            args=(task_input, plan, past_steps, response, episode_id),
            daemon=True,
            name=f"l4-extract-ep{episode_id}",
        )
        thread.start()
        logger.debug(f"L4 经验提取已提交后台: episode={episode_id}")

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _extract_and_store(
        self,
        task_input: str,
        plan: list[str],
        past_steps: list[tuple[str, str]],
        response: str,
        episode_id: int,
    ) -> None:
        """提取经验并写入 Milvus (在线程中执行)"""
        try:
            # 1. 格式化输入
            plan_text = "\n".join(f"- {s}" for s in plan) if plan else "无"
            steps_text = "\n".join(
                f"步骤: {s}\n结果: {r[:500]}" for s, r in past_steps
            ) if past_steps else "无"
            response_summary = response[:2000] if response else ""

            prompt = EXTRACTION_PROMPT.format(
                task_input=task_input[:1000],
                plan=plan_text,
                past_steps=steps_text,
                response_summary=response_summary,
            )

            # 2. LLM 提取
            experience = self._call_llm(prompt)
            if experience is None:
                logger.info(f"L4: episode={episode_id} 无可提取经验, 跳过")
                return

            logger.info(
                f"L4: episode={episode_id} 提取到经验: "
                f"title='{experience.get('title', '')}' "
                f"tags={experience.get('tags', [])}"
            )

            # 3. 写入 Milvus
            self._store_to_milvus(experience, episode_id)

        except Exception as e:
            logger.warning(f"L4 经验提取失败: episode={episode_id}, {e}")

    def _call_llm(self, prompt: str) -> dict[str, Any] | None:
        """调用 LLM 提取经验, 返回解析后的 dict 或 None"""
        try:
            from langchain_qwq import ChatQwen
            from app.config import config

            llm = ChatQwen(
                model=self.model_name,
                api_key=config.dashscope_api_key,
                temperature=0.3,
                streaming=False,
            )

            resp = llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)

            # 提取 JSON 块
            json_str = self._extract_json(content)
            if not json_str:
                return None

            result = json.loads(json_str)
            if result is None or not isinstance(result, dict):
                return None

            # 验证必要字段
            if not result.get("title"):
                return None

            return result

        except json.JSONDecodeError:
            logger.warning("L4: LLM 返回的 JSON 解析失败")
            return None
        except Exception as e:
            logger.warning(f"L4: LLM 调用失败: {e}")
            return None

    def _store_to_milvus(
        self, experience: dict[str, Any], episode_id: int
    ) -> None:
        """将提取的经验向量化后写入 Milvus"""
        try:
            from app.services.vector_embedding_service import vector_embedding_service
            from app.core.milvus_client import milvus_manager

            # 构建可检索的文本
            searchable_text = (
                f"【{experience.get('title', '')}】\n"
                f"问题: {experience.get('problem', '')}\n"
                f"根因: {experience.get('root_cause', '')}\n"
                f"方案: {experience.get('solution', '')}\n"
                f"应用建议: {experience.get('how_to_apply', '')}"
            )

            # 向量化
            vector = vector_embedding_service.embed_query(searchable_text)

            # Milvus 写入
            milvus_manager.connect()
            collection = milvus_manager.get_collection()

            doc_id = f"{EXPERIENCE_ID_PREFIX}{episode_id}"

            # 幂等: 先删除旧记录
            try:
                collection.delete(f'id == "{doc_id}"')
            except Exception:
                pass

            metadata = {
                "type": EXPERIENCE_META_TYPE,
                "source_episode_id": episode_id,
                "title": experience.get("title", ""),
                "tags": experience.get("tags", []),
                "problem": experience.get("problem", ""),
                "root_cause": experience.get("root_cause", ""),
                "solution": experience.get("solution", ""),
                "how_to_apply": experience.get("how_to_apply", ""),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }

            collection.insert([
                [doc_id],
                [vector],
                [searchable_text[:8000]],
                [json.dumps(metadata, ensure_ascii=False)],
            ])
            collection.flush()

            logger.info(
                f"L4 经验已存储到 Milvus: {doc_id}, "
                f"title='{experience.get('title', '')}'"
            )

        except Exception as e:
            logger.warning(f"L4: Milvus 写入失败: {e}")

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """从 LLM 响应中提取 JSON 块"""
        # 尝试提取 ```json ... ``` 块
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # 尝试提取 ``` ... ``` 块
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # 尝试找到 { 和 } 的完整 JSON
        brace_start = text.find("{")
        if brace_start >= 0:
            brace_count = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[brace_start : i + 1]

        return None

    # ------------------------------------------------------------------
    # 删除经验
    # ------------------------------------------------------------------

    def delete_experience(self, episode_id: int) -> bool:
        """删除某个 episode 对应的语义经验"""
        try:
            from app.core.milvus_client import milvus_manager

            milvus_manager.connect()
            collection = milvus_manager.get_collection()
            doc_id = f"{EXPERIENCE_ID_PREFIX}{episode_id}"
            collection.delete(f'id == "{doc_id}"')
            collection.flush()
            logger.info(f"L4 经验已删除: {doc_id}")
            return True
        except Exception as e:
            logger.warning(f"L4 经验删除失败: {e}")
            return False


# 全局单例 (延迟创建, 避免未配置 API Key 时启动失败)
_semantic_extractor: SemanticExtractor | None = None


def get_semantic_extractor(model_name: str = "qwen-plus") -> SemanticExtractor:
    """获取全局 SemanticExtractor 单例"""
    global _semantic_extractor
    if _semantic_extractor is None:
        _semantic_extractor = SemanticExtractor(model_name=model_name)
    return _semantic_extractor
