"""重排服务 - 基于百炼 DashScope TextReRank API 对召回文档进行精排

流程: 向量检索返回 N 篇候选 → 重排模型对每篇(query, doc)打分 → 按相关性降序返回

使用 DashScope TextReRank API (gte-rerank 模型):
    https://help.aliyun.com/zh/model-studio/rerank-model-api
"""

from typing import List, Tuple

from dashscope import TextReRank
from langchain_core.documents import Document
from loguru import logger

from app.config import config


class RerankService:
    """重排服务 - 对召回文档进行语义相关性精排"""

    def __init__(self):
        """初始化重排服务"""
        self._validate_config()

    def _validate_config(self):
        """验证重排所需配置"""
        if not config.dashscope_api_key:
            logger.warning("DASHSCOPE_API_KEY 未配置，重排服务将无法正常工作")

    def rerank(
        self, query: str, documents: List[str], top_n: int | None = None
    ) -> List[Tuple[int, float]]:
        """调用百炼重排 API，对文档列表按与 query 的相关性重新打分排序

        Args:
            query: 用户查询
            documents: 待重排的文档文本列表
            top_n: 返回前 N 个结果，None 表示返回全部（已按分数降序排列）

        Returns:
            List[Tuple[int, float]]: [(原始索引, 相关性分数), ...] 按分数降序

        Raises:
            RuntimeError: API 调用失败时直接抛出（不降级）
        """
        if not documents:
            logger.warning("重排收到空文档列表，跳过")
            return []

        logger.info(
            f"开始重排: query='{query[:80]}...', "
            f"文档数={len(documents)}, top_n={top_n}"
        )

        # 调用 DashScope TextReRank API
        response = TextReRank.call(
            model=config.rerank_model,
            query=query,
            documents=documents,
            top_n=top_n,
            return_documents=False,
            api_key=config.dashscope_api_key,
        )

        # 检查 API 响应状态 —— 失败不降级
        if response.status_code != 200:
            error_msg = (
                f"重排 API 调用失败: status_code={response.status_code}, "
                f"code={response.code}, message={response.message}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # 解析结果
        results = response.output.results
        if results is None:
            logger.warning("重排 API 返回空结果")
            return []

        # 构建 (原始索引, 分数) 列表，已是按分数降序
        reranked = [
            (item.index, item.relevance_score) for item in results
        ]

        logger.info(
            f"重排完成: 返回 {len(reranked)} 个结果, "
            f"最高分={reranked[0][1]:.4f}, 最低分={reranked[-1][1]:.4f}"
            if reranked else "重排完成: 无结果"
        )

        return reranked

    def rerank_documents(
        self, query: str, docs: List[Document], top_n: int | None = None
    ) -> List[Document]:
        """对 LangChain Document 列表进行重排，返回重新排序后的 Document 列表

        这是面向项目内部的主要接口，直接操作 Document 对象。

        Args:
            query: 用户查询
            docs: 待重排的 Document 列表
            top_n: 重排后保留前 N 个，None 表示保留全部

        Returns:
            List[Document]: 按重排分数降序排列的 Document 列表
        """
        if not docs:
            return []

        if len(docs) == 1:
            logger.debug("仅 1 篇文档，跳过高开销重排")
            return docs

        # 提取文档文本
        doc_texts = [doc.page_content for doc in docs]

        # 调用重排
        reranked = self.rerank(query, doc_texts, top_n=top_n)

        # 按重排结果重新排序 Document
        reordered_docs = [docs[idx] for idx, _ in reranked]
        return reordered_docs


# 全局单例，与项目其他 service 一致
rerank_service = RerankService()
