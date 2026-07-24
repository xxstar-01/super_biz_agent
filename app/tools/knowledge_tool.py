"""知识检索工具 - 从向量数据库中检索相关信息，支持重排精排"""

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.rerank_service import rerank_service
from app.services.vector_store_manager import vector_store_manager


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    检索流程:
        1. 向量检索召回候选文档（数量由 rerank_retrieve_n 或 rag_top_k 控制）
        2. 如果启用重排，调用百炼重排模型对候选文档精排
        3. 取最终需要的文档数（rerank_top_k），格式化返回

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 召回阶段取多少文档：启用重排时多召回候选，否则直接用 rag_top_k
        retrieve_k = config.rerank_retrieve_n if config.rerank_enabled else config.rag_top_k
        logger.debug(f"召回候选文档数: {retrieve_k}")

        # 1. 从向量存储中检索候选文档
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": retrieve_k}
        )

        docs = retriever.invoke(query)

        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        logger.info(f"向量检索召回 {len(docs)} 个候选文档")

        # 2. 重排：用百炼重排模型对候选文档精排
        if config.rerank_enabled and len(docs) > 1:
            logger.info("启用重排，调用百炼重排模型...")
            docs = rerank_service.rerank_documents(query, docs)

        # 3. 截取最终需要的文档数
        final_k = min(config.rerank_top_k if config.rerank_enabled else config.rag_top_k, len(docs))
        docs = docs[:final_k]

        # 4. 格式化文档为上下文
        context = format_docs(docs)

        logger.info(f"最终返回 {len(docs)} 个文档（重排={'启用' if config.rerank_enabled else '关闭'}）")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        raise


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")
        
        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        
        header_str = " > ".join(headers) if headers else ""
        
        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"
        
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)
