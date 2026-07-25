"""Markdown 文档处理器 - 三阶段分片策略

分片流程:
1. 按标题（#、##）分割，保留章节结构
2. 使用 RecursiveCharacterTextSplitter 对大块进行二次分割
3. 合并小于 300 字符的碎片到相邻块

依赖: 无外部依赖，使用 LangChain 内置的 MarkdownHeaderTextSplitter
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from loguru import logger

from app.services.processors.base import DocumentProcessor


class MarkdownProcessor(DocumentProcessor):
    """Markdown 文档处理器"""

    supported_extensions = {"md", "markdown"}

    def __init__(self):
        super().__init__()
        # 标题分割器：只按一级和二级标题分割，避免过度碎片化
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
            ],
            strip_headers=False,  # 保留标题在内容中
        )
        # 递归字符分割器（二次分割，使用更大的 chunk_size）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            f"MarkdownProcessor 初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, overlap={self.chunk_overlap}"
        )

    def extract_text(self, file_path: str) -> str:
        """读取 Markdown 文件的原始文本"""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        content = path.read_text(encoding="utf-8")
        logger.info(f"Markdown 文本提取完成: {file_path}, 长度: {len(content)} 字符")
        return content

    def split(self, content: str, file_path: str = "") -> List[Document]:
        """
        Markdown 三阶段分片:

        阶段1: MarkdownHeaderTextSplitter 按 #/## 标题分割
        阶段2: RecursiveCharacterTextSplitter 对超长块做二次分割
        阶段3: 合并 <300 字符的小片段到相邻块

        这种策略保留了文档的章节结构，同时控制每块的大小。
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # 阶段1: 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 阶段2: 按大小进一步分割
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # 阶段3: 合并太小的分片 (< 300字符)
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件元数据
            for doc in final_docs:
                doc.metadata.update(self._build_metadata(file_path))

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise
