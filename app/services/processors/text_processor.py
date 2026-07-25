"""纯文本文档处理器 - 单阶段递归字符分片

分片流程:
1. 直接读取 UTF-8 编码文本
2. 使用 RecursiveCharacterTextSplitter 按字符递归分割
   - chunk_size = 配置值 × 2（默认 1600 字符）
   - overlap = 配置值（默认 100 字符）

依赖: 无外部依赖，纯 Python + LangChain
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.services.processors.base import DocumentProcessor


class TextProcessor(DocumentProcessor):
    """纯文本文档处理器"""

    supported_extensions = {"txt"}

    def __init__(self):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            f"TextProcessor 初始化完成, chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def extract_text(self, file_path: str) -> str:
        """读取纯文本文件内容"""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        content = path.read_text(encoding="utf-8")
        logger.info(f"文本提取完成: {file_path}, 长度: {len(content)} 字符")
        return content

    def split(self, content: str, file_path: str = "") -> List[Document]:
        """
        纯文本单阶段分片:

        直接使用 RecursiveCharacterTextSplitter 将文本按字符递归分割。
        分割器按照 ["\n\n", "\n", " ", ""] 的优先级查找分割点，
        优先在段落边界（双换行）处切割，其次是单换行，最后是空格。
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        docs = self.text_splitter.create_documents(
            texts=[content],
            metadatas=[self._build_metadata(file_path)],
        )
        logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
        return docs
