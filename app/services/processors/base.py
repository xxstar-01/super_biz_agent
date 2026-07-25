"""文档处理器基类 - 定义统一接口和处理器注册中心"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import config


class DocumentProcessor(ABC):
    """文档处理器抽象基类

    每种文件类型继承此类，实现文本提取和分片逻辑。
    子类需要声明 supported_extensions 类属性。
    """

    # 子类必须覆盖：支持的文件扩展名集合（不含点，小写）
    supported_extensions: set[str] = set()

    def __init__(self):
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

    def can_handle(self, extension: str) -> bool:
        """检查此处理器是否可以处理给定扩展名"""
        return extension.lower() in self.supported_extensions

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """从文件中提取文本内容

        Args:
            file_path: 文件路径

        Returns:
            str: 提取的文本内容
        """
        ...

    def split(self, content: str, file_path: str = "") -> List[Document]:
        """将文本内容分割为文档块（子类可覆盖以定制分片策略）

        默认使用 RecursiveCharacterTextSplitter 进行分片。
        子类可以覆盖此方法实现自定义分片逻辑。

        Args:
            content: 文本内容
            file_path: 文件路径（用于元数据）

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文档内容为空: {file_path}")
            return []

        ext = Path(file_path).suffix if file_path else ""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents(
            texts=[content],
            metadatas=[self._build_metadata(file_path)],
        )
        logger.info(f"默认分片完成: {file_path} -> {len(docs)} 个分片")
        return docs

    def process(self, file_path: str) -> List[Document]:
        """完整处理流程：提取文本 → 分片 → 添加元数据

        Args:
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        content = self.extract_text(file_path)
        if not content or not content.strip():
            logger.warning(f"文件内容为空: {file_path}")
            return []
        return self.split(content, file_path)

    def _build_metadata(self, file_path: str) -> dict:
        """构建文档元数据"""
        path = Path(file_path)
        return {
            "_source": path.as_posix(),
            "_extension": path.suffix.lower(),
            "_file_name": path.name,
        }

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """合并太小的分片

        Args:
            documents: 文档列表
            min_size: 最小分片大小（字符数）

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
                current_doc.page_content += "\n\n" + doc.page_content
            else:
                merged_docs.append(current_doc)
                current_doc = doc

        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs


class ProcessorRegistry:
    """处理器注册中心 - 管理文件类型到处理器的映射

    使用方式:
        registry = ProcessorRegistry()
        registry.register(MarkdownProcessor)
        registry.register(TextProcessor)
        processor = registry.get_processor(".pdf")
        docs = processor.process("/path/to/file.pdf")
    """

    def __init__(self):
        self._processors: Dict[str, Type[DocumentProcessor]] = {}
        self._instances: Dict[str, DocumentProcessor] = {}

    def register(self, processor_cls: Type[DocumentProcessor]) -> None:
        """注册一个处理器类

        Args:
            processor_cls: 处理器类（非实例）
        """
        for ext in processor_cls.supported_extensions:
            ext_lower = ext.lower()
            self._processors[ext_lower] = processor_cls
            logger.info(f"注册文档处理器: .{ext_lower} -> {processor_cls.__name__}")

    def get_processor(self, extension: str) -> Optional[DocumentProcessor]:
        """根据扩展名获取处理器实例（懒加载）

        Args:
            extension: 文件扩展名（如 ".pdf" 或 "pdf"）

        Returns:
            DocumentProcessor 实例，未找到返回 None
        """
        ext = extension.lstrip(".").lower()
        processor_cls = self._processors.get(ext)
        if processor_cls is None:
            return None

        if ext not in self._instances:
            self._instances[ext] = processor_cls()

        return self._instances[ext]

    def get_supported_extensions(self) -> List[str]:
        """获取所有支持的扩展名列表"""
        return sorted(self._processors.keys())

    def get_processor_info(self) -> Dict[str, str]:
        """获取处理器信息（扩展名 → 处理器类名）"""
        return {ext: cls.__name__ for ext, cls in self._processors.items()}


# 全局注册中心
processor_registry = ProcessorRegistry()
