"""文档分割服务模块 - 基于处理器注册模式的多类型文档分割

架构说明:
- 每种文件类型有独立的 DocumentProcessor 处理器（见 processors/ 目录）
- DocumentSplitterService 是调度中心，根据文件扩展名匹配处理器
- 通过 ProcessorRegistry 管理扩展名 → 处理器的映射
- 新增文件类型只需：① 创建处理器类 → ② 注册到 registry

支持的文件类型及分片策略一览:

| 扩展名 | 处理器              | 文本提取           | 分片策略                                |
|--------|---------------------|--------------------|-----------------------------------------|
| .md    | MarkdownProcessor   | UTF-8 直接读取     | 标题分割 → 递归字符分割 → 合并小片段    |
| .txt   | TextProcessor       | UTF-8 直接读取     | 递归字符分割（单阶段）                  |
| .pdf   | PDFProcessor        | pymupdf 逐页提取   | 按页分割 → 递归字符分割 → 合并小片段    |
| .docx  | WordProcessor       | python-docx 逐段落 | 递归字符分割（单阶段）                  |

所有处理器统一使用 RecursiveCharacterTextSplitter:
- chunk_size: 配置值 × 2（默认 1600 字符）
- chunk_overlap: 配置值（默认 100 字符）
- 分割优先级: "\n\n" → "\n" → " " → ""（优先在段落边界切割）
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from loguru import logger

from app.services.processors import (
    MarkdownProcessor,
    PDFProcessor,
    TextProcessor,
    WordProcessor,
    processor_registry,
)


# --- 注册所有处理器 ---
# 新增文件类型时，在此处添加一行 register() 即可
processor_registry.register(TextProcessor)
processor_registry.register(MarkdownProcessor)
processor_registry.register(PDFProcessor)
processor_registry.register(WordProcessor)
# ---


class DocumentSplitterService:
    """文档分割服务 - 处理器调度中心

    根据文件扩展名自动匹配对应的处理器，执行文本提取和分片。
    保持了原有 API 的向后兼容性。
    """

    def __init__(self):
        self._supported_extensions = processor_registry.get_supported_extensions()
        logger.info(
            f"文档分割服务初始化完成, "
            f"支持类型: {self._supported_extensions}"
        )

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """
        智能分割文档 - 根据文件扩展名自动选择处理器

        这是主要入口。也支持直接传入文件路径，内部会读取文件内容。

        Args:
            content: 文档文本内容（如果 file_path 为非 md/txt 类型，
                     此参数可传空字符串，方法会从文件路径自动读取）
            file_path: 文件路径（用于判断文件类型和提取元数据）

        Returns:
            List[Document]: 文档分片列表
        """
        if not file_path:
            logger.warning("未提供文件路径，无法判断类型，回退到纯文本处理")
            return self.split_text(content, file_path)

        ext = Path(file_path).suffix.lower()
        processor = processor_registry.get_processor(ext)

        if processor is None:
            logger.warning(
                f"未找到扩展名 '{ext}' 的处理器，回退到纯文本处理"
            )
            return self.split_text(content, file_path)

        # 如果调用方已提供文本内容（如旧代码的 md/txt 流程），
        # 直接走 split() 避免重复读取文件；否则走完整 process()
        if content and content.strip():
            return processor.split(content, file_path)
        else:
            return processor.process(file_path)

    # ------------------------------------------------------------------
    # 向后兼容的便捷方法
    # ------------------------------------------------------------------

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """分割 Markdown 文档（兼容旧 API，委托给 MarkdownProcessor）"""
        processor = processor_registry.get_processor("md")
        if processor is None:
            return []
        return processor.split(content, file_path)

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """分割纯文本文档（兼容旧 API，委托给 TextProcessor）"""
        processor = processor_registry.get_processor("txt")
        if processor is None:
            return []
        return processor.split(content, file_path)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_supported_extensions(self) -> List[str]:
        """获取所有支持的文件扩展名"""
        return self._supported_extensions

    def get_processor_info(self) -> dict:
        """获取处理器信息（扩展名 → 类名）"""
        return processor_registry.get_processor_info()


# 全局单例
document_splitter_service = DocumentSplitterService()
