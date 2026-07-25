"""文档处理器包 - 支持多种文件类型的提取和分片"""

from app.services.processors.base import DocumentProcessor, processor_registry
from app.services.processors.text_processor import TextProcessor
from app.services.processors.markdown_processor import MarkdownProcessor
from app.services.processors.pdf_processor import PDFProcessor
from app.services.processors.word_processor import WordProcessor

__all__ = [
    "DocumentProcessor",
    "processor_registry",
    "TextProcessor",
    "MarkdownProcessor",
    "PDFProcessor",
    "WordProcessor",
]
