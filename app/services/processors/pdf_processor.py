"""PDF 文档处理器 - 按页提取 + 递归字符分片

分片流程:
1. 使用 pymupdf (fitz) 逐页提取文本
2. 按页边界作为天然断点
3. 对每页内容使用 RecursiveCharacterTextSplitter 二次分割
4. 合并小于 300 字符的碎片

依赖: pymupdf (>=1.24.0) - MIT 协议，业界最快的 PDF 文本提取库之一

选型理由（与其他方案对比）:
- pymupdf: 速度最快，文本顺序准确，支持复杂排版，MIT 协议
- pdfplumber: 表格提取更强，但纯文本场景速度慢 5-10x
- pypdf: 纯 Python，零依赖但文本提取质量一般
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.services.processors.base import DocumentProcessor


class PDFProcessor(DocumentProcessor):
    """PDF 文档处理器"""

    supported_extensions = {"pdf"}

    def __init__(self):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info(
            f"PDFProcessor 初始化完成, chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def extract_text(self, file_path: str) -> str:
        """使用 pymupdf 逐页提取 PDF 文本

        每页文本后追加换行，保留页边界信息。
        对提取失败的页面给出警告但不中断。
        """
        import fitz  # pymupdf

        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        doc = fitz.open(str(path))
        pages_text = []
        failed_pages = []

        try:
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text("text")  # 提取纯文本（非 HTML/字典模式）
                    if text and text.strip():
                        pages_text.append(text.strip())
                except Exception as e:
                    failed_pages.append(page_num + 1)
                    logger.warning(f"PDF 第 {page_num + 1} 页提取失败: {e}")

            if failed_pages:
                logger.warning(
                    f"PDF 部分页面提取失败: {file_path}, "
                    f"失败页码: {failed_pages}, 总页数: {len(doc)}"
                )
        finally:
            doc.close()

        full_text = "\n\n".join(pages_text)
        logger.info(
            f"PDF 文本提取完成: {file_path}, "
            f"总页数: {len(doc)}, 成功: {len(pages_text)}, "
            f"失败: {len(failed_pages)}, 文本长度: {len(full_text)} 字符"
        )
        return full_text

    def split(self, content: str, file_path: str = "") -> List[Document]:
        """
        PDF 分片策略:

        1. 按 "\\n\\n"（页边界）预分割
        2. 对每页内容用 RecursiveCharacterTextSplitter 再次分割
        3. 合并 <300 字符的小片段

        页边界给分割器提供了天然的语义断点，
        避免了在段落中间生硬切割的问题。
        """
        if not content or not content.strip():
            logger.warning(f"PDF 内容为空: {file_path}")
            return []

        # 按页分割（每页之间已经是 \n\n，split 即可）
        pages = content.split("\n\n")
        all_docs = []

        for i, page_text in enumerate(pages):
            if not page_text or not page_text.strip():
                continue

            # 对每页做二次分割
            chunks = self.text_splitter.create_documents(
                texts=[page_text.strip()],
                metadatas=[{
                    "_source": Path(file_path).as_posix(),
                    "_extension": ".pdf",
                    "_file_name": Path(file_path).name,
                    "_page": i + 1,  # 1-based 页码
                }],
            )
            all_docs.extend(chunks)

        # 合并太小的分片
        final_docs = self._merge_small_chunks(all_docs, min_size=300)

        logger.info(
            f"PDF 分割完成: {file_path} -> "
            f"{len(pages)} 页 -> {len(final_docs)} 个分片"
        )
        return final_docs
