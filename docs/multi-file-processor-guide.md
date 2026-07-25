# 知识库多文件类型处理器实现指南

> 版本: 1.2.1 → 1.3.0  
> 日期: 2026-07-25  
> 改动范围: 文档上传 → 分片 → 向量化全链路

---

## 一、改了什么

### 之前（只支持 2 种文件）

```
用户上传 .md / .txt → UTF-8 读取 → 硬编码分片逻辑 → 向量化
```

- `ALLOWED_EXTENSIONS = ["txt", "md"]`
- 分片逻辑写死在 `DocumentSplitterService` 里
- 新增文件类型需要改多处代码

### 之后（支持 4 种文件，可扩展）

```
用户上传 .md/.txt/.pdf/.docx → 按扩展名匹配处理器 → 处理器提取文本 →
处理器按类型分片 → 向量化
```

- `ALLOWED_EXTENSIONS = ["txt", "md", "pdf", "docx"]`
- **处理器注册模式**：每种文件类型一个 Processor 类
- 新增文件类型只需 2 步：写处理器 → 注册

---

## 二、文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `pyproject.toml` | 修改 | 新增 `pymupdf>=1.24.0`、`python-docx>=1.1.0` |
| `app/services/processors/__init__.py` | **新增** | 处理器包导出 |
| `app/services/processors/base.py` | **新增** | `DocumentProcessor` 基类 + `ProcessorRegistry` 注册中心 |
| `app/services/processors/text_processor.py` | **新增** | `.txt` 处理器 |
| `app/services/processors/markdown_processor.py` | **新增** | `.md` 处理器（从旧代码迁移） |
| `app/services/processors/pdf_processor.py` | **新增** | `.pdf` 处理器 |
| `app/services/processors/word_processor.py` | **新增** | `.docx` 处理器 |
| `app/services/document_splitter_service.py` | **重写** | 改为调度中心，委托给各处理器 |
| `app/api/file.py` | 修改 | `ALLOWED_EXTENSIONS` 扩展 |
| `app/services/vector_index_service.py` | 修改 | glob 匹配扩展、不再硬编码 UTF-8 读取 |
| `static/index.html` | 修改 | `<input accept>` 加 `.pdf,.docx` |
| `static/app.js` | 修改 | `allowedExtensions` 扩展、错误提示更新 |
| `docs/multi-file-processor-guide.md` | **新增** | 本文档 |

---

## 三、架构设计

### 3.1 处理器注册模式

```
                    ┌─────────────────────────┐
                    │  DocumentSplitterService │  ← 调度中心（旧类，接口不变）
                    │  split_document()        │
                    └─────────────┬───────────┘
                                  │ 按扩展名查找处理器
                                  ▼
                    ┌─────────────────────────┐
                    │    ProcessorRegistry     │  ← 全局注册中心
                    │    {ext → ProcessorCls}  │
                    └─────────────┬───────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
  ┌──────────────┐       ┌──────────────┐        ┌──────────────┐
  │ TextProcessor│       │ PDFProcessor │        │WordProcessor │
  │ (.txt)       │       │ (.pdf)       │        │ (.docx)      │
  └──────────────┘       └──────────────┘        └──────────────┘
          │
          ▼
  ┌──────────────┐
  │MarkdownProc..│
  │ (.md)        │
  └──────────────┘
```

### 3.2 基类接口

```python
class DocumentProcessor(ABC):
    supported_extensions: set[str] = set()  # 子类声明支持哪些扩展名

    @abstractmethod
    def extract_text(self, file_path: str) -> str: ...
        # 从文件中提取文本内容

    def split(self, content: str, file_path: str) -> List[Document]: ...
        # 将文本分片（子类可覆盖）

    def process(self, file_path: str) -> List[Document]: ...
        # 完整流程：提取 → 分片
```

每个处理器只需要实现 `extract_text()`，`split()` 有默认实现。

### 3.3 如何添加新文件类型

只需 2 步：

**步骤 1**：创建处理器文件，例如 `app/services/processors/html_processor.py`：

```python
class HTMLProcessor(DocumentProcessor):
    supported_extensions = {"html", "htm"}

    def extract_text(self, file_path: str) -> str:
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text()
```

**步骤 2**：在 `document_splitter_service.py` 中注册：

```python
from app.services.processors.html_processor import HTMLProcessor
processor_registry.register(HTMLProcessor)
```

然后更新 `app/api/file.py` 的 `ALLOWED_EXTENSIONS` 即可。

---

## 四、各文件类型的分片策略

| 类型 | 扩展名 | 文本提取库 | 分片方式 |
|------|--------|-----------|---------|
| 纯文本 | `.txt` | 原生 UTF-8 读取 | 单阶段：`RecursiveCharacterTextSplitter`（chunk=1600, overlap=100） |
| Markdown | `.md` `.markdown` | 原生 UTF-8 读取 | 三阶段：标题分割→递归字符分割→合并小片段 |
| PDF | `.pdf` | **pymupdf** (fitz) 逐页提取 | 按页分割→每页递归字符分割→合并小片段（<300字符） |
| Word | `.docx` | **python-docx** 逐段落+表格提取 | 单阶段：`RecursiveCharacterTextSplitter`（chunk=1600, overlap=100） |

### 4.1 Markdown 分片（三阶段）

```
原始 Markdown
  │
  ▼
阶段1: MarkdownHeaderTextSplitter
  按 # (h1) 和 ## (h2) 标题分割
  保留标题文本在内容中（strip_headers=False）
  │
  ▼
阶段2: RecursiveCharacterTextSplitter
  chunk_size=1600, overlap=100
  对超长章节做二次切割
  分割优先级: \n\n → \n → " " → ""
  │
  ▼
阶段3: _merge_small_chunks(min_size=300)
  遍历所有分片
  如果当前块 < 300字符 且 合并后 < 1600字符 → 向前合并
  否则保留独立
```

### 4.2 PDF 分片

```
原始 PDF
  │
  ▼ pymupdf.open() → 逐页 page.get_text("text")
按页提取为纯文本，页间用 \n\n 分隔
  │
  ▼ 按 "\n\n" 分割（恢复页边界）
每页作为独立单元
  │
  ▼ 每页用 RecursiveCharacterTextSplitter(chunk=1600, overlap=100)
  │
  ▼ _merge_small_chunks(min_size=300)
合并过小的跨页碎片
```

每块元数据中包含 `_page` 字段（页码，1-based），便于溯源。

### 4.3 Word 分片

```
原始 .docx
  │
  ▼ python-docx: 遍历 doc.paragraphs → 获取段落文本
  │               遍历 doc.tables → 行内单元格用 " | " 连接
段落间用 \n\n 分隔
  │
  ▼ RecursiveCharacterTextSplitter(chunk=1600, overlap=100)
  优先在段落边界（\n\n）切割
```

---

## 五、关键代码解读

### 5.1 注册中心 (`base.py`)

```python
class ProcessorRegistry:
    def __init__(self):
        self._processors: Dict[str, Type[DocumentProcessor]] = {}  # ext → 类
        self._instances: Dict[str, DocumentProcessor] = {}         # ext → 实例（懒加载）

    def register(self, processor_cls: Type[DocumentProcessor]) -> None:
        for ext in processor_cls.supported_extensions:
            self._processors[ext.lower()] = processor_cls

    def get_processor(self, extension: str) -> Optional[DocumentProcessor]:
        ext = extension.lstrip(".").lower()
        cls = self._processors.get(ext)
        if cls is None:
            return None
        if ext not in self._instances:
            self._instances[ext] = cls()  # 懒加载：首次访问时才实例化
        return self._instances[ext]
```

设计要点：
- **类级注册**：注册时存类而非实例，避免初始化所有处理器
- **懒加载**：`get_processor()` 首次调用时才创建实例
- **多扩展名**：一个处理器可声明多个扩展名（如 `{"md", "markdown"}`）

### 5.2 调度中心 (`document_splitter_service.py`)

```python
def split_document(self, content: str, file_path: str = "") -> List[Document]:
    ext = Path(file_path).suffix.lower()
    processor = processor_registry.get_processor(ext)

    if processor is None:
        return self.split_text(content, file_path)  # 未知类型降级为纯文本

    # 如果调用方已提供文本内容，直接 split 避免重复读取
    if content and content.strip():
        return processor.split(content, file_path)
    else:
        return processor.process(file_path)  # 完整流程: 提取 + 分片
```

调度逻辑：
1. 按文件扩展名匹配处理器
2. 匹配不到 → 降级为纯文本处理
3. 有内容 → 直接用 `split()`（兼容旧调用方式）
4. 无内容 → 走 `process()` 完整流程

### 5.3 索引服务变更 (`vector_index_service.py`)

**之前**（只处理文本）：
```python
content = path.read_text(encoding="utf-8")  # PDF/DOCX 会报错!
documents = document_splitter_service.split_document(content, normalized_path)
```

**之后**（委托给处理器）：
```python
documents = document_splitter_service.split_document("", normalized_path)
# 调度到 PDFProcessor/WordProcessor 时，内部用 pymupdf/python-docx 读取
```

`index_directory()` 也从硬编码 glob 改为动态获取：
```python
supported = document_splitter_service.get_supported_extensions()
for ext in supported:
    files.extend(dir_path.glob(f"*.{ext}"))
```

---

## 六、依赖选型理由

| 库 | 用途 | 选型理由 |
|----|------|---------|
| `pymupdf` (fitz) | PDF 文本提取 | 速度最快、文本顺序准确、支持复杂排版、MIT 协议 |
| `python-docx` | Word 文档读取 | 轻量纯 Python、API 简洁、事实标准、MIT 协议 |

备选方案对比：
- **pdfplumber** vs pymupdf：表格提取更强，但纯文本慢 5-10x
- **pypdf** vs pymupdf：纯 Python 零依赖，但文本提取质量一般
- **textract** vs python-docx：功能全但依赖过多（需要 LibreOffice）

---

## 七、安装与验证

```bash
# 安装新依赖
pip install pymupdf python-docx

# 验证导入
python -c "
from app.services.document_splitter_service import document_splitter_service
print('支持类型:', document_splitter_service.get_supported_extensions())
print('处理器:', document_splitter_service.get_processor_info())
"

# 验证 PDF 处理
python -c "
from app.services.processors import processor_registry
p = processor_registry.get_processor('pdf')
print(f'处理器: {p.__class__.__name__}')
# text = p.extract_text('test.pdf')
# docs = p.split(text, 'test.pdf')
"

# 验证 Word 处理
python -c "
from app.services.processors import processor_registry
p = processor_registry.get_processor('docx')
print(f'处理器: {p.__class__.__name__}')
"
```

---

## 八、配置说明

分片参数在 `.env` 或 `config.py` 中统一管理：

```
CHUNK_MAX_SIZE=800    # 基础 chunk 大小（处理器内部 ×2 使用，即 1600）
CHUNK_OVERLAP=100     # 块间重叠字符数
```

所有处理器共享这两个配置，通过基类 `DocumentProcessor.__init__()` 读取。

需要为特定文件类型设置不同的分片参数时，在对应处理器的 `__init__` 中覆盖即可：

```python
class PDFProcessor(DocumentProcessor):
    def __init__(self):
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # PDF 分片更小
            chunk_overlap=200,    # 重叠更多
            ...
        )
```
