"""
文档加载模块
=================
负责批量读取 PDF / Word 论文、文本分块，保留页码与来源元数据。

核心功能:
  1. load_directory() — 批量加载目录下所有 PDF / Word
  2. split_documents() — 递归字符分块（可调切片大小 / 重叠长度）
  3. load_and_split()  — 一键: 加载 + 分块
"""

import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 兼容直接运行和包导入两种方式
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".doc")


class DocumentLoader:
    """PDF / Word 文档加载与分块器。"""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        """
        Args:
            chunk_size:    单个文本块最大字符数（默认取 Config.CHUNK_SIZE）
            chunk_overlap: 相邻块重叠字符数（默认取 Config.CHUNK_OVERLAP）
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

        # 递归字符分块器，优先按段落 → 换行 → 句号 → 空格 切分
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "!", "?", " ", ""],
            length_function=len,
        )

    def load_pdf(self, file_path: str) -> list[Document]:
        """
        加载单个 PDF 文件，按页拆分为 Document 对象列表。
        每个 Document 的 metadata 包含 source（文件路径）和 page（页码，从 0 开始）。

        Args:
            file_path: PDF 文件绝对路径
        Returns:
            List[Document]，每个元素对应一页
        """
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        # 补充分区元数据（方便后续溯源）
        filename = os.path.basename(file_path)
        for page in pages:
            page.metadata["file_name"] = filename
            page.metadata["page_label"] = page.metadata.get("page", 0) + 1  # 页码从 1 开始显示
        return pages

    def load_docx(self, file_path: str) -> list[Document]:
        """
        加载单个 Word 文档 (.docx)，按段落拆分为 Document 对象列表。

        Args:
            file_path: Word 文件绝对路径
        Returns:
            List[Document]
        """
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(file_path)
        pages = loader.load()
        filename = os.path.basename(file_path)
        for page in pages:
            page.metadata["file_name"] = filename
            page.metadata["page_label"] = page.metadata.get("page", 0) + 1
        return pages

    def load_file(self, file_path: str) -> list[Document]:
        """根据扩展名自动选择加载器。"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self.load_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return self.load_docx(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")

    def load_directory(self, dir_path: str | None = None) -> list[Document]:
        """
        批量加载目录下所有 PDF / Word 文件。

        Args:
            dir_path: 文档所在目录（默认取 Config.PDF_DIR）
        Returns:
            List[Document]，所有页面的集合
        Raises:
            FileNotFoundError: 目录不存在或无支持的文件
        """
        dir_path = dir_path or Config.PDF_DIR
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"文档目录不存在: {dir_path}")

        doc_files = sorted(
            f for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
            and f.lower().endswith(SUPPORTED_EXTENSIONS)
        )
        if not doc_files:
            raise FileNotFoundError(f"目录中未找到 PDF / Word 文件: {dir_path}")

        all_pages = []
        for doc_file in doc_files:
            file_path = os.path.join(dir_path, doc_file)
            logger.info("加载文档: %s", doc_file)
            try:
                pages = self.load_file(file_path)
                all_pages.extend(pages)
                logger.info("  %s — %d 页", doc_file, len(pages))
            except Exception as e:
                logger.error("加载失败 %s: %s", doc_file, e)

        logger.info("共加载 %d 篇文档，%d 页", len(doc_files), len(all_pages))
        return all_pages

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        将文档列表按递归字符分块器切分为小块。

        Args:
            documents: Document 对象列表
        Returns:
            List[Document]，分块后的文档列表（元数据继承自原文档）
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info("分块: %d 页 -> %d 个文本块 (size=%d, overlap=%d)",
                     len(documents), len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks

    def load_and_split(self, dir_path: str | None = None) -> list[Document]:
        """
        一键完成: 加载目录 PDF → 文本分块。

        Args:
            dir_path: PDF 所在目录
        Returns:
            List[Document]，分块后的文档列表
        """
        logger.info("步骤 1/2: 加载文档 ...")
        pages = self.load_directory(dir_path)
        logger.info("步骤 2/2: 文本分块 ...")
        chunks = self.split_documents(pages)
        return chunks
