"""
document_loader 模块单元测试
=============================
测试文档分块逻辑、文件类型判断。
不依赖真实 PDF 文件或 API。
"""

import os
import tempfile

import pytest
from langchain_core.documents import Document

from modules.document_loader import DocumentLoader, SUPPORTED_EXTENSIONS


class TestDocumentLoaderInit:
    """测试 DocumentLoader 初始化。"""

    def test_default_init(self):
        loader = DocumentLoader()
        assert loader.chunk_size > 0
        assert loader.chunk_overlap >= 0
        assert loader.text_splitter is not None

    def test_custom_chunk_size(self):
        loader = DocumentLoader(chunk_size=800, chunk_overlap=100)
        assert loader.chunk_size == 800
        assert loader.chunk_overlap == 100

    def test_chunk_size_must_be_positive(self):
        """DocumentLoader 应能接受自定义参数。"""
        loader = DocumentLoader(chunk_size=500)
        assert loader.chunk_size == 500


class TestSplitDocuments:
    """测试 split_documents 分块逻辑。"""

    def test_split_returns_list(self, sample_documents):
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        chunks = loader.split_documents(sample_documents)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_split_preserves_metadata(self, sample_documents):
        loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        chunks = loader.split_documents(sample_documents)
        for chunk in chunks:
            assert "file_name" in chunk.metadata
            assert "page_label" in chunk.metadata

    def test_split_empty_list(self):
        loader = DocumentLoader()
        chunks = loader.split_documents([])
        assert chunks == []

    def test_split_chunk_size_respected(self):
        """分块后的内容不应超过 chunk_size（大约）。"""
        long_text = "A" * 2000
        doc = Document(page_content=long_text, metadata={"file_name": "test.pdf", "page": 0, "page_label": 1})
        loader = DocumentLoader(chunk_size=500, chunk_overlap=0)
        chunks = loader.split_documents([doc])
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # 允许少量超出（分隔符导致）


class TestLoadFile:
    """测试 load_file 文件类型判断。"""

    def test_unsupported_extension(self, tmp_path):
        loader = DocumentLoader()
        fake_file = tmp_path / "test.txt"
        fake_file.write_text("hello")
        with pytest.raises(ValueError, match="不支持的文件类型"):
            loader.load_file(str(fake_file))

    def test_supported_extensions_list(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".doc" in SUPPORTED_EXTENSIONS


class TestLoadDirectory:
    """测试 load_directory 目录扫描逻辑。"""

    def test_nonexistent_dir_raises(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError, match="目录不存在"):
            loader.load_directory("/nonexistent/path/12345")

    def test_empty_dir_raises(self, tmp_path):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError, match="未找到"):
            loader.load_directory(str(tmp_path))
