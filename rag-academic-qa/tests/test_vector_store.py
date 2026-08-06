"""
vector_store 模块单元测试
==========================
测试 has_index、rerank fallback 逻辑。
不依赖真实 API（使用 mock 或已禁用 rerank 的状态）。
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
from langchain_core.documents import Document

from modules.vector_store import VectorStore


class TestHasIndex:
    """测试 has_index 静态方法。"""

    def test_has_index_nonexistent_path(self):
        assert VectorStore.has_index("/nonexistent/path/12345") is False

    def test_has_index_empty_dir(self, tmp_path):
        assert VectorStore.has_index(str(tmp_path)) is False

    def test_has_index_with_index_file(self, tmp_path):
        # 创建假的 index.faiss 文件
        (tmp_path / "index.faiss").write_bytes(b"fake_index")
        assert VectorStore.has_index(str(tmp_path)) is True

    def test_has_index_default_path(self):
        """默认路径的测试不依赖特定结果，只确保不报错。"""
        result = VectorStore.has_index()
        assert isinstance(result, bool)


class TestRerankFallback:
    """测试 rerank 降级逻辑（不调用真实 API）。"""

    def test_rerank_disabled_returns_faiss_scores(self):
        """当 _rerank_disabled=True 时，应直接用 FAISS 分数排序。"""
        # 临时设置 _rerank_disabled
        original = VectorStore._rerank_disabled
        VectorStore._rerank_disabled = True

        try:
            vs = MagicMock()
            docs = [
                Document(page_content="doc A", metadata={"file_name": "a.pdf", "page_label": 1}),
                Document(page_content="doc B", metadata={"file_name": "b.pdf", "page_label": 1}),
                Document(page_content="doc C", metadata={"file_name": "c.pdf", "page_label": 1}),
            ]
            scores = [0.5, 0.9, 0.3]

            result = VectorStore.rerank(vs, "query", docs, top_n=2, scores=scores)

            assert len(result) == 2
            # 分数最高的应该排第一
            assert result[0][1] == 0.9
            assert result[1][1] == 0.5
        finally:
            VectorStore._rerank_disabled = original

    def test_rerank_disabled_no_scores(self):
        """当 _rerank_disabled=True 且无 scores 时，返回前 top_n 个。"""
        original = VectorStore._rerank_disabled
        VectorStore._rerank_disabled = True

        try:
            vs = MagicMock()
            docs = [
                Document(page_content="doc A", metadata={}),
                Document(page_content="doc B", metadata={}),
                Document(page_content="doc C", metadata={}),
            ]

            result = VectorStore.rerank(vs, "query", docs, top_n=2, scores=None)
            assert len(result) == 2
            assert result[0][1] == 0.0
        finally:
            VectorStore._rerank_disabled = original

    def test_rerank_empty_documents(self):
        """空文档列表应返回空列表。"""
        original = VectorStore._rerank_disabled
        VectorStore._rerank_disabled = True

        try:
            vs = MagicMock()
            result = VectorStore.rerank(vs, "query", [], top_n=5, scores=None)
            assert result == []
        finally:
            VectorStore._rerank_disabled = original

    def test_rerank_fewer_docs_than_top_n(self):
        """文档数少于 top_n 时应返回全部。"""
        original = VectorStore._rerank_disabled
        VectorStore._rerank_disabled = True

        try:
            vs = MagicMock()
            docs = [
                Document(page_content="only doc", metadata={}),
            ]
            scores = [0.8]

            result = VectorStore.rerank(vs, "query", docs, top_n=5, scores=scores)
            assert len(result) == 1
            assert result[0][1] == 0.8
        finally:
            VectorStore._rerank_disabled = original
