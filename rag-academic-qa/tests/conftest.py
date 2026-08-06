"""
pytest 全局 fixtures
=====================
为测试提供项目根目录路径和公共 mock 数据。
"""

import os
import sys

import pytest

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """提供一组模拟的 Document 对象（模拟 PDF 分页结果）。"""
    return [
        Document(
            page_content="Image restoration aims to recover high-quality images from degraded observations.",
            metadata={"source": "/data/papers/restormer.pdf", "file_name": "restormer.pdf", "page": 0, "page_label": 1},
        ),
        Document(
            page_content="The transformer architecture has shown great success in image processing tasks.",
            metadata={"source": "/data/papers/restormer.pdf", "file_name": "restormer.pdf", "page": 1, "page_label": 2},
        ),
        Document(
            page_content="SwinIR builds upon the Swin Transformer for image super-resolution.",
            metadata={"source": "/data/papers/swinir.pdf", "file_name": "swinir.pdf", "page": 0, "page_label": 1},
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Document]:
    """提供分块后的 Document 列表（模拟 chunk 结果）。"""
    return [
        Document(
            page_content="Restormer is a transformer-based model for image restoration. "
                         "It uses gated depthwise convolution for multi-scale aggregation.",
            metadata={"source": "/data/papers/restormer.pdf", "file_name": "restormer.pdf", "page": 0, "page_label": 1},
        ),
        Document(
            page_content="The key innovation of Restormer is the use of transposed attention blocks "
                         "that compute attention across feature channels rather than spatial dimensions.",
            metadata={"source": "/data/papers/restormer.pdf", "file_name": "restormer.pdf", "page": 1, "page_label": 2},
        ),
    ]
