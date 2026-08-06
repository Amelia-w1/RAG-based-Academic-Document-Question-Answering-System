"""
config 模块单元测试
====================
测试 Config 类的属性加载、校验逻辑。
"""

import os
import tempfile
from unittest.mock import patch

from config import Config


class TestConfigAttributes:
    """测试 Config 属性是否存在且类型正确。"""

    def test_api_key_exists(self):
        assert hasattr(Config, "DASHSCOPE_API_KEY")
        assert isinstance(Config.DASHSCOPE_API_KEY, str)

    def test_llm_model_default(self):
        assert Config.LLM_MODEL in ("qwen-turbo", "qwen-plus", "qwen-max")

    def test_embedding_model_default(self):
        assert "text-embedding" in Config.EMBEDDING_MODEL

    def test_chunk_size_is_positive_int(self):
        assert isinstance(Config.CHUNK_SIZE, int)
        assert Config.CHUNK_SIZE > 0

    def test_chunk_overlap_non_negative(self):
        assert isinstance(Config.CHUNK_OVERLAP, int)
        assert Config.CHUNK_OVERLAP >= 0

    def test_retrieval_k_positive(self):
        assert isinstance(Config.RETRIEVAL_K, int)
        assert Config.RETRIEVAL_K > 0

    def test_rerank_top_n_positive(self):
        assert isinstance(Config.RERANK_TOP_N, int)
        assert Config.RERANK_TOP_N > 0

    def test_temperature_range(self):
        assert isinstance(Config.TEMPERATURE, float)
        assert 0.0 <= Config.TEMPERATURE <= 2.0

    def test_pdf_dir_is_string(self):
        assert isinstance(Config.PDF_DIR, str)
        assert len(Config.PDF_DIR) > 0

    def test_faiss_index_dir_is_string(self):
        assert isinstance(Config.FAISS_INDEX_DIR, str)
        assert len(Config.FAISS_INDEX_DIR) > 0


class TestConfigValidate:
    """测试 Config.validate() 校验逻辑。"""

    def test_validate_returns_tuple(self):
        result = Config.validate()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    @patch.object(Config, "DASHSCOPE_API_KEY", "")
    def test_validate_no_api_key(self):
        ok, msg = Config.validate()
        assert ok is False
        assert "DASHSCOPE_API_KEY" in msg

    @patch.object(Config, "PDF_DIR", "/nonexistent/path/12345")
    def test_validate_bad_pdf_dir(self):
        ok, msg = Config.validate()
        assert ok is False
        assert "PDF" in msg or "目录" in msg

    @patch.object(Config, "DASHSCOPE_API_KEY", "sk-test-key")
    @patch.object(Config, "PDF_DIR", tempfile.gettempdir())
    def test_validate_success(self):
        ok, msg = Config.validate()
        assert ok is True
        assert "通过" in msg


class TestConfigDisplay:
    """测试 Config.display() 输出格式。"""

    def test_display_returns_string(self):
        result = Config.display()
        assert isinstance(result, str)

    def test_display_contains_key_fields(self):
        result = Config.display()
        assert "对话模型" in result
        assert "向量模型" in result
        assert "切片大小" in result
        assert "API Key" in result

    def test_display_hides_api_key(self):
        result = Config.display()
        # 不应该包含实际的 key 值
        assert "sk-" not in result or "已设置" in result
