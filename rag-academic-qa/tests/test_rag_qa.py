"""
rag_qa 模块单元测试
=====================
测试 format_context、extract_citations 等纯逻辑方法。
不依赖真实 LLM API（通过 mock 构造 RAGQA 实例）。
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from modules.rag_qa import (
    ANSWER_STYLE_GUIDE,
    FALLBACK_SYSTEM_PROMPT,
    HUMAN_PROMPT,
    RAGQA,
    SYSTEM_PROMPT,
)


class TestPromptTemplates:
    """测试 Prompt 模板内容。"""

    def test_system_prompt_exists(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_system_prompt_has_rules(self):
        assert "回答规则" in SYSTEM_PROMPT
        assert "检索到的文献上下文" in SYSTEM_PROMPT
        assert "[片段N]" in SYSTEM_PROMPT
        assert "结论 / 依据 / 小结" in SYSTEM_PROMPT

    def test_answer_style_guide_is_present(self):
        assert "输出要求" in ANSWER_STYLE_GUIDE
        assert "不要在正文末尾重复输出完整引用清单" in ANSWER_STYLE_GUIDE

    def test_human_prompt_has_placeholders(self):
        assert "{context}" in HUMAN_PROMPT
        assert "{question}" in HUMAN_PROMPT
        assert "固定结构" in HUMAN_PROMPT

    def test_fallback_prompt_has_style_guardrails(self):
        assert "通用知识回答" in FALLBACK_SYSTEM_PROMPT
        assert "结论 / 依据 / 小结" in FALLBACK_SYSTEM_PROMPT


class TestFormatContext:
    """测试 format_context 方法。"""

    @pytest.fixture
    def rag_qa_instance(self):
        """用 mock 创建 RAGQA 实例（不初始化真实 LLM）。"""
        with patch.object(RAGQA, "__init__", return_value=None):
            instance = RAGQA.__new__(RAGQA)
            return instance

    def test_format_returns_string(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        result = rag_qa_instance.format_context(docs_with_scores)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_contains_fragment_ids(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        result = rag_qa_instance.format_context(docs_with_scores)
        assert "[片段1]" in result
        assert "[片段2]" in result

    def test_format_contains_file_names(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        result = rag_qa_instance.format_context(docs_with_scores)
        assert "restormer.pdf" in result

    def test_format_contains_page_numbers(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        result = rag_qa_instance.format_context(docs_with_scores)
        assert "页码" in result

    def test_format_contains_relevance_scores(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        result = rag_qa_instance.format_context(docs_with_scores)
        assert "相关性" in result
        assert "0.8500" in result

    def test_format_empty_list(self, rag_qa_instance):
        result = rag_qa_instance.format_context([])
        assert result == ""


class TestExtractCitations:
    """测试 extract_citations 方法。"""

    @pytest.fixture
    def rag_qa_instance(self):
        with patch.object(RAGQA, "__init__", return_value=None):
            instance = RAGQA.__new__(RAGQA)
            return instance

    def test_returns_list_of_dicts(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        citations = rag_qa_instance.extract_citations(docs_with_scores)
        assert isinstance(citations, list)
        assert len(citations) == len(sample_chunks)
        for cite in citations:
            assert isinstance(cite, dict)

    def test_citation_has_required_keys(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        citations = rag_qa_instance.extract_citations(docs_with_scores)
        for cite in citations:
            assert "fragment_id" in cite
            assert "file_name" in cite
            assert "page" in cite
            assert "content" in cite
            assert "relevance_score" in cite

    def test_citation_fragment_ids_sequential(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        citations = rag_qa_instance.extract_citations(docs_with_scores)
        for i, cite in enumerate(citations, 1):
            assert cite["fragment_id"] == i

    def test_citation_file_names_correct(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85) for doc in sample_chunks]
        citations = rag_qa_instance.extract_citations(docs_with_scores)
        assert citations[0]["file_name"] == "restormer.pdf"

    def test_citation_score_rounded(self, rag_qa_instance, sample_chunks):
        docs_with_scores = [(doc, 0.85123456) for doc in sample_chunks]
        citations = rag_qa_instance.extract_citations(docs_with_scores)
        assert citations[0]["relevance_score"] == round(0.85123456, 4)

    def test_citations_empty_list(self, rag_qa_instance):
        citations = rag_qa_instance.extract_citations([])
        assert citations == []
