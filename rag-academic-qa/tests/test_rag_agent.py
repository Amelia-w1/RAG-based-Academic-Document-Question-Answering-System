"""
RAGAgent（LangGraph 多智能体）单元测试
========================================
使用 FakeChatModel 与 MonkeyPatch 的向量库，在完全离线（无 API Key / 无索引）
的情况下验证：
  - 图能正常编译、RAGAgent 可构造
  - QueryAgent 策略选择 / JudgeAgent 条件分支（有结果→generate，无结果→fallback）
  - retrieve_plan 正确产出 citations
  - generate_stream 能逐 token 产出（流式契约）
  - ask 返回结构正确的 dict
"""

import os
import sys

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# 确保项目根在 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from unittest.mock import patch


@pytest.fixture
def fake_llm():
    """可顺序消费的假大模型：先 judge 再 generate，可应对多次调用。"""
    return FakeListChatModel(responses=[
        "sufficient",
        "[片段1] 这是基于文献的测试回答。",
        "[片段1] 这是基于文献的测试回答（兜底）。",
        "sufficient",
    ])


def _make_agent(fake_llm, docs, fallback_enabled=False):
    """构造 RAGAgent，并用假 LLM 与假检索替换真实依赖。"""
    with patch("modules.rag_qa.ChatTongyi", return_value=fake_llm), \
         patch("modules.advanced_retrieval.ChatTongyi", return_value=fake_llm):
        from modules.rag_agent import RAGAgent
        from modules.vector_store import VectorStore

        vs = VectorStore()  # 构造不会触发网络

        def fake_retrieve(self, query, k=None, top_n=None, file_filter=None):
            return docs

        VectorStore.retrieve = fake_retrieve
        return RAGAgent(vs, strategy_name="vector", fallback_enabled=fallback_enabled)


def _sample_docs():
    doc = Document(
        page_content="图像复原旨在从退化观测中恢复清晰图像。",
        metadata={"file_name": "a.pdf", "page": 1, "source": "a.pdf"},
    )
    return [(doc, 0.9)]


def test_retrieve_plan_with_docs(fake_llm):
    agent = _make_agent(fake_llm, _sample_docs())
    plan = agent.retrieve_plan("什么是图像复原？")
    assert plan["docs"], "有检索结果时 docs 不应为空"
    assert plan["citations"], "应提取到引用"
    assert plan["retrieval_decision"] == "generate"
    assert plan["strategy"] == "vector"


def test_retrieve_plan_no_docs_fallback_off(fake_llm):
    agent = _make_agent(fake_llm, [])  # 无结果且未开启兜底
    plan = agent.retrieve_plan("未知问题？")
    assert plan["docs"] == []
    assert plan["retrieval_decision"] == "fallback"
    assert plan["fallback"] is False


def test_retrieve_plan_no_docs_fallback_on(fake_llm):
    agent = _make_agent(fake_llm, [], fallback_enabled=True)
    plan = agent.retrieve_plan("未知问题？")
    assert plan["docs"] == []
    assert plan["fallback"] is True


def test_generate_stream_and_ask(fake_llm):
    agent = _make_agent(fake_llm, _sample_docs())
    plan = agent.retrieve_plan("什么是图像复原？")
    tokens = list(agent.generate_stream("什么是图像复原？", plan))
    full = "".join(tokens)
    assert full, "流式输出不应为空"

    result = agent.ask("什么是图像复原？")
    assert result["answer"], "ask 应返回回答"
    assert result["citations"], "ask 应返回引用"
    assert result["retrieval_decision"] == "generate"
    assert "trace" in result and result["trace"]


def test_strategy_selection_hybrid(fake_llm):
    """含比较关系的问题应被 QueryAgent 选为 hybrid 策略。"""
    agent = _make_agent(fake_llm, _sample_docs())
    plan = agent.retrieve_plan("CNN 与 Transformer 在图像复原上的区别？")
    assert plan["strategy"] == "hybrid"


def test_generate_stream_with_history_returns_strings(fake_llm):
    """注入历史时 _select_chain 必须返回完整链，流式输出为字符串而非 ChatPromptValue。"""
    agent = _make_agent(fake_llm, _sample_docs())
    plan = agent.retrieve_plan("什么是图像复原？")
    plan["history_text"] = "[用户] 之前的问题\n[助手] 之前的回答"
    tokens = list(agent.generate_stream("什么是图像复原？", plan))
    assert tokens, "流式输出不应为空"
    for token in tokens:
        assert isinstance(token, str), f"流式 token 必须是 str，得到 {type(token).__name__}"


def test_generate_stream_fallback_returns_strings(fake_llm):
    """兜底模式时 _select_chain 必须返回完整链，流式输出为字符串而非 ChatPromptValue。"""
    agent = _make_agent(fake_llm, _sample_docs(), fallback_enabled=True)
    plan = agent.retrieve_plan("什么是图像复原？")
    # 强制让 JudgeAgent 走兜底分支：无检索结果但兜底开启
    plan["docs"] = []
    plan["fallback"] = True
    tokens = list(agent.generate_stream("什么是图像复原？", plan))
    assert tokens, "流式输出不应为空"
    for token in tokens:
        assert isinstance(token, str), f"流式 token 必须是 str，得到 {type(token).__name__}"
