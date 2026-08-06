"""
RAG 系统评估模块
==================
基于 LLM-as-Judge 方法评估 RAG 系统质量，不依赖 OpenAI，
使用 DashScope (Tongyi/Qwen) 作为评估 LLM。

评估指标:
  1. Faithfulness (忠实度)      — 回答是否忠实于检索上下文
  2. Answer Relevancy (答案相关性) — 回答与问题的相关程度
  3. Context Precision (上下文精确率) — 检索结果的精确度
  4. Context Recall (上下文召回率)   — 检索结果覆盖问题的程度
"""

from .evaluator import RAGEvaluator, EvaluationResult
from .test_questions import TEST_QUESTIONS

__all__ = ["RAGEvaluator", "EvaluationResult", "TEST_QUESTIONS"]
