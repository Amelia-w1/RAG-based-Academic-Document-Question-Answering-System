"""
RAG 评估器
==================
对 RAG 系统进行全面评估，计算各项指标并生成报告。

核心功能:
  1. evaluate_single()  — 评估单个问题
  2. evaluate_batch()   — 批量评估测试问题集
  3. generate_report()  — 生成评估报告（文本 + JSON）

用法:
  python -m evaluation.run_evaluation
  或
  from evaluation import RAGEvaluator
  evaluator = RAGEvaluator(vs, qa)
  results = evaluator.evaluate_batch()
  report = evaluator.generate_report(results)
"""

from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SingleResult:
    """单个问题的评估结果。"""
    question: str
    category: str = ""
    answer: str = ""
    context: str = ""
    citations_count: int = 0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    keyword_coverage: float = 0.0  # 期望关键词覆盖率
    error: str = ""


@dataclass
class EvaluationResult:
    """整体评估结果。"""
    total_questions: int = 0
    success_count: int = 0
    fail_count: int = 0
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_keyword_coverage: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    results: list[SingleResult] = field(default_factory=list)
    config_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_questions": self.total_questions,
                "success_count": self.success_count,
                "fail_count": self.fail_count,
                "avg_faithfulness": round(self.avg_faithfulness, 4),
                "avg_answer_relevancy": round(self.avg_answer_relevancy, 4),
                "avg_context_precision": round(self.avg_context_precision, 4),
                "avg_context_recall": round(self.avg_context_recall, 4),
                "avg_keyword_coverage": round(self.avg_keyword_coverage, 4),
                "avg_retrieval_time": round(self.avg_retrieval_time, 3),
                "avg_generation_time": round(self.avg_generation_time, 3),
            },
            "config": self.config_info,
            "details": [asdict(r) for r in self.results],
        }


class RAGEvaluator:
    """
    RAG 系统评估器。

    使用 LLM-as-Judge 方法，通过 DashScope LLM 评估四个维度:
    忠实度、答案相关性、上下文精确率、上下文召回率。
    """

    def __init__(self, vector_store, rag_qa, strategy_name: str = "vector") -> None:
        """
        Args:
            vector_store:  VectorStore 实例（已加载索引）
            rag_qa:        RAGQA 实例
            strategy_name: 使用的检索策略名称
        """
        self.vs = vector_store
        self.qa = rag_qa
        self.strategy_name = strategy_name

        # 评估用 LLM（使用较低温度确保稳定性）
        from langchain_community.chat_models.tongyi import ChatTongyi
        self.eval_llm = ChatTongyi(
            model=Config.LLM_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
            temperature=0.0,  # 评估需要确定性输出
        )

        # 初始化指标
        from evaluation.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextPrecisionMetric,
            ContextRecallMetric,
        )
        self.metrics = {
            "faithfulness": FaithfulnessMetric(self.eval_llm),
            "answer_relevancy": AnswerRelevancyMetric(self.eval_llm),
            "context_precision": ContextPrecisionMetric(self.eval_llm),
            "context_recall": ContextRecallMetric(self.eval_llm),
        }

    def evaluate_single(self, question: str, expected_keywords: list[str] | None = None,
                        k: int | None = None, top_n: int | None = None,
                        file_filter: list[str] | None = None) -> SingleResult:
        """
        评估单个问题。

        Args:
            question:          测试问题
            expected_keywords: 期望关键词列表
            k:                 检索候选数
            top_n:             重排序保留数
            file_filter:       文件过滤
        Returns:
            SingleResult
        """
        result = SingleResult(question=question)
        logger.info("评估问题: %s", question)

        try:
            # Step 1: RAG 问答
            t0 = time.time()

            # 根据策略选择检索方式
            if self.strategy_name and self.strategy_name != "vector":
                try:
                    from modules.advanced_retrieval import get_strategy
                    strategy = get_strategy(self.strategy_name, self.vs)
                    docs_with_scores = strategy.retrieve(
                        question, k=k, top_n=top_n, file_filter=file_filter
                    )
                    context = self.qa.format_context(docs_with_scores)
                    answer = self.qa.chain.invoke({
                        "context": context, "question": question
                    })
                    citations = self.qa.extract_citations(docs_with_scores)
                    result.retrieval_time = time.time() - t0
                except Exception as e:
                    logger.warning("策略 %s 失败，回退标准检索: %s", self.strategy_name, e)
                    rag_result = self.qa.ask(question, k=k, top_n=top_n, file_filter=file_filter)
                    answer = rag_result["answer"]
                    context = rag_result["context"]
                    citations = rag_result["citations"]
                    result.retrieval_time = time.time() - t0
            else:
                rag_result = self.qa.ask(question, k=k, top_n=top_n, file_filter=file_filter)
                answer = rag_result["answer"]
                context = rag_result["context"]
                citations = rag_result["citations"]
                result.retrieval_time = time.time() - t0

            t1 = time.time()

            result.answer = answer
            result.context = context[:500]  # 截断存储
            result.citations_count = len(citations)
            result.generation_time = time.time() - t1

            # Step 2: 计算各项指标
            logger.info("  计算评估指标 ...")
            result.faithfulness = self.metrics["faithfulness"].score(
                question=question, answer=answer, context=context
            )
            result.answer_relevancy = self.metrics["answer_relevancy"].score(
                question=question, answer=answer
            )
            result.context_precision = self.metrics["context_precision"].score(
                question=question, context=context
            )
            result.context_recall = self.metrics["context_recall"].score(
                question=question, context=context, answer=answer
            )

            # Step 3: 关键词覆盖率
            if expected_keywords:
                answer_lower = answer.lower()
                covered = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
                result.keyword_coverage = covered / len(expected_keywords)

            logger.info(
                "  结果: 忠实度=%.2f, 相关性=%.2f, 精确率=%.2f, 召回率=%.2f, 关键词=%.2f",
                result.faithfulness, result.answer_relevancy,
                result.context_precision, result.context_recall,
                result.keyword_coverage,
            )

        except Exception as e:
            logger.error("评估失败: %s", e)
            result.error = str(e)

        return result

    def evaluate_batch(self, test_questions: list[dict] | None = None,
                       k: int | None = None, top_n: int | None = None,
                       file_filter: list[str] | None = None) -> EvaluationResult:
        """
        批量评估测试问题集。

        Args:
            test_questions: 测试问题列表（None 则使用默认）
            k:              检索候选数
            top_n:          重排序保留数
            file_filter:    文件过滤
        Returns:
            EvaluationResult
        """
        if test_questions is None:
            from evaluation.test_questions import TEST_QUESTIONS
            test_questions = TEST_QUESTIONS

        logger.info("开始批量评估，共 %d 个问题，策略: %s", len(test_questions), self.strategy_name)

        result = EvaluationResult(
            total_questions=len(test_questions),
            config_info={
                "llm_model": Config.LLM_MODEL,
                "embedding_model": Config.EMBEDDING_MODEL,
                "chunk_size": Config.CHUNK_SIZE,
                "chunk_overlap": Config.CHUNK_OVERLAP,
                "retrieval_k": k or Config.RETRIEVAL_K,
                "rerank_top_n": top_n or Config.RERANK_TOP_N,
                "temperature": Config.TEMPERATURE,
                "strategy": self.strategy_name,
            }
        )

        for i, tq in enumerate(test_questions):
            logger.info("[%d/%d] %s", i + 1, len(test_questions), tq["question"])
            sr = self.evaluate_single(
                question=tq["question"],
                expected_keywords=tq.get("expected_keywords"),
                k=k, top_n=top_n, file_filter=file_filter,
            )
            sr.category = tq.get("category", "")
            result.results.append(sr)

            if sr.error:
                result.fail_count += 1
            else:
                result.success_count += 1

        # 计算平均值
        successful = [r for r in result.results if not r.error]
        if successful:
            n = len(successful)
            result.avg_faithfulness = sum(r.faithfulness for r in successful) / n
            result.avg_answer_relevancy = sum(r.answer_relevancy for r in successful) / n
            result.avg_context_precision = sum(r.context_precision for r in successful) / n
            result.avg_context_recall = sum(r.context_recall for r in successful) / n
            result.avg_keyword_coverage = sum(r.keyword_coverage for r in successful) / n
            result.avg_retrieval_time = sum(r.retrieval_time for r in successful) / n
            result.avg_generation_time = sum(r.generation_time for r in successful) / n

        return result

    def generate_report(self, result: EvaluationResult) -> str:
        """
        生成可读的评估报告文本。

        Args:
            result: 评估结果
        Returns:
            格式化的报告字符串
        """
        lines = [
            "=" * 70,
            "  RAG 系统评估报告",
            "=" * 70,
            "",
            f"  评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  检索策略: {self.strategy_name}",
            f"  对话模型: {result.config_info.get('llm_model', 'N/A')}",
            f"  切片大小: {result.config_info.get('chunk_size', 'N/A')}",
            f"  检索 K:   {result.config_info.get('retrieval_k', 'N/A')}",
            f"  Top-N:    {result.config_info.get('rerank_top_n', 'N/A')}",
            "",
            "-" * 70,
            "  评估汇总",
            "-" * 70,
            f"  总问题数:       {result.total_questions}",
            f"  成功:           {result.success_count}",
            f"  失败:           {result.fail_count}",
            "",
            f"  平均忠实度:       {result.avg_faithfulness:.4f}",
            f"  平均答案相关性:   {result.avg_answer_relevancy:.4f}",
            f"  平均上下文精确率: {result.avg_context_precision:.4f}",
            f"  平均上下文召回率: {result.avg_context_recall:.4f}",
            f"  平均关键词覆盖率: {result.avg_keyword_coverage:.4f}",
            "",
            f"  平均检索时间:     {result.avg_retrieval_time:.3f}s",
            f"  平均生成时间:     {result.avg_generation_time:.3f}s",
            "",
            "-" * 70,
            "  详细结果",
            "-" * 70,
        ]

        for i, sr in enumerate(result.results, 1):
            status = "PASS" if not sr.error else "FAIL"
            lines.append(f"\n  [{i}] [{status}] {sr.question}")
            lines.append(f"      类别: {sr.category}")
            if sr.error:
                lines.append(f"      错误: {sr.error}")
            else:
                lines.append(f"      忠实度: {sr.faithfulness:.2f} | 相关性: {sr.answer_relevancy:.2f} | "
                             f"精确率: {sr.context_precision:.2f} | 召回率: {sr.context_recall:.2f}")
                lines.append(f"      关键词覆盖: {sr.keyword_coverage:.2f} | "
                             f"引用数: {sr.citations_count} | "
                             f"检索: {sr.retrieval_time:.2f}s | 生成: {sr.generation_time:.2f}s")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def save_report(self, result: EvaluationResult, output_dir: str = "evaluation/reports") -> str:
        """
        保存评估报告到文件（文本 + JSON）。

        Args:
            result:     评估结果
            output_dir: 输出目录
        Returns:
            报告文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 文本报告
        txt_path = os.path.join(output_dir, f"eval_report_{self.strategy_name}_{timestamp}.txt")
        report_text = self.generate_report(result)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        # JSON 报告
        json_path = os.path.join(output_dir, f"eval_report_{self.strategy_name}_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info("评估报告已保存: %s, %s", txt_path, json_path)
        return txt_path
