"""
对比实验模块
==================
对 RAG 系统进行参数对比实验，生成数据表格。

实验维度:
  1. chunk_size 对比（需重建索引）
  2. retrieval_k 对比
  3. rerank_top_n 对比
  4. 有无 rerank 对比
  5. 检索策略对比（vector / hyde / multi_query / hybrid）

输出:
  experiments/results/experiment_<name>_<timestamp>.json
  experiments/results/experiment_<name>_<timestamp>.txt
"""

from __future__ import annotations

import os
import sys
import json
import time
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """单组实验配置。"""
    name: str
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    retrieval_k: int | None = None
    rerank_top_n: int | None = None
    strategy: str = "vector"
    disable_rerank: bool = False
    description: str = ""


@dataclass
class ExperimentResult:
    """单组实验结果。"""
    config: ExperimentConfig
    avg_faithfulness: float = 0.0
    avg_answer_relevancy: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_keyword_coverage: float = 0.0
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    success_rate: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "metrics": {
                "avg_faithfulness": round(self.avg_faithfulness, 4),
                "avg_answer_relevancy": round(self.avg_answer_relevancy, 4),
                "avg_context_precision": round(self.avg_context_precision, 4),
                "avg_context_recall": round(self.avg_context_recall, 4),
                "avg_keyword_coverage": round(self.avg_keyword_coverage, 4),
                "avg_retrieval_time": round(self.avg_retrieval_time, 3),
                "avg_generation_time": round(self.avg_generation_time, 3),
                "success_rate": round(self.success_rate, 4),
            },
            "error": self.error,
        }


class ExperimentRunner:
    """对比实验执行器。"""

    def __init__(self) -> None:
        self.results: list[ExperimentResult] = []

    def run_experiment(self, exp_config: ExperimentConfig,
                       test_questions: list[dict] | None = None,
                       rebuild_index: bool = False) -> ExperimentResult:
        """
        运行单组实验。

        Args:
            exp_config:     实验配置
            test_questions: 测试问题（None 使用默认）
            rebuild_index:  是否需要重建索引（chunk_size 变化时需要）
        Returns:
            ExperimentResult
        """
        from modules import VectorStore, RAGQA, DocumentLoader
        from evaluation import RAGEvaluator, TEST_QUESTIONS

        if test_questions is None:
            test_questions = TEST_QUESTIONS

        result = ExperimentResult(config=exp_config)
        logger.info("开始实验: %s", exp_config.name)

        try:
            # 如果需要重建索引
            if rebuild_index and exp_config.chunk_size:
                logger.info("重建索引 (chunk_size=%d) ...", exp_config.chunk_size)
                # 更新 Config
                Config.CHUNK_SIZE = exp_config.chunk_size
                Config.CHUNK_OVERLAP = exp_config.chunk_overlap or 50

                # 删除旧索引
                import shutil
                if VectorStore.has_index():
                    shutil.rmtree(Config.FAISS_INDEX_DIR)

                # 重新构建
                loader = DocumentLoader(
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.CHUNK_OVERLAP,
                )
                chunks = loader.load_and_split(Config.PDF_DIR)
                vs = VectorStore()
                vs.build_index(chunks, save=True)
            else:
                vs = VectorStore()
                vs.load_index()

            # 配置 rerank 禁用
            if exp_config.disable_rerank:
                VectorStore._rerank_disabled = True
            else:
                VectorStore._rerank_disabled = False

            # 初始化 RAG 引擎和评估器
            qa = RAGQA(vs)
            evaluator = RAGEvaluator(vs, qa, strategy_name=exp_config.strategy)

            # 获取文件列表
            import glob
            doc_files = [
                os.path.basename(f)
                for f in sorted(glob.glob(os.path.join(Config.PDF_DIR, "*.pdf")))
            ]

            # 运行评估
            eval_result = evaluator.evaluate_batch(
                test_questions=test_questions,
                k=exp_config.retrieval_k,
                top_n=exp_config.rerank_top_n,
                file_filter=doc_files if doc_files else None,
            )

            # 提取指标
            result.avg_faithfulness = eval_result.avg_faithfulness
            result.avg_answer_relevancy = eval_result.avg_answer_relevancy
            result.avg_context_precision = eval_result.avg_context_precision
            result.avg_context_recall = eval_result.avg_context_recall
            result.avg_keyword_coverage = eval_result.avg_keyword_coverage
            result.avg_retrieval_time = eval_result.avg_retrieval_time
            result.avg_generation_time = eval_result.avg_generation_time
            result.success_rate = eval_result.success_count / max(eval_result.total_questions, 1)

            logger.info("实验完成: %s (忠实度=%.4f, 相关性=%.4f)",
                        exp_config.name, result.avg_faithfulness, result.avg_answer_relevancy)

        except Exception as e:
            logger.error("实验失败: %s — %s", exp_config.name, e)
            result.error = str(e)

        self.results.append(result)
        return result

    def generate_comparison_table(self) -> str:
        """生成对比表格文本。"""
        lines = [
            "=" * 120,
            "  RAG 对比实验结果",
            f"  生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 120,
            "",
        ]

        if not self.results:
            lines.append("  无实验结果")
            return "\n".join(lines)

        # 表头
        header = (
            f"{'实验名称':<25} {'策略':<12} {'chunk':<8} {'K':<5} {'TopN':<5} "
            f"{'忠实度':<8} {'相关性':<8} {'精确率':<8} {'召回率':<8} "
            f"{'关键词':<8} {'检索时间':<10} {'生成时间':<10}"
        )
        lines.append(header)
        lines.append("-" * 120)

        for r in self.results:
            c = r.config
            if r.error:
                row = f"{c.name:<25} {c.strategy:<12} {str(c.chunk_size or '-'):<8} " \
                      f"{str(c.retrieval_k or '-'):<5} {str(c.rerank_top_n or '-'):<5} " \
                      f"{'ERROR':<8} {'':<8} {'':<8} {'':<8} {'':<8} {'':<10} {'':<10}"
            else:
                row = (
                    f"{c.name:<25} {c.strategy:<12} {str(c.chunk_size or '-'):<8} "
                    f"{str(c.retrieval_k or '-'):<5} {str(c.rerank_top_n or '-'):<5} "
                    f"{r.avg_faithfulness:<8.4f} {r.avg_answer_relevancy:<8.4f} "
                    f"{r.avg_context_precision:<8.4f} {r.avg_context_recall:<8.4f} "
                    f"{r.avg_keyword_coverage:<8.4f} "
                    f"{r.avg_retrieval_time:<10.3f} {r.avg_generation_time:<10.3f}"
                )
            lines.append(row)

        lines.append("=" * 120)
        return "\n".join(lines)

    def save_results(self, experiment_name: str = "experiment") -> str:
        """保存实验结果到文件。"""
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 文本表格
        txt_path = os.path.join(output_dir, f"{experiment_name}_{timestamp}.txt")
        table = self.generate_comparison_table()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(table)

        # JSON 数据
        json_path = os.path.join(output_dir, f"{experiment_name}_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)

        logger.info("实验结果已保存: %s, %s", txt_path, json_path)
        return txt_path
