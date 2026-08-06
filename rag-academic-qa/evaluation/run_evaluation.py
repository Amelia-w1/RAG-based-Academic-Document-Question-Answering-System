#!/usr/bin/env python
"""
RAG 系统评估脚本
==================
运行评估并生成报告。

用法:
  python -m evaluation.run_evaluation                          # 使用默认问题和策略
  python -m evaluation.run_evaluation --strategy hyde          # 指定策略
  python -m evaluation.run_evaluation --k 15 --top-n 8         # 自定义参数
  python -m evaluation.run_evaluation --all-strategies         # 对比所有策略

输出:
  evaluation/reports/eval_report_<strategy>_<timestamp>.txt
  evaluation/reports/eval_report_<strategy>_<timestamp>.json
"""

import argparse
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from modules import VectorStore, RAGQA
from evaluation import RAGEvaluator, TEST_QUESTIONS
from evaluation.evaluator import EvaluationResult
from utils.logger import get_logger

logger = get_logger(__name__)


def run_evaluation(strategy: str = "vector", k: int = None, top_n: int = None,
                   file_filter: list = None) -> EvaluationResult:
    """运行单次评估。"""
    # 校验配置
    ok, msg = Config.validate()
    if not ok:
        print(f"[错误] {msg}")
        sys.exit(1)

    # 加载向量库
    vs = VectorStore()
    if not VectorStore.has_index():
        print("[错误] 向量库不存在，请先运行: python main.py build")
        sys.exit(1)
    vs.load_index()

    # 初始化 RAG 引擎
    qa = RAGQA(vs)

    # 初始化评估器
    evaluator = RAGEvaluator(vs, qa, strategy_name=strategy)

    # 获取文件列表（如果未指定 file_filter）
    if not file_filter:
        import glob
        pdf_dir = Config.PDF_DIR
        doc_files = [
            os.path.basename(f)
            for f in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
        ]
        if doc_files:
            file_filter = doc_files
            print(f"[信息] 使用全部 {len(file_filter)} 篇文档进行评估")
        else:
            print("[错误] 未找到任何 PDF 文档")
            sys.exit(1)

    # 运行评估
    print(f"\n[评估] 策略={strategy}, K={k or Config.RETRIEVAL_K}, "
          f"TopN={top_n or Config.RERANK_TOP_N}, 问题数={len(TEST_QUESTIONS)}\n")

    result = evaluator.evaluate_batch(
        test_questions=TEST_QUESTIONS,
        k=k, top_n=top_n, file_filter=file_filter,
    )

    # 生成并保存报告
    report = evaluator.generate_report(result)
    print(report)

    txt_path = evaluator.save_report(result)
    print(f"\n[完成] 报告已保存至: {txt_path}")

    return result


def compare_strategies(k: int = None, top_n: int = None,
                       file_filter: list = None) -> None:
    """对比所有检索策略。"""
    strategies = ["vector", "hyde", "multi_query", "hybrid"]
    all_results = {}

    print("=" * 70)
    print("  RAG 检索策略对比评估")
    print("=" * 70)

    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"  正在评估策略: {strategy}")
        print(f"{'='*50}")
        try:
            result = run_evaluation(strategy=strategy, k=k, top_n=top_n, file_filter=file_filter)
            all_results[strategy] = result
        except Exception as e:
            print(f"  [错误] 策略 {strategy} 评估失败: {e}")
            all_results[strategy] = None

    # 输出对比表格
    print("\n")
    print("=" * 70)
    print("  策略对比汇总")
    print("=" * 70)
    header = f"{'策略':<15} {'忠实度':<10} {'相关性':<10} {'精确率':<10} {'召回率':<10} {'关键词':<10}"
    print(header)
    print("-" * 70)
    for strategy, result in all_results.items():
        if result:
            row = (f"{strategy:<15} {result.avg_faithfulness:<10.4f} "
                   f"{result.avg_answer_relevancy:<10.4f} "
                   f"{result.avg_context_precision:<10.4f} "
                   f"{result.avg_context_recall:<10.4f} "
                   f"{result.avg_keyword_coverage:<10.4f}")
        else:
            row = f"{strategy:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}"
        print(row)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="RAG 系统评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", type=str, default="vector",
                        choices=["vector", "hyde", "multi_query", "hybrid"],
                        help="检索策略（默认 vector）")
    parser.add_argument("--k", type=int, default=None, help="向量检索候选数")
    parser.add_argument("--top-n", type=int, default=None, help="重排序 Top-N")
    parser.add_argument("--all-strategies", action="store_true",
                        help="对比所有检索策略")

    args = parser.parse_args()

    if args.all_strategies:
        compare_strategies(k=args.k, top_n=args.top_n)
    else:
        run_evaluation(
            strategy=args.strategy, k=args.k, top_n=args.top_n
        )


if __name__ == "__main__":
    main()
