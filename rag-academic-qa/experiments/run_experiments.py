#!/usr/bin/env python
"""
RAG 对比实验脚本
==================
运行参数对比实验，生成数据表格。

实验类型:
  1. chunk_size   — 不同切片大小对比（需重建索引）
  2. k_values     — 不同检索候选数 K 对比
  3. top_n        — 不同重排序 Top-N 对比
  4. rerank       — 有无 rerank 对比
  5. strategies   — 不同检索策略对比
  6. all          — 运行全部实验

用法:
  python -m experiments.run_experiments --type k_values
  python -m experiments.run_experiments --type strategies
  python -m experiments.run_experiments --type all
  python -m experiments.run_experiments --type chunk_size --chunk-sizes 500 800 1200
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from utils.logger import get_logger

logger = get_logger(__name__)


def experiment_k_values(runner: ExperimentRunner, k_values: list[int] = None):
    """实验: 不同检索候选数 K。"""
    if k_values is None:
        k_values = [5, 10, 15, 20]

    print("\n" + "=" * 60)
    print(f"  实验: 检索候选数 K 对比 (K={k_values})")
    print("=" * 60)

    for k in k_values:
        config = ExperimentConfig(
            name=f"K={k}",
            retrieval_k=k,
            rerank_top_n=Config.RERANK_TOP_N,
            strategy="vector",
            description=f"检索候选数 K={k}",
        )
        runner.run_experiment(config)


def experiment_top_n(runner: ExperimentRunner, top_n_values: list[int] = None):
    """实验: 不同重排序 Top-N。"""
    if top_n_values is None:
        top_n_values = [3, 5, 8, 10]

    print("\n" + "=" * 60)
    print(f"  实验: 重排序 Top-N 对比 (TopN={top_n_values})")
    print("=" * 60)

    for top_n in top_n_values:
        config = ExperimentConfig(
            name=f"TopN={top_n}",
            retrieval_k=Config.RETRIEVAL_K,
            rerank_top_n=top_n,
            strategy="vector",
            description=f"重排序 Top-N={top_n}",
        )
        runner.run_experiment(config)


def experiment_rerank(runner: ExperimentRunner):
    """实验: 有无 rerank 对比。"""
    print("\n" + "=" * 60)
    print("  实验: 有无 Rerank 对比")
    print("=" * 60)

    # 有 rerank
    config_with = ExperimentConfig(
        name="with_rerank",
        retrieval_k=Config.RETRIEVAL_K,
        rerank_top_n=Config.RERANK_TOP_N,
        strategy="vector",
        disable_rerank=False,
        description="启用重排序",
    )
    runner.run_experiment(config_with)

    # 无 rerank
    config_without = ExperimentConfig(
        name="without_rerank",
        retrieval_k=Config.RETRIEVAL_K,
        rerank_top_n=Config.RERANK_TOP_N,
        strategy="vector",
        disable_rerank=True,
        description="禁用重排序",
    )
    runner.run_experiment(config_without)


def experiment_strategies(runner: ExperimentRunner):
    """实验: 不同检索策略对比。"""
    strategies = ["vector", "hyde", "multi_query", "hybrid"]

    print("\n" + "=" * 60)
    print(f"  实验: 检索策略对比 ({', '.join(strategies)})")
    print("=" * 60)

    for strategy in strategies:
        config = ExperimentConfig(
            name=f"strategy={strategy}",
            retrieval_k=Config.RETRIEVAL_K,
            rerank_top_n=Config.RERANK_TOP_N,
            strategy=strategy,
            description=f"检索策略: {strategy}",
        )
        runner.run_experiment(config)


def experiment_chunk_size(runner: ExperimentRunner, chunk_sizes: list[int] = None):
    """实验: 不同切片大小对比（需重建索引）。"""
    if chunk_sizes is None:
        chunk_sizes = [300, 500, 800, 1200]

    print("\n" + "=" * 60)
    print(f"  实验: 切片大小对比 (chunk_size={chunk_sizes})")
    print("  注意: 此实验需要多次重建索引，耗时较长")
    print("=" * 60)

    original_chunk_size = Config.CHUNK_SIZE
    original_chunk_overlap = Config.CHUNK_OVERLAP

    for cs in chunk_sizes:
        config = ExperimentConfig(
            name=f"chunk={cs}",
            chunk_size=cs,
            chunk_overlap=cs // 10,  # overlap 设为 chunk_size 的 10%
            retrieval_k=Config.RETRIEVAL_K,
            rerank_top_n=Config.RERANK_TOP_N,
            strategy="vector",
            description=f"切片大小={cs}",
        )
        runner.run_experiment(config, rebuild_index=True)

    # 恢复原始索引
    print("\n  恢复原始索引 ...")
    Config.CHUNK_SIZE = original_chunk_size
    Config.CHUNK_OVERLAP = original_chunk_overlap

    from modules import DocumentLoader, VectorStore
    import shutil
    if VectorStore.has_index():
        shutil.rmtree(Config.FAISS_INDEX_DIR)
    loader = DocumentLoader(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    chunks = loader.load_and_split(Config.PDF_DIR)
    vs = VectorStore()
    vs.build_index(chunks, save=True)
    print("  原始索引已恢复")


def main():
    parser = argparse.ArgumentParser(
        description="RAG 对比实验工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--type", type=str, default="k_values",
                        choices=["chunk_size", "k_values", "top_n", "rerank", "strategies", "all"],
                        help="实验类型（默认 k_values）")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=None,
                        help="chunk_size 实验的参数列表")
    parser.add_argument("--k-values", type=int, nargs="+", default=None,
                        help="K 值实验的参数列表")

    args = parser.parse_args()

    # 校验配置
    ok, msg = Config.validate()
    if not ok:
        print(f"[错误] {msg}")
        sys.exit(1)

    runner = ExperimentRunner()

    if args.type == "all":
        experiment_k_values(runner, args.k_values)
        experiment_top_n(runner)
        experiment_rerank(runner)
        experiment_strategies(runner)
        experiment_chunk_size(runner, args.chunk_sizes)
    elif args.type == "chunk_size":
        experiment_chunk_size(runner, args.chunk_sizes)
    elif args.type == "k_values":
        experiment_k_values(runner, args.k_values)
    elif args.type == "top_n":
        experiment_top_n(runner)
    elif args.type == "rerank":
        experiment_rerank(runner)
    elif args.type == "strategies":
        experiment_strategies(runner)

    # 生成对比表格
    table = runner.generate_comparison_table()
    print("\n" + table)

    # 保存结果
    txt_path = runner.save_results(experiment_name=args.type)
    print(f"\n[完成] 结果已保存至: {txt_path}")


if __name__ == "__main__":
    main()
