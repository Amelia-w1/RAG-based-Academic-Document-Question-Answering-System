#!/usr/bin/env python
"""
RAG 学术文献智能问答工具 — CLI 入口
================================================
基于 RAG 的学术文献智能问答系统
技术栈: Python + LangChain + FAISS + DashScope API

用法:
  python main.py build     构建向量库（读取 data/papers/ 下 PDF）
  python main.py ask       单次提问
  python main.py chat      交互式问答（连续对话）
  python main.py info      查看当前配置

示例:
  python main.py build
  python main.py ask "什么是图像去噪中的 BM3D 算法？"
  python main.py chat
  python main.py info
"""

import argparse
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from modules import DocumentLoader, VectorStore, RAGQA, ConversationMemory
from utils.logger import get_logger

logger = get_logger(__name__)


# ======================== 分隔线 ========================
SEP = "=" * 60
THIN_SEP = "-" * 60


def cmd_build(args):
    """构建 FAISS 向量库。"""
    print(SEP)
    print("  构建 FAISS 向量库")
    print(SEP)

    ok, msg = Config.validate()
    if not ok:
        logger.error(msg)
        print(f"  [错误] {msg}")
        sys.exit(1)

    # 强制重建：删除旧索引
    if args.force and VectorStore.has_index():
        print("  [提示] --force 模式，删除旧索引 ...")
        import shutil
        shutil.rmtree(Config.FAISS_INDEX_DIR)

    # 检查是否已有索引
    if VectorStore.has_index() and not args.force:
        print(f"  [提示] 向量库已存在: {Config.FAISS_INDEX_DIR}")
        choice = input("  是否重新构建？(y/N): ").strip().lower()
        if choice != "y":
            print("  已取消。")
            return

    # Step 1: 加载 PDF + 分块
    loader = DocumentLoader(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = loader.load_and_split(Config.PDF_DIR)

    # Step 2: 构建 FAISS 索引
    vs = VectorStore()
    vs.build_index(chunks, save=True)

    print(SEP)
    print("  向量库构建完成！")
    print(f"  索引位置: {Config.FAISS_INDEX_DIR}")
    print(f"  向量总数: {vs.vectorstore.index.ntotal}")
    print(f"  现在可以运行: python main.py ask \"你的问题\"")
    print(SEP)


def cmd_ask(args):
    """单次问答。"""
    print(SEP)
    print("  RAG 文献问答")
    print(SEP)

    ok, msg = Config.validate()
    if not ok:
        logger.error(msg)
        print(f"  [错误] {msg}")
        sys.exit(1)

    # 加载向量库
    vs = VectorStore()
    if not VectorStore.has_index():
        logger.error("向量库不存在")
        print(f"  [错误] 向量库不存在，请先运行: python main.py build")
        sys.exit(1)
    vs.load_index()

    # 初始化 RAG 引擎
    qa = RAGQA(vs)

    # 提问
    question = args.question
    strategy = args.strategy
    logger.info("单次提问: %s (策略: %s)", question, strategy)
    print(f"\n  问题: {question}")
    print(f"  策略: {strategy}")
    print()

    # 使用检索策略
    if strategy != "vector":
        try:
            from modules.advanced_retrieval import get_strategy
            strat = get_strategy(strategy, vs)
            docs_with_scores = strat.retrieve(
                question,
                k=args.k or Config.RETRIEVAL_K,
                top_n=args.top_n or Config.RERANK_TOP_N,
            )
            context = qa.format_context(docs_with_scores)
            answer = qa.chain.invoke({"context": context, "question": question})
            citations = qa.extract_citations(docs_with_scores)
            result = {
                "question": question,
                "answer": answer,
                "citations": citations,
                "context": context,
            }
        except ImportError:
            print(f"  [提示] 高级检索模块未安装，使用标准检索")
            result = qa.ask(
                question,
                k=args.k or Config.RETRIEVAL_K,
                top_n=args.top_n or Config.RERANK_TOP_N,
            )
    else:
        result = qa.ask(
            question,
            k=args.k or Config.RETRIEVAL_K,
            top_n=args.top_n or Config.RERANK_TOP_N,
        )

    # 输出回答
    print(f"\n  回答:\n")
    # 缩进输出回答
    for line in result["answer"].split("\n"):
        print(f"  {line}")

    # 输出引用
    print(f"\n{THIN_SEP}")
    print(f"  引用来源（共 {len(result['citations'])} 条）:\n")
    for cite in result["citations"]:
        print(f"  [片段{cite['fragment_id']}] {cite['file_name']} — 第 {cite['page']} 页")
        print(f"    相关性: {cite['relevance_score']}")
        # 显示原文片段（截断显示）
        snippet = cite["content"][:150].replace("\n", " ")
        print(f"    原文: {snippet}...")
        print()

    print(SEP)


def cmd_chat(args):
    """交互式问答模式（支持多轮对话）。"""
    print(SEP)
    print("  RAG 文献问答 — 交互模式（多轮对话）")
    print(f"  输入问题开始问答，输入 quit/exit/q 退出")
    print(f"  输入 clear 清空对话历史")
    print(SEP)

    ok, msg = Config.validate()
    if not ok:
        logger.error(msg)
        print(f"  [错误] {msg}")
        sys.exit(1)

    # 加载向量库
    vs = VectorStore()
    if not VectorStore.has_index():
        logger.error("向量库不存在")
        print(f"  [错误] 向量库不存在，请先运行: python main.py build")
        sys.exit(1)
    vs.load_index()

    # 初始化 RAG 引擎和对话记忆
    qa = RAGQA(vs)
    memory = ConversationMemory(max_turns=5)

    print(f"\n  向量库已就绪（{vs.vectorstore.index.ntotal} 个向量）")
    print(f"  模型: {Config.LLM_MODEL} | 温度: {Config.TEMPERATURE}")
    print(f"  多轮对话: 已启用（最多保留 {memory.max_turns} 轮历史）")
    print()

    while True:
        try:
            question = input("\n  问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("  再见！")
            break
        if question.lower() == "clear":
            memory.clear()
            print("  [提示] 对话历史已清空")
            continue

        print(THIN_SEP)

        # 多轮问答（带历史）
        result = qa.ask(
            question,
            k=args.k or Config.RETRIEVAL_K,
            top_n=args.top_n or Config.RERANK_TOP_N,
            memory=memory,
            use_history=True,
        )

        # 记录到对话历史
        memory.add_user_message(question)
        memory.add_ai_message(result["answer"])

        # 显示改写后的查询（如果有）
        if result.get("rewritten_query") and result["rewritten_query"] != question:
            print(f"  [查询改写] {result['rewritten_query']}")

        # 输出回答
        print(f"  回答> {result['answer']}")

        # 输出引用
        if result["citations"]:
            print(f"\n  引用来源:")
            for cite in result["citations"]:
                print(f"    [片段{cite['fragment_id']}] {cite['file_name']} — 第 {cite['page']} 页 "
                      f"(相关性: {cite['relevance_score']})")

        print(f"\n  [对话轮数: {memory.get_turn_count()} | 估算 token: {memory.get_token_estimate()}]")
        print(THIN_SEP)


def cmd_info(args):
    """查看当前配置。"""
    print(Config.display())


# ======================== 参数解析 ========================
def main():
    parser = argparse.ArgumentParser(
        description="RAG 学术文献智能问答工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py build                          构建向量库
  python main.py build --force                  强制重建向量库
  python main.py build --chunk-size 800         自定义切片大小
  python main.py ask "什么是BM3D算法？"          单次提问
  python main.py ask "..." --k 15 --top-n 8     自定义检索参数
  python main.py chat                           交互模式
  python main.py info                           查看配置
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # build 子命令
    p_build = subparsers.add_parser("build", help="构建 FAISS 向量库")
    p_build.add_argument("--force", action="store_true", help="强制重建（删除旧索引）")
    p_build.add_argument("--chunk-size", type=int, default=None, help=f"切片大小（默认 {Config.CHUNK_SIZE}）")
    p_build.add_argument("--chunk-overlap", type=int, default=None, help=f"重叠长度（默认 {Config.CHUNK_OVERLAP}）")

    # ask 子命令
    p_ask = subparsers.add_parser("ask", help="单次提问")
    p_ask.add_argument("question", help="你的问题")
    p_ask.add_argument("--k", type=int, default=None, help=f"向量检索候选数（默认 {Config.RETRIEVAL_K}）")
    p_ask.add_argument("--top-n", type=int, default=None, help=f"重排序 Top-N（默认 {Config.RERANK_TOP_N}）")
    p_ask.add_argument("--strategy", type=str, default="vector",
                       choices=["vector", "hyde", "multi_query", "hybrid"],
                       help="检索策略（默认 vector）")

    # chat 子命令
    p_chat = subparsers.add_parser("chat", help="交互式问答模式（多轮对话）")
    p_chat.add_argument("--k", type=int, default=None, help=f"向量检索候选数（默认 {Config.RETRIEVAL_K}）")
    p_chat.add_argument("--top-n", type=int, default=None, help=f"重排序 Top-N（默认 {Config.RERANK_TOP_N}）")
    p_chat.add_argument("--strategy", type=str, default="vector",
                        choices=["vector", "hyde", "multi_query", "hybrid"],
                        help="检索策略（默认 vector）")

    # info 子命令
    subparsers.add_parser("info", help="查看当前配置")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # 分发命令
    commands = {
        "build": cmd_build,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
