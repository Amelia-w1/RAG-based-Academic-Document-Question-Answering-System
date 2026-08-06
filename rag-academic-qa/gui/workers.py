"""
GUI 工作线程模块
==================
将耗时操作（构建索引、RAG 问答）放到 QThread 中执行，
通过信号通知主线程更新 UI，避免界面卡顿。

核心类:
  1. BuildWorker — 构建/重建 FAISS 向量库
  2. AskWorker   — 执行 RAG 问答（流式输出）
"""

import os
import sys
import logging

from PySide6.QtCore import QThread, Signal

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langchain_core.output_parsers import StrOutputParser

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# modules 延迟到 run() 中导入，避免 GUI 启动时需要全部依赖


class _SignalLogHandler(logging.Handler):
    """将日志记录通过 Qt Signal 转发到主线程的 handler。"""

    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def emit(self, record):
        try:
            self._signal.emit(self.format(record))
        except RuntimeError:
            # 信号接收端可能已销毁
            pass


class BuildWorker(QThread):
    """
    构建 FAISS 向量库的工作线程。

    信号:
      progress(str)   — 进度消息
      finished_ok(str) — 构建成功，附带统计信息
      finished_err(str)— 构建失败，附带错误信息
    """

    progress = Signal(str)
    finished_ok = Signal(str)
    finished_err = Signal(str)

    def __init__(self, chunk_size=None, chunk_overlap=None, force=False):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.force = force

    def run(self):
        # 安装日志 handler，将模块日志转发到 GUI 进度信号
        root_logger = logging.getLogger("rag_academic_qa")
        handler = _SignalLogHandler(self.progress)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)

        try:
            from modules import DocumentLoader, VectorStore

            # 校验配置
            ok, msg = Config.validate()
            if not ok:
                self.finished_err.emit(msg)
                return

            # 强制重建
            if self.force and VectorStore.has_index():
                import shutil
                self.progress.emit("删除旧索引 ...")
                shutil.rmtree(Config.FAISS_INDEX_DIR)

            # Step 1: 加载 PDF + 分块
            self.progress.emit("正在加载 PDF 论文 ...")

            loader = DocumentLoader(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks = loader.load_and_split(Config.PDF_DIR)

            # Step 2: 构建 FAISS 索引
            self.progress.emit("正在生成 Embedding 并构建索引 ...")
            vs = VectorStore()
            vs.build_index(chunks, save=True)

            total = vs.vectorstore.index.ntotal
            self.finished_ok.emit(
                f"向量库构建完成！共 {total} 个向量\n"
                f"索引位置: {Config.FAISS_INDEX_DIR}"
            )

        except Exception as e:
            logger.error("构建失败: %s", e)
            self.finished_err.emit(f"构建失败: {e}")
        finally:
            root_logger.removeHandler(handler)


class AskWorker(QThread):
    """
    RAG 问答工作线程（流式输出）。
    支持多轮对话历史注入。

    信号:
      retrieved(list)  — 检索完成，附带引用列表
      chunk(str)       — 流式输出的回答片段
      finished_ok(str) — 回答完成，附带完整回答文本
      finished_err(str)— 回答失败
    """

    retrieved = Signal(list)
    chunk = Signal(str)
    finished_ok = Signal(str)
    finished_err = Signal(str)

    def __init__(self, question, k=None, top_n=None, file_filter=None,
                 history_text=None, use_history=True, strategy_name="vector",
                 fallback_enabled=False):
        super().__init__()
        self.question = question
        self.k = k
        self.top_n = top_n
        self.file_filter = file_filter or []
        self.history_text = history_text  # 格式化后的历史文本
        self.use_history = use_history
        self.strategy_name = strategy_name
        self.fallback_enabled = fallback_enabled
        self._vs = None
        self._qa = None
        self._full_answer = ""

    def run(self):
        try:
            from modules import VectorStore, RAGAgent

            # 加载向量库
            self._vs = VectorStore()
            self._vs.load_index()

            # 初始化多智能体 RAG 引擎（LangGraph 编排）
            self._agent = RAGAgent(
                self._vs,
                strategy_name=self.strategy_name,
                use_history=self.use_history,
                fallback_enabled=self.fallback_enabled,
            )

            # 构建对话历史对象（从 history_text 恢复）
            memory = None
            if self.use_history and self.history_text:
                from modules.conversation import ConversationMemory, Message
                memory = ConversationMemory()
                # 将格式化的历史文本解析为独立的用户/助手消息
                current_role: str | None = None
                current_lines: list[str] = []
                for line in self.history_text.splitlines():
                    if line.startswith("[用户] "):
                        if current_role is not None:
                            memory._messages.append(
                                Message(role=current_role, content="\n".join(current_lines))
                            )
                        current_role = "user"
                        current_lines = [line[5:]]
                    elif line.startswith("[助手] "):
                        if current_role is not None:
                            memory._messages.append(
                                Message(role=current_role, content="\n".join(current_lines))
                            )
                        current_role = "assistant"
                        current_lines = [line[5:]]
                    elif current_role is not None:
                        current_lines.append(line)
                if current_role is not None:
                    memory._messages.append(
                        Message(role=current_role, content="\n".join(current_lines))
                    )

            # —— 检索子图：查询理解 → 检索 → 质量判断 ——
            # 先返回引用与决策，保证引用卡片在回答流式输出前渲染
            plan = self._agent.retrieve_plan(self.question, memory)
            if plan["docs"]:
                self.retrieved.emit(plan["citations"])

            # 未检索到且未开启兜底：直接告知
            if not plan["docs"] and not plan["fallback"]:
                msg = "未检索到相关文献片段，无法回答。"
                self.chunk.emit(msg)
                self._full_answer = msg
                self.finished_ok.emit(msg)
                return

            # —— 生成子图：流式输出回答 token ——
            for token in self._agent.generate_stream(self.question, plan, memory):
                self._full_answer += token
                self.chunk.emit(token)

            self.finished_ok.emit(self._full_answer)

        except FileNotFoundError as e:
            logger.error("向量库不存在: %s", e)
            self.finished_err.emit(f"向量库不存在: {e}\n请先构建索引。")
        except Exception as e:
            logger.error("问答失败: %s", e)
            self.finished_err.emit(f"问答失败: {e}")
