"""
RAG 学术文献问答系统 — 全局配置模块
通过 .env 文件管理 API 密钥与可调参数，所有模块共享此配置。
"""

import os
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()


class Config:
    """全局配置类，所有参数均可通过 .env 文件或环境变量覆盖。"""

    # ======================== DashScope API ========================
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")

    # 对话模型（默认 qwen-turbo，可选 qwen-plus / qwen-max）
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen-turbo")

    # 向量模型（默认 text-embedding-v2）
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")

    # 重排序模型（默认 gte-rerank）
    RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank")

    # ======================== 文档处理参数 ========================
    # 文本切片大小（字符数）
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))

    # 切片重叠长度（字符数）
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ======================== 检索参数 ========================
    # 向量检索返回的候选数量（重排序前）
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "10"))

    # 重排序后保留的 Top-N 文档数
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

    # ======================== 生成参数 ========================
    # LLM 温度（0=确定性输出，1=高随机性；RAG 场景建议低温度）
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

    # ======================== 路径配置 ========================
    # PDF 论文目录
    PDF_DIR = os.getenv("PDF_DIR", os.path.join(os.path.dirname(__file__), "data", "papers"))

    # FAISS 向量库持久化目录
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", os.path.join(os.path.dirname(__file__), "faiss_index"))

    # ======================== 校验 ========================
    @classmethod
    def validate(cls) -> tuple[bool, str]:
        """校验必要配置是否完整，返回 (bool, str)。"""
        if not cls.DASHSCOPE_API_KEY:
            return False, "DASHSCOPE_API_KEY 未设置，请在 .env 文件中配置。"
        if not os.path.isdir(cls.PDF_DIR):
            return False, f"PDF 目录不存在: {cls.PDF_DIR}"
        return True, "配置校验通过。"

    @classmethod
    def display(cls) -> str:
        """返回当前配置的可读字符串（隐藏 API Key）。"""
        lines = [
            "=" * 60,
            "  RAG 学术文献问答系统 — 当前配置",
            "=" * 60,
            f"  对话模型:     {cls.LLM_MODEL}",
            f"  向量模型:     {cls.EMBEDDING_MODEL}",
            f"  重排序模型:   {cls.RERANK_MODEL}",
            f"  切片大小:     {cls.CHUNK_SIZE} 字符",
            f"  重叠长度:     {cls.CHUNK_OVERLAP} 字符",
            f"  检索数量 K:   {cls.RETRIEVAL_K}",
            f"  重排序 Top-N: {cls.RERANK_TOP_N}",
            f"  模型温度:     {cls.TEMPERATURE}",
            f"  PDF 目录:     {cls.PDF_DIR}",
            f"  向量库目录:   {cls.FAISS_INDEX_DIR}",
            f"  API Key:      {'已设置 ✓' if cls.DASHSCOPE_API_KEY else '未设置 ✗'}",
            "=" * 60,
        ]
        return "\n".join(lines)
