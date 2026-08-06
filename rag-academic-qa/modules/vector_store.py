"""
向量库模块
==================
负责文本向量化、FAISS 向量库的构建/持久化/加载，
以及向量检索 + DashScope 重排序（gte-rerank）。

核心功能:
  1. build_index()   — 构建 FAISS 索引并持久化
  2. save_index()    — 保存索引到磁盘
  3. load_index()    — 从磁盘加载已有索引
  4. search()        — 向量相似度检索
  5. rerank()        — 调用 gte-rerank 重排序
  6. retrieve()      — 向量检索 + 重排序（一步到位）
  7. has_index()     — 检查本地索引是否存在
"""

import os
import sys

import dashscope
from dashscope import TextReRank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """FAISS 向量库管理器，集成 DashScope Embedding 与重排序。"""

    # 类级标记：rerank 是否可用（首次 403 后自动禁用）
    _rerank_disabled = False

    def __init__(self):
        self.embeddings = DashScopeEmbeddings(
            model=Config.EMBEDDING_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
        )
        self.vectorstore: FAISS = None

    # ======================== 索引构建与持久化 ========================

    def build_index(self, documents: list[Document], save: bool = True) -> FAISS:
        """
        从文档列表构建 FAISS 向量库。

        Args:
            documents: Document 对象列表（已分块）
            save:      是否持久化到磁盘
        Returns:
            FAISS 向量库实例
        """
        if not documents:
            raise ValueError("文档列表为空，无法构建索引")

        logger.info("正在生成 Embedding 并构建 FAISS 索引 ...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)

        if save:
            self.save_index()

        logger.info("FAISS 索引构建成功，共 %d 个向量", self.vectorstore.index.ntotal)
        return self.vectorstore

    def save_index(self, path: str | None = None) -> None:
        """将 FAISS 索引持久化到磁盘。"""
        path = path or Config.FAISS_INDEX_DIR
        os.makedirs(path, exist_ok=True)
        self.vectorstore.save_local(path)
        logger.info("索引已保存至 %s", path)

    def load_index(self, path: str | None = None) -> FAISS:
        """
        从磁盘加载已有的 FAISS 索引。

        Args:
            path: 索引目录（默认 Config.FAISS_INDEX_DIR）
        Returns:
            FAISS 向量库实例
        Raises:
            FileNotFoundError: 索引文件不存在
        """
        path = path or Config.FAISS_INDEX_DIR
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FAISS 索引不存在: {path}\n请先运行 build 命令构建索引。"
            )

        self.vectorstore = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True,  # 本地构建的索引，安全
        )
        logger.info("FAISS 索引已加载，共 %d 个向量", self.vectorstore.index.ntotal)
        return self.vectorstore

    @staticmethod
    def has_index(path: str | None = None) -> bool:
        """检查本地是否已存在 FAISS 索引。"""
        path = path or Config.FAISS_INDEX_DIR
        return os.path.exists(os.path.join(path, "index.faiss"))

    # ======================== 检索与重排序 ========================

    def search(self, query: str, k: int | None = None, file_filter: list[str] | None = None) -> list[tuple[Document, float]]:
        """
        向量相似度检索（带分数）。

        Args:
            query:        用户问题
            k:            返回的候选数量（默认 Config.RETRIEVAL_K）
            file_filter:  按文件名过滤，只返回属于这些文件的片段（None 表示不过滤）
        Returns:
            List[Tuple[Document, float]]，按相似度降序排列
        """
        k = k or Config.RETRIEVAL_K
        if self.vectorstore is None:
            raise RuntimeError("向量库未加载，请先 build_index 或 load_index")

        # 未指定文件过滤，直接返回全局检索结果
        if not file_filter:
            return self.vectorstore.similarity_search_with_score(query, k=k)

        # 指定了文件过滤：扩大候选池，避免目标文件被挤出 top-k
        filter_set = set(file_filter)
        candidate_k = max(k * 5, 50)
        candidates = self.vectorstore.similarity_search_with_score(query, k=candidate_k)

        filtered = []
        for doc, score in candidates:
            file_name = doc.metadata.get("file_name", "")
            source = doc.metadata.get("source", "")
            basename = os.path.basename(source)
            if file_name in filter_set or basename in filter_set:
                filtered.append((doc, score))

        return filtered[:k]

    def rerank(self, query: str, documents: list[Document], top_n: int | None = None, scores: list[float] | None = None) -> list[tuple[Document, float]]:
        """
        调用 DashScope gte-rerank 对检索结果重排序。

        Args:
            query:     用户问题
            documents: 待重排序的 Document 列表
            top_n:     重排序后保留的数量（默认 Config.RERANK_TOP_N）
            scores:    FAISS 原始相似度分数（rerank 不可用时作为 fallback）
        Returns:
            List[Tuple[Document, float]]，按相关性降序排列
        """
        top_n = top_n or Config.RERANK_TOP_N
        if not documents:
            return []

        # 如果 rerank 已被禁用，直接用 FAISS 分数
        if VectorStore._rerank_disabled:
            if scores:
                paired = list(zip(documents, scores))
                paired.sort(key=lambda x: x[1], reverse=True)
                return paired[:top_n]
            return [(doc, 0.0) for doc in documents[:top_n]]

        if len(documents) <= top_n:
            if scores:
                paired = list(zip(documents, scores))
                paired.sort(key=lambda x: x[1], reverse=True)
                return paired
            return [(doc, 0.0) for doc in documents]

        dashscope.api_key = Config.DASHSCOPE_API_KEY

        # 提取文本用于重排序
        texts = [doc.page_content for doc in documents]

        try:
            resp = TextReRank.call(
                model=Config.RERANK_MODEL,
                query=query,
                documents=texts,
                top_n=top_n,
                return_documents=False,
            )

            if resp.status_code != 200:
                # 403/AccessDenied → 永久禁用 rerank，后续直接用 FAISS 分数
                if resp.status_code == 403:
                    VectorStore._rerank_disabled = True
                    logger.warning("重排序模型不可用（权限不足），自动改用向量相似度排序")
                else:
                    logger.warning("重排序失败 (HTTP %d)，改用向量相似度排序", resp.status_code)
                # 用 FAISS 原始分数 fallback
                if scores:
                    paired = list(zip(documents, scores))
                    paired.sort(key=lambda x: x[1], reverse=True)
                    return paired[:top_n]
                return [(doc, 0.0) for doc in documents[:top_n]]

            # 解析重排序结果，映射回原始 Document
            reranked = []
            results = resp.output.results
            for result in results:
                idx = result.index if hasattr(result, "index") else result["index"]
                score = (
                    result.relevance_score
                    if hasattr(result, "relevance_score")
                    else result["relevance_score"]
                )
                reranked.append((documents[idx], float(score)))

            return reranked

        except Exception as e:
            logger.warning("重排序异常: %s，改用向量相似度排序", e)
            VectorStore._rerank_disabled = True
            if scores:
                paired = list(zip(documents, scores))
                paired.sort(key=lambda x: x[1], reverse=True)
                return paired[:top_n]
            return [(doc, 0.0) for doc in documents[:top_n]]

    def retrieve(self, query: str, k: int | None = None, top_n: int | None = None, file_filter: list[str] | None = None) -> list[tuple[Document, float]]:
        """
        完整检索流程: 向量检索 → 重排序（一步到位）。

        Args:
            query:        用户问题
            k:            向量检索候选数（默认 Config.RETRIEVAL_K）
            top_n:        重排序后保留数（默认 Config.RERANK_TOP_N）
            file_filter:  按文件名过滤（None 表示不过滤）
        Returns:
            List[Tuple[Document, float]]，重排序后的 Top-N 结果
        """
        k = k or Config.RETRIEVAL_K
        top_n = top_n or Config.RERANK_TOP_N

        # Step 1: 向量检索（带文件过滤）
        docs_with_scores = self.search(query, k=k, file_filter=file_filter)
        documents = [doc for doc, _ in docs_with_scores]
        scores = [float(score) for _, score in docs_with_scores]

        # Step 2: 重排序（传入 FAISS 分数作为 fallback）
        reranked = self.rerank(query, documents, top_n=top_n, scores=scores)

        return reranked
