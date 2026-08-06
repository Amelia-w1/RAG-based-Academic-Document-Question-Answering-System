"""
高级检索策略模块
==================
实现三种高级检索策略，提升 RAG 检索准确率:

  1. HyDE (Hypothetical Document Embedding)
     — 先让 LLM 生成假设性答案，用假设答案的向量去检索
     — 适合查询与文档表述差异大的场景

  2. Multi-Query Retrieval (多查询检索)
     — 让 LLM 将一个问题改写为多个子问题
     — 分别检索后合并去重，扩大召回

  3. Hybrid Search (混合检索)
     — 向量检索 + BM25 关键词检索融合
     — 通过 Reciprocal Rank Fusion (RRF) 合并排序

用法:
  strategy = HyDERetrieval(vector_store, llm)
  results = strategy.retrieve(query, k=10)
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from modules.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


# ======================== 检索策略基类 ========================

class BaseRetrievalStrategy(ABC):
    """检索策略抽象基类。"""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store

    @abstractmethod
    def retrieve(self, query: str, k: int = 10, file_filter: list[str] | None = None,
                 **kwargs) -> list[tuple[Document, float]]:
        """执行检索，返回 (Document, score) 列表。"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称。"""
        ...


# ======================== 基础向量检索（默认策略） ========================

class VectorRetrieval(BaseRetrievalStrategy):
    """标准向量检索 + 重排序（系统默认策略）。"""

    @property
    def name(self) -> str:
        return "vector"

    def retrieve(self, query: str, k: int = 10, file_filter: list[str] | None = None,
                 top_n: int | None = None, **kwargs) -> list[tuple[Document, float]]:
        """向量检索 + 重排序。"""
        return self.vs.retrieve(query, k=k, top_n=top_n, file_filter=file_filter)


# ======================== HyDE 策略 ========================

# HyDE Prompt：让 LLM 生成假设性文档
HYDE_PROMPT = """请根据以下问题，写一段可能包含答案的学术文献片段（约 200-300 字）。
不要回答问题本身，而是模拟一篇可能包含该问题答案的论文段落。
用学术写作风格，可以包含技术术语。

问题：{question}

假设性文献片段："""


class HyDERetrieval(BaseRetrievalStrategy):
    """
    HyDE (Hypothetical Document Embedding) 检索策略。

    流程:
      1. 让 LLM 根据问题生成假设性答案文档
      2. 用假设文档的向量去检索（而非原始问题）
      3. 对检索结果重排序

    优势: 当问题表述与文档表述差异大时，假设文档能桥接语义鸿沟。
    """

    def __init__(self, vector_store: VectorStore, llm: ChatTongyi | None = None) -> None:
        super().__init__(vector_store)
        self.llm = llm or ChatTongyi(
            model=Config.LLM_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
            temperature=0.3,  # HyDE 需要一定创造性
        )
        self.hyde_prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        self.hyde_chain = self.hyde_prompt | self.llm | StrOutputParser()

    @property
    def name(self) -> str:
        return "hyde"

    def generate_hypothetical_document(self, query: str) -> str:
        """
        让 LLM 生成假设性文档。

        Args:
            query: 用户查询
        Returns:
            假设性文档文本
        """
        logger.info("[HyDE] 正在生成假设性文档 ...")
        try:
            doc = self.hyde_chain.invoke({"question": query})
            logger.debug("[HyDE] 假设文档: %s...", doc[:100])
            return doc
        except Exception as e:
            logger.warning("[HyDE] 生成假设文档失败: %s，回退到原始查询", e)
            return query

    def retrieve(self, query: str, k: int = 10, file_filter: list[str] | None = None,
                 top_n: int | None = None, **kwargs) -> list[tuple[Document, float]]:
        """
        HyDE 检索流程: 生成假设文档 → 向量检索 → 重排序。

        Args:
            query:        用户查询
            k:            向量检索候选数
            file_filter:  按文件名过滤
            top_n:        重排序后保留数
        Returns:
            List[Tuple[Document, float]]
        """
        # Step 1: 生成假设性文档
        hyde_doc = self.generate_hypothetical_document(query)

        # Step 2: 用假设文档进行向量检索（而非原始问题）
        results = self.vs.retrieve(hyde_doc, k=k, top_n=top_n, file_filter=file_filter)

        logger.info("[HyDE] 检索完成，返回 %d 条结果", len(results))
        return results


# ======================== Multi-Query 检索策略 ========================

# 多查询改写 Prompt
MULTI_QUERY_PROMPT = """你是一个检索查询改写助手。请将以下问题改写为 3 个不同角度的子问题，
用于从学术论文中检索相关信息。每个子问题应该从不同角度切入，覆盖问题的不同方面。

要求：
1. 生成恰好 3 个子问题
2. 每行一个，不要编号
3. 保持与原问题相同的语言
4. 只输出子问题，不要添加解释

原问题：{question}

子问题："""


class MultiQueryRetrieval(BaseRetrievalStrategy):
    """
    Multi-Query 检索策略。

    流程:
      1. 让 LLM 将问题改写为多个子问题
      2. 对每个子问题分别进行向量检索
      3. 合并结果，去重并按 Reciprocal Rank Fusion 排序

    优势: 多角度检索扩大召回率，减少单一查询的偏差。
    """

    def __init__(self, vector_store: VectorStore, llm: ChatTongyi | None = None,
                 num_queries: int = 3) -> None:
        super().__init__(vector_store)
        self.llm = llm or ChatTongyi(
            model=Config.LLM_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
            temperature=0.3,
        )
        self.num_queries = num_queries
        self.multi_query_prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
        self.multi_query_chain = self.multi_query_prompt | self.llm | StrOutputParser()

    @property
    def name(self) -> str:
        return "multi_query"

    def generate_queries(self, query: str) -> list[str]:
        """
        让 LLM 生成多个子查询。

        Args:
            query: 原始查询
        Returns:
            子查询列表（包含原始查询）
        """
        logger.info("[MultiQuery] 正在生成子查询 ...")
        try:
            result = self.multi_query_chain.invoke({"question": query})
            # 按行分割，过滤空行
            sub_queries = [
                line.strip()
                for line in result.strip().split("\n")
                if line.strip()
            ]
            # 确保原始查询也在列表中
            all_queries = [query] + sub_queries[:self.num_queries]
            logger.info("[MultiQuery] 生成 %d 个查询", len(all_queries))
            for i, q in enumerate(all_queries):
                logger.debug("[MultiQuery] Q%d: %s", i, q)
            return all_queries
        except Exception as e:
            logger.warning("[MultiQuery] 生成子查询失败: %s，使用原始查询", e)
            return [query]

    def _reciprocal_rank_fusion(self, result_lists: list[list[tuple[Document, float]]],
                                k: int = 60) -> list[tuple[Document, float]]:
        """
        Reciprocal Rank Fusion (RRF) 算法。
        将多个排序列表融合为一个排序列表。

        Args:
            result_lists: 多个检索结果列表
            k:            RRF 平滑参数（默认 60）
        Returns:
            融合后的 (Document, rrf_score) 列表
        """
        # 用文档内容的 hash 作为去重 key
        doc_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for results in result_lists:
            for rank, (doc, score) in enumerate(results):
                # 用内容前 100 字符作为唯一标识
                key = doc.page_content[:100]
                doc_map[key] = doc
                doc_scores[key] += 1.0 / (k + rank + 1)

        # 按 RRF 分数降序排列
        fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[key], score) for key, score in fused]

    def retrieve(self, query: str, k: int = 10, file_filter: list[str] | None = None,
                 top_n: int | None = None, **kwargs) -> list[tuple[Document, float]]:
        """
        Multi-Query 检索流程: 生成子查询 → 分别检索 → RRF 融合 → 取 Top-N。

        Args:
            query:        用户查询
            k:            每个子查询的向量检索候选数
            file_filter:  按文件名过滤
            top_n:        最终保留数
        Returns:
            List[Tuple[Document, float]]
        """
        # Step 1: 生成子查询
        queries = self.generate_queries(query)

        # Step 2: 对每个子查询进行向量检索
        all_results = []
        for q in queries:
            results = self.vs.search(q, k=k, file_filter=file_filter)
            all_results.append(results)

        # Step 3: RRF 融合
        fused = self._reciprocal_rank_fusion(all_results)

        # Step 4: 取 Top-N（如果指定了 top_n，还需要重排序）
        if top_n:
            # 尝试用 rerank 对融合后的结果重排序
            documents = [doc for doc, _ in fused[:max(top_n * 2, k)]]
            scores = [score for _, score in fused[:max(top_n * 2, k)]]
            reranked = self.vs.rerank(query, documents, top_n=top_n, scores=scores)
            logger.info("[MultiQuery] RRF 融合 + 重排序完成，返回 %d 条", len(reranked))
            return reranked

        logger.info("[MultiQuery] RRF 融合完成，返回 %d 条", len(fused[:k]))
        return fused[:k]


# ======================== Hybrid 检索策略 ========================

class HybridRetrieval(BaseRetrievalStrategy):
    """
    混合检索策略: 向量检索 + BM25 关键词检索 + RRF 融合。

    流程:
      1. 向量检索（语义相似度）
      2. BM25 检索（关键词精确匹配）
      3. Reciprocal Rank Fusion 融合两路结果
      4. 对融合结果重排序

    优势: 向量检索擅长语义匹配，BM25 擅长精确关键词匹配，互补提升。
    """

    def __init__(self, vector_store: VectorStore, bm25_k1: float = 1.5,
                 bm25_b: float = 0.75) -> None:
        super().__init__(vector_store)
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self._bm25_index = None
        self._bm25_docs = None

    @property
    def name(self) -> str:
        return "hybrid"

    def _build_bm25_index(self, all_docs: list[Document]) -> None:
        """构建 BM25 索引。"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("[Hybrid] rank_bm25 未安装，Hybrid 检索将退化为纯向量检索")
            self._bm25_index = None
            return

        # 简单分词：按空格和标点分割
        import re
        tokenized = []
        for doc in all_docs:
            # 中英文混合分词：英文按空格，中文按字符
            text = doc.page_content.lower()
            tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text)
            tokenized.append(tokens)

        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_docs = all_docs
        logger.info("[Hybrid] BM25 索引构建完成，%d 篇文档", len(all_docs))

    def _bm25_search(self, query: str, k: int = 10,
                     file_filter: list[str] | None = None) -> list[tuple[Document, float]]:
        """
        BM25 关键词检索。

        Args:
            query:        查询
            k:            返回数量
            file_filter:  文件名过滤
        Returns:
            List[Tuple[Document, float]]
        """
        if self._bm25_index is None or self._bm25_docs is None:
            return []

        import re
        query_tokens = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', query.lower())
        scores = self._bm25_index.get_scores(query_tokens)

        # 按 BM25 分数排序
        scored = list(zip(self._bm25_docs, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 文件过滤
        if file_filter:
            filter_set = set(file_filter)
            scored = [
                (doc, score) for doc, score in scored
                if doc.metadata.get("file_name", "") in filter_set
                or os.path.basename(doc.metadata.get("source", "")) in filter_set
            ]

        return scored[:k]

    def _reciprocal_rank_fusion(self, result_lists: list[list[tuple[Document, float]]],
                                k: int = 60) -> list[tuple[Document, float]]:
        """Reciprocal Rank Fusion 融合算法。"""
        doc_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for results in result_lists:
            for rank, (doc, score) in enumerate(results):
                key = doc.page_content[:100]
                doc_map[key] = doc
                doc_scores[key] += 1.0 / (k + rank + 1)

        fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[key], score) for key, score in fused]

    def retrieve(self, query: str, k: int = 10, file_filter: list[str] | None = None,
                 top_n: int | None = None, **kwargs) -> list[tuple[Document, float]]:
        """
        Hybrid 检索流程: 向量检索 + BM25 检索 → RRF 融合 → 重排序。

        Args:
            query:        用户查询
            k:            每路检索的候选数
            file_filter:  按文件名过滤
            top_n:        最终保留数
        Returns:
            List[Tuple[Document, float]]
        """
        # Step 1: 向量检索
        vector_results = self.vs.search(query, k=k, file_filter=file_filter)
        logger.info("[Hybrid] 向量检索返回 %d 条", len(vector_results))

        # Step 2: BM25 检索（延迟构建索引）
        if self._bm25_index is None:
            # 从向量库中获取所有文档构建 BM25 索引
            all_docs = list(self.vs.vectorstore.docstore._dict.values())
            self._build_bm25_index(all_docs)

        bm25_results = self._bm25_search(query, k=k, file_filter=file_filter)
        logger.info("[Hybrid] BM25 检索返回 %d 条", len(bm25_results))

        if not bm25_results:
            # BM25 不可用，退化为纯向量检索 + 重排序
            logger.warning("[Hybrid] BM25 不可用，退化为纯向量检索")
            return self.vs.retrieve(query, k=k, top_n=top_n, file_filter=file_filter)

        # Step 3: RRF 融合
        fused = self._reciprocal_rank_fusion([vector_results, bm25_results])

        # Step 4: 取 Top-N 并重排序
        if top_n:
            documents = [doc for doc, _ in fused[:max(top_n * 2, k)]]
            scores = [score for _, score in fused[:max(top_n * 2, k)]]
            reranked = self.vs.rerank(query, documents, top_n=top_n, scores=scores)
            logger.info("[Hybrid] RRF 融合 + 重排序完成，返回 %d 条", len(reranked))
            return reranked

        return fused[:k]


# ======================== 策略工厂 ========================

STRATEGY_REGISTRY = {
    "vector": VectorRetrieval,
    "hyde": HyDERetrieval,
    "multi_query": MultiQueryRetrieval,
    "hybrid": HybridRetrieval,
}

STRATEGY_DESCRIPTIONS = {
    "vector": "标准向量检索 + 重排序（默认）",
    "hyde": "HyDE 假设性文档嵌入检索",
    "multi_query": "多查询检索（LLM 改写 + RRF 融合）",
    "hybrid": "混合检索（向量 + BM25 + RRF 融合）",
}


def get_strategy(name: str, vector_store: VectorStore, **kwargs) -> BaseRetrievalStrategy:
    """
    根据名称获取检索策略实例。

    Args:
            name:         策略名称 (vector / hyde / multi_query / hybrid)
            vector_store: VectorStore 实例
            **kwargs:     传递给策略构造函数的额外参数
    Returns:
            BaseRetrievalStrategy 实例
    Raises:
            ValueError: 未知策略名称
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"未知检索策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(vector_store, **kwargs)
