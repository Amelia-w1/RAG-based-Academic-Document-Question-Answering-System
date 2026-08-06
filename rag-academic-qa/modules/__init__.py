"""
RAG 学术文献问答系统 — 模块包
包含核心模块:
  - document_loader: 文档加载与分块
  - vector_store:    FAISS 向量库构建、加载、检索与重排序
  - rag_qa:          RAG 问答（自定义 Prompt + 引用溯源 + 多轮对话）
  - conversation:    对话历史管理（滑动窗口）
  - advanced_retrieval: 高级检索策略（HyDE / Multi-Query / Hybrid）
"""

from .document_loader import DocumentLoader
from .vector_store import VectorStore
from .rag_qa import RAGQA
from .rag_agent import RAGAgent
from .conversation import ConversationMemory

__all__ = ["DocumentLoader", "VectorStore", "RAGQA", "RAGAgent", "ConversationMemory"]
