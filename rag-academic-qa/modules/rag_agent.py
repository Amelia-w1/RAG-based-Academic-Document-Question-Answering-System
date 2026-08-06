"""
RAG 多智能体编排模块（基于 LangGraph）
========================================
将原有"检索 → 生成"的线性 RAG 流程，显式建模为一组协作的智能体（Agent），
通过 LangGraph 的状态图（StateGraph）进行编排，使每个职责成为一个可观测、
可分支、可扩展的节点：

  1. QueryAgent    （查询理解智能体）：结合对话历史改写查询，并依据问题特征
                    自主选择检索策略（vector / hybrid / multi_query / hyde）。
  2. RetrieveAgent （检索智能体）    ：按选定策略执行向量 / 混合 / 多查询检索
                    并融合重排序，返回候选文献片段。
  3. JudgeAgent    （质量判断智能体）：评估检索结果与问题的相关性，决定是否
                    进入"生成"还是触发"通用知识兜底"，形成条件分支。
  4. GenerateAgent （生成智能体）    ：基于文献上下文（或兜底模式）调用大模型
                    生成带 [片段N] 引用的回答，支持流式输出。
  5. CiteAgent     （引用校验智能体）：从检索结果中提取引用元数据，过滤并整理
                    与回答对应的引用清单。

设计要点：
  - 检索子图（Query → Retrieve → Judge）同步执行，产出 citations 后先返回给 UI，
    保证"引用卡片"在回答流式输出前即可渲染（满足 GUI 的流式契约）。
  - 生成子图（Generate → Cite）通过 graph.stream(stream_mode="messages") 流式
    输出大模型 token，兼容原有逐字显示体验。
  - 复用 rag_qa 的 Prompt / 链、advanced_retrieval 的检索策略与 vector_store 的
    重排序能力，不重复造轮子，且行为与原 RAG 管线一致。

对外接口（与 RAGQA 对齐，便于平滑替换）：
  - ask(question, memory) -> dict          完整问答（含 answer / citations）
  - ask_stream(question, plan, memory)     生成阶段流式输出（Iterator[str]）
  - retrieve_plan(question, memory) -> dict 仅跑检索子图，返回 citations 与决策
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from modules.vector_store import VectorStore
from modules.rag_qa import RAGQA
from modules.advanced_retrieval import get_strategy
from utils.logger import get_logger

logger = get_logger(__name__)


# ======================== 质量判断 Prompt ========================
JUDGE_PROMPT = """你是一个检索质量评审员。下面给出用户问题与检索到的文献片段摘要。
请判断这些片段是否足以回答用户问题。

用户问题：
{question}

检索到的文献片段（摘要）：
{context}

评审规则：
- 若片段中包含回答该问题所需的关键信息，回复：sufficient
- 若片段与问题明显无关或信息严重不足，回复：insufficient
只回复 sufficient 或 insufficient，不要解释。"""


class AgentState(TypedDict, total=False):
    """多智能体工作流在节点间传递的状态。"""
    question: str
    history_text: str
    rewritten_query: str
    strategy: str
    docs: list[tuple[Document, float]]
    retrieval_decision: str   # "generate" | "fallback"
    fallback: bool
    answer: str
    citations: list[dict]
    trace: list[str]          # 各智能体的决策轨迹，便于可观测


class RAGAgent:
    """
    基于 LangGraph 的多智能体 RAG 编排器。

    将查询理解、检索、质量判断、生成、引用校验建模为五个协作智能体，
    通过状态图串联，JudgeAgent 处形成"生成 / 兜底"条件分支。
    """

    def __init__(self, vector_store: VectorStore,
                 strategy_name: str = "vector",
                 use_history: bool = True,
                 fallback_enabled: bool = False,
                 k: int | None = None,
                 top_n: int | None = None,
                 file_filter: list[str] | None = None,
                 llm=None) -> None:
        """
        Args:
            vector_store:    已加载索引的 VectorStore 实例
            strategy_name:   默认检索策略（vector/hyde/multi_query/hybrid）
            use_history:     是否注入多轮对话历史
            fallback_enabled:是否允许通用知识兜底
            k / top_n:       检索 / 重排序参数（None 取 Config 默认）
            file_filter:     按文件名过滤检索结果
            llm:             可选，注入自定义/测试用大模型
        """
        self.vs = vector_store
        self.strategy_name = strategy_name or "vector"
        self.use_history = use_history
        self.fallback_enabled = fallback_enabled
        self._k = k or Config.RETRIEVAL_K
        self._top_n = top_n or Config.RERANK_TOP_N
        self._file_filter = file_filter or []
        # 复用 RAGQA 的 Prompt / 链 / 改写 / 引用提取逻辑
        self.qa = RAGQA(vector_store)
        self.llm = llm or self.qa.llm
        self._memory = None
        self._retrieval_graph = self._build_retrieval_graph()
        self._generation_graph = self._build_generation_graph()

    # ======================== 图构建 ========================

    def _build_retrieval_graph(self):
        """构建检索子图：QueryAgent → RetrieveAgent → JudgeAgent。"""
        g = StateGraph(AgentState)
        g.add_node("query_agent", self._query_agent)
        g.add_node("retrieve_agent", self._retrieve_agent)
        g.add_node("judge_agent", self._judge_agent)
        g.add_edge(START, "query_agent")
        g.add_edge("query_agent", "retrieve_agent")
        g.add_edge("retrieve_agent", "judge_agent")
        g.add_edge("judge_agent", END)
        return g.compile()

    def _build_generation_graph(self):
        """构建生成子图：GenerateAgent → CiteAgent。"""
        g = StateGraph(AgentState)
        g.add_node("generate_agent", self._generate_agent)
        g.add_node("cite_agent", self._cite_agent)
        g.add_edge(START, "generate_agent")
        g.add_edge("generate_agent", "cite_agent")
        g.add_edge("cite_agent", END)
        return g.compile()

    # ======================== 检索子图节点 ========================

    def _query_agent(self, state: AgentState) -> dict:
        """QueryAgent：改写查询 + 选择检索策略。"""
        question = state["question"]
        rewritten = question
        if self.use_history and self._memory is not None and self._memory.get_turn_count() > 0:
            rewritten = self.qa.rewrite_query(question, self._memory)
        strategy = self._select_strategy(rewritten)
        logger.info("[QueryAgent] 改写: '%s' → '%s' | 策略: %s", question, rewritten, strategy)
        return {
            "rewritten_query": rewritten,
            "strategy": strategy,
            "trace": [f"QueryAgent: rewrite='{rewritten}' strategy={strategy}"],
        }

    def _select_strategy(self, query: str) -> str:
        """
        依据问题特征选择检索策略（QueryAgent 的决策逻辑）。
        若用户在构造时显式指定了非 vector 策略，则尊重用户选择。
        """
        if self.strategy_name != "vector":
            return self.strategy_name
        # 启发式：含比较/并列关系的问题更适合混合检索（向量+BM25 互补）
        compare_hints = ["比较", "对比", "区别", "差异", "和", "与", "以及", "还是", "vs", "VS"]
        if any(hint in query for hint in compare_hints):
            return "hybrid"
        return "vector"

    def _retrieve_agent(self, state: AgentState) -> dict:
        """RetrieveAgent：执行检索策略并融合重排序。"""
        strategy_name = state["strategy"]
        try:
            strategy = get_strategy(strategy_name, self.vs)
            docs = strategy.retrieve(
                state["rewritten_query"],
                k=self._k,
                top_n=self._top_n,
                file_filter=self._file_filter or None,
            )
        except Exception as e:
            logger.warning("[RetrieveAgent] 策略 %s 失败，回退标准向量检索: %s", strategy_name, e)
            docs = self.vs.retrieve(
                state["rewritten_query"],
                k=self._k,
                top_n=self._top_n,
                file_filter=self._file_filter or None,
            )
        logger.info("[RetrieveAgent] 策略 %s 检索返回 %d 条", strategy_name, len(docs))
        return {
            "docs": docs,
            "trace": [f"RetrieveAgent: strategy={strategy_name} docs={len(docs)}"],
        }

    def _judge_agent(self, state: AgentState) -> dict:
        """
        JudgeAgent：评估检索质量，决定生成还是兜底。

        决策规则：
          - 无检索结果 → 进入兜底（若开启）或直接告知无法回答
          - 有结果     → 调用 LLM 判断相关性；不足且未开启兜底时仍生成
                        （行为与原管线一致，避免过度拒答）
        """
        docs = state.get("docs") or []
        if not docs:
            logger.info("[JudgeAgent] 无检索结果 → fallback=%s", self.fallback_enabled)
            return {
                "retrieval_decision": "fallback",
                "fallback": self.fallback_enabled,
                "trace": ["JudgeAgent: no docs → fallback"],
            }
        sufficient = self._judge_quality(state["question"], docs)
        decision = "generate"
        logger.info("[JudgeAgent] sufficient=%s → decision=%s fallback=%s",
                    sufficient, decision, self.fallback_enabled)
        return {
            "retrieval_decision": decision,
            "fallback": self.fallback_enabled,
            "trace": [f"JudgeAgent: sufficient={sufficient} decision={decision}"],
        }

    def _judge_quality(self, question: str, docs: list[tuple[Document, float]]) -> bool:
        """用 LLM 判断检索片段是否足以回答问题；异常时默认充分。"""
        try:
            snippet = "\n".join(
                f"[片段{i}] {doc.page_content[:200]}"
                for i, (doc, _) in enumerate(docs[:3], 1)
            )
            prompt = JUDGE_PROMPT.format(question=question, context=snippet)
            resp = self.llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = text.strip().lower()
            if "insufficient" in text:
                return False
            return True
        except Exception as e:
            logger.warning("[JudgeAgent] LLM 判断失败，默认充分: %s", e)
            return True

    # ======================== 生成子图节点 ========================

    def _select_chain(self, state: AgentState):
        """根据兜底/历史标志选择完整生成链（Prompt → LLM → 字符串输出）。"""
        use_hist = self.use_history and state.get("history_text")
        if state.get("fallback"):
            chain = (
                self.qa.fallback_chain_with_history
                if use_hist
                else self.qa.fallback_chain
            )
        elif use_hist:
            chain = self.qa.chain_with_history
        else:
            chain = self.qa.chain
        return chain, use_hist

    def _generate_agent(self, state: AgentState) -> dict:
        """GenerateAgent：基于上下文生成带引用的回答。"""
        context = self.qa.format_context(state["docs"])
        chain, use_hist = self._select_chain(state)
        if use_hist:
            answer = chain.invoke({
                "context": context,
                "question": state["question"],
                "history": state["history_text"],
            })
        else:
            answer = chain.invoke({
                "context": context,
                "question": state["question"],
            })
        return {"answer": answer}

    def _cite_agent(self, state: AgentState) -> dict:
        """CiteAgent：提取并整理与回答对应的引用清单。"""
        citations = self.qa.extract_citations(state["docs"])
        trace = state.get("trace") or []
        trace = trace + ["CiteAgent: extracted %d citations" % len(citations)]
        return {"citations": citations, "trace": trace}

    # ======================== 对外接口 ========================

    def retrieve_plan(self, question: str, memory=None) -> dict:
        """
        仅执行检索子图，返回 citations 与决策（供 UI 在流式输出前渲染引用卡片）。

        Returns:
            dict（AgentState 子集）: rewritten_query, strategy, docs,
            retrieval_decision, fallback, citations, trace
        """
        self._memory = memory
        history_text = ""
        if self.use_history and memory is not None:
            history_text = memory.get_history_text()

        init: AgentState = {
            "question": question,
            "history_text": history_text,
            "rewritten_query": question,
            "strategy": self.strategy_name,
            "docs": [],
            "retrieval_decision": "generate",
            "fallback": self.fallback_enabled,
            "answer": "",
            "citations": [],
            "trace": [],
        }
        result = self._retrieval_graph.invoke(init)
        # 检索到结果即提取引用（无论最终是否兜底）
        if result.get("docs"):
            result["citations"] = self.qa.extract_citations(result["docs"])
        else:
            result["citations"] = []
        return result

    def generate_stream(self, question: str, plan: dict, memory=None) -> Iterator[str]:
        """
        执行生成并以流式方式产出回答 token（Iterator[str]）。

        复用 GenerateAgent 的链选择逻辑，直接通过 chain.stream 逐 token 输出，
        兼容 GUI 的逐字显示；引用卡片已在 retrieve_plan 阶段随 citations 先返回。
        """
        self._memory = memory
        chain, use_hist = self._select_chain(plan)
        if use_hist:
            stream = chain.stream({
                "context": self.qa.format_context(plan.get("docs", [])),
                "question": question,
                "history": plan.get("history_text", ""),
            })
        else:
            stream = chain.stream({
                "context": self.qa.format_context(plan.get("docs", [])),
                "question": question,
            })
        for token in stream:
            yield token

    def ask(self, question: str, memory=None) -> dict:
        """
        完整问答（非流式），返回与 RAGQA.ask 对齐的 dict。

        Returns:
            dict: question, rewritten_query, strategy, answer, citations,
                  retrieval_decision, trace
        """
        plan = self.retrieve_plan(question, memory)
        gen_input: AgentState = {
            "question": question,
            "history_text": plan.get("history_text", ""),
            "docs": plan.get("docs", []),
            "fallback": plan.get("fallback", self.fallback_enabled),
            "answer": "",
            "citations": [],
            "trace": plan.get("trace", []),
        }
        result = self._generation_graph.invoke(gen_input)
        return {
            "question": question,
            "rewritten_query": plan["rewritten_query"],
            "strategy": plan["strategy"],
            "answer": result.get("answer", ""),
            "citations": result.get("citations", plan["citations"]),
            "retrieval_decision": plan["retrieval_decision"],
            "trace": result.get("trace", []),
        }
