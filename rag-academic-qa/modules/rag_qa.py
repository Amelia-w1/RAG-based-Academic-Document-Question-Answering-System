"""
RAG 问答模块
==================
自定义 Prompt 限制大模型仅依据检索上下文作答，减少幻觉。
返回回答 + 引用段落（含文件名、页码），便于核对原文。
支持多轮对话（带历史记忆）和高级检索策略。

核心功能:
  1. ask()              — 单次/多轮问答（检索 → 重排序 → 生成 → 引用溯源）
  2. ask_stream()       — 流式问答（逐 token 返回）
  3. format_context()   — 将检索结果格式化为带引用标记的上下文
  4. extract_citations() — 从检索结果中提取引用信息
  5. rewrite_query()    — 结合对话历史改写查询（多轮场景）
"""

import os
import sys
from collections.abc import Iterator

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from modules.vector_store import VectorStore
from modules.conversation import ConversationMemory
from utils.logger import get_logger

logger = get_logger(__name__)


# ======================== 自定义 Prompt 模板 ========================
ANSWER_STYLE_GUIDE = """【输出要求 - 必须遵守】
1. 保持学术问答风格，回答要简洁、准确、专业。
2. 默认使用固定结构输出：
   - 结论：先直接回答问题核心。
   - 依据：再说明文献中的关键证据与推理。
   - 小结：最后用一句话收束。
3. 在正文中使用 [片段N] 标注依据来源，N 必须对应上方上下文中的片段编号。
4. 不要编造上下文里没有的实验结果、结论或页码。
5. 不要在正文末尾重复输出完整引用清单；系统会单独展示引用列表。
6. 回答语言与提问语言保持一致（中文问中文答，英文问英文答）。
"""

SYSTEM_PROMPT = """你是一个专业的学术论文问答助手，专注于图像复原领域文献。

【回答规则 — 必须严格遵守】
1. 仅根据下方"检索到的文献上下文"回答问题，禁止使用任何外部知识。
2. 如果上下文不足以回答问题，明确回复:"根据已有文献，无法回答该问题。"
3. 在正文中使用 [片段N] 标记依据来源，对应上下文中的编号。
4. 采用固定的学术回答结构：结论 / 依据 / 小结。
5. 回答需准确、专业、有条理，避免笼统概括；在上下文允许的情况下，尽量给出充分、完整的阐述。
6. 不要在正文中重复输出完整引用清单，引用明细由系统单独返回。

""" + ANSWER_STYLE_GUIDE

# 通用知识兜底 Prompt：检索不到时允许模型用自身知识回答
FALLBACK_SYSTEM_PROMPT = """你是一个专业的学术论文问答助手，专注于图像复原领域文献。

【回答规则 — 通用知识兜底模式】
1. 优先根据下方"检索到的文献上下文"回答问题。
2. 如果上下文不足以回答问题，可以使用你的通用知识来回答，但必须在回答开头标注：【通用知识回答，非来自文献】。
3. 当基于文献上下文回答时，仍然使用 [片段N] 标记来源。
4. 采用固定的学术回答结构：结论 / 依据 / 小结。
5. 如果基于文献上下文回答，不要重复输出完整引用清单；系统会单独展示引用列表。
6. 如果纯通用知识回答且没有可用文献引用，可省略引用清单。

""" + ANSWER_STYLE_GUIDE

FALLBACK_HUMAN_PROMPT = """检索到的文献上下文（可能为空或相关性较低）：

{context}

问题：{question}

请基于上述上下文回答。如果上下文不足以回答，请使用通用知识回答并在开头标注【通用知识回答，非来自文献】。如果基于文献上下文回答，请在正文中使用 [片段N] 标注依据来源。"""

# 多轮对话版本（兜底模式）
FALLBACK_HUMAN_PROMPT_WITH_HISTORY = """以下是之前的对话历史（供理解上下文参考）：

{history}

检索到的文献上下文（可能为空或相关性较低）：

{context}

问题：{question}

请基于上述上下文回答。如果上下文不足以回答，请使用通用知识回答并在开头标注【通用知识回答，非来自文献】。如果基于文献上下文回答，请在正文中使用 [片段N] 标注依据来源。"""

HUMAN_PROMPT = """检索到的文献上下文：

{context}

问题：{question}

请基于上述上下文回答。正文采用“结论 / 依据 / 小结”的固定结构，并在关键论断后使用 [片段N] 标注来源。"""

# 多轮对话 Prompt（包含历史上下文）
HUMAN_PROMPT_WITH_HISTORY = """以下是之前的对话历史（供理解上下文参考，回答仍须基于下方检索到的文献）：

{history}

检索到的文献上下文：

{context}

问题：{question}

请基于上述上下文回答。如果问题涉及之前的对话内容，可以结合上下文理解问题意图，但回答必须基于检索到的文献。正文采用“结论 / 依据 / 小结”的固定结构，并在关键论断后使用 [片段N] 标注来源。"""

# 查询改写 Prompt（将多轮对话中的指代消解为独立查询）
QUERY_REWRITE_PROMPT = """请根据以下对话历史，将用户的最新问题改写为一个独立、完整的检索查询。

要求：
1. 消解代词和指代（如"它"、"这个方法"等）
2. 保持问题的核心意图不变
3. 只输出改写后的查询，不要添加任何解释

对话历史：
{history}

用户最新问题：{question}

改写后的独立查询："""


class RAGQA:
    """RAG 问答引擎，集成检索、重排序与受限生成，支持多轮对话。"""

    def __init__(self, vector_store: VectorStore) -> None:
        """
        Args:
            vector_store: VectorStore 实例（已加载索引）
        """
        self.vs = vector_store
        self.llm = ChatTongyi(
            model=Config.LLM_MODEL,
            dashscope_api_key=Config.DASHSCOPE_API_KEY,
            temperature=Config.TEMPERATURE,
        )
        # 单轮对话 Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        # 多轮对话 Prompt（带历史）
        self.prompt_with_history = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT_WITH_HISTORY),
        ])
        # 通用知识兜底 Prompt（单轮）
        self.fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", FALLBACK_SYSTEM_PROMPT),
            ("human", FALLBACK_HUMAN_PROMPT),
        ])
        # 通用知识兜底 Prompt（多轮）
        self.fallback_prompt_with_history = ChatPromptTemplate.from_messages([
            ("system", FALLBACK_SYSTEM_PROMPT),
            ("human", FALLBACK_HUMAN_PROMPT_WITH_HISTORY),
        ])
        # 查询改写链
        self.rewrite_prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()
        # 构建链: Prompt → LLM → 字符串输出
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.chain_with_history = self.prompt_with_history | self.llm | StrOutputParser()
        self.fallback_chain = self.fallback_prompt | self.llm | StrOutputParser()
        self.fallback_chain_with_history = (
            self.fallback_prompt_with_history | self.llm | StrOutputParser()
        )

    def format_context(self, docs_with_scores: list[tuple[Document, float]]) -> str:
        """
        将检索结果格式化为带引用标记的上下文字符串。
        每个片段标注 [片段N]、来源文件名、页码。

        Args:
            docs_with_scores: List[Tuple[Document, float]]
        Returns:
            格式化后的上下文字符串
        """
        blocks = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            file_name = doc.metadata.get("file_name",
                       os.path.basename(doc.metadata.get("source", "Unknown")))
            page_label = doc.metadata.get("page_label",
                         doc.metadata.get("page", 0) + 1)
            content = doc.page_content.strip()

            block = (
                f"[片段{i}] 来源: {file_name} | 页码: {page_label} | 相关性: {score:.4f}\n"
                f"{content}"
            )
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)

    def extract_citations(self, docs_with_scores: list[tuple[Document, float]]) -> list[dict]:
        """
        从检索结果中提取引用信息（文件名、页码、原文片段）。

        Args:
            docs_with_scores: List[Tuple[Document, float]]
        Returns:
            List[dict]，每个 dict 包含 fragment_id / file_name / page / content / score
        """
        citations = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            file_name = doc.metadata.get("file_name",
                       os.path.basename(doc.metadata.get("source", "Unknown")))
            page_label = doc.metadata.get("page_label",
                         doc.metadata.get("page", 0) + 1)

            citations.append({
                "fragment_id": i,
                "file_name": file_name,
                "page": page_label,
                "content": doc.page_content.strip(),
                "relevance_score": round(score, 4),
            })
        return citations

    def rewrite_query(self, question: str, memory: ConversationMemory | None = None) -> str:
        """
        根据对话历史改写查询，消解代词和指代。

        例如:
          历史: "BM3D 算法的原理是什么？" → "BM3D 是一种去噪方法..."
          当前: "它的计算复杂度怎样？"
          改写: "BM3D 算法的计算复杂度怎样？"

        Args:
            question: 用户当前问题
            memory:   对话历史（None 或空历史则直接返回原问题）
        Returns:
            改写后的独立查询
        """
        if memory is None or memory.get_turn_count() == 0:
            return question

        try:
            history_text = memory.get_history_text()
            rewritten = self.rewrite_chain.invoke({
                "history": history_text,
                "question": question,
            })
            rewritten = rewritten.strip()
            logger.info("查询改写: '%s' → '%s'", question, rewritten)
            return rewritten if rewritten else question
        except Exception as e:
            logger.warning("查询改写失败，使用原始问题: %s", e)
            return question

    def ask(self, question: str, k: int | None = None, top_n: int | None = None,
            file_filter: list[str] | None = None,
            memory: ConversationMemory | None = None,
            use_history: bool = True,
            fallback_enabled: bool = False) -> dict:
        """
        完整 RAG 问答流程:
          1. (可选) 查询改写 → 2. 向量检索 → 3. 重排序 → 4. 格式化上下文 → 5. LLM 生成 → 6. 提取引用

        Args:
            question:         用户问题
            k:                向量检索候选数（默认 Config.RETRIEVAL_K）
            top_n:            重排序后保留数（默认 Config.RERANK_TOP_N）
            file_filter:      按文件名过滤（None 表示不过滤）
            memory:           对话历史（None 表示单轮模式）
            use_history:      是否在生成时注入对话历史
            fallback_enabled: 启用通用知识兜底（检索不到时用 LLM 自身知识回答）
        Returns:
            dict: {
                "question":        原始问题,
                "rewritten_query": 改写后的查询（无改写则与 question 相同）,
                "answer":          LLM 回答,
                "citations":       引用列表,
                "context":         格式化上下文（调试用）
            }
        """
        # Step 0: 查询改写（多轮对话场景）
        search_query = question
        if use_history and memory and memory.get_turn_count() > 0:
            search_query = self.rewrite_query(question, memory)
            logger.info("使用改写查询进行检索: %s", search_query)

        # Step 1+2: 向量检索 + 重排序
        docs_with_scores = self.vs.retrieve(search_query, k=k, top_n=top_n, file_filter=file_filter)

        # 未检索到结果
        if not docs_with_scores:
            if fallback_enabled:
                # 兜底模式：用通用知识回答
                context = "（未检索到相关文献片段）"
                use_hist = use_history and memory and memory.get_turn_count() > 0
                if use_hist:
                    chain = self.fallback_prompt_with_history | self.llm | StrOutputParser()
                    answer = chain.invoke({
                        "context": context,
                        "question": question,
                        "history": memory.get_history_text(),
                    })
                else:
                    chain = self.fallback_prompt | self.llm | StrOutputParser()
                    answer = chain.invoke({
                        "context": context,
                        "question": question,
                    })
                return {
                    "question": question,
                    "rewritten_query": search_query,
                    "answer": answer,
                    "citations": [],
                    "context": context,
                }
            return {
                "question": question,
                "rewritten_query": search_query,
                "answer": "未检索到相关文献片段，无法回答。",
                "citations": [],
                "context": "",
            }

        # Step 3: 格式化上下文
        context = self.format_context(docs_with_scores)

        # Step 4: LLM 生成
        use_hist = use_history and memory and memory.get_turn_count() > 0
        if fallback_enabled:
            # 兜底模式：优先文献，不足时用通用知识
            if use_hist:
                chain = self.fallback_prompt_with_history | self.llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context,
                    "question": question,
                    "history": memory.get_history_text(),
                })
            else:
                chain = self.fallback_prompt | self.llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context,
                    "question": question,
                })
        elif use_hist:
            history_text = memory.get_history_text()
            chain_with_history = self.prompt_with_history | self.llm | StrOutputParser()
            answer = chain_with_history.invoke({
                "context": context,
                "question": question,
                "history": history_text,
            })
        else:
            answer = self.chain.invoke({
                "context": context,
                "question": question,
            })

        # Step 5: 提取引用
        citations = self.extract_citations(docs_with_scores)

        return {
            "question": question,
            "rewritten_query": search_query,
            "answer": answer,
            "citations": citations,
            "context": context,
        }

    def ask_stream(self, question: str, k: int | None = None, top_n: int | None = None,
                   file_filter: list[str] | None = None,
                   memory: ConversationMemory | None = None,
                   use_history: bool = True,
                   fallback_enabled: bool = False) -> Iterator[str]:
        """
        流式问答 — 逐 token 返回回答（适合交互式场景）。
        支持多轮对话历史注入和通用知识兜底。

        Args:
            question:         用户问题
            k:                向量检索候选数
            top_n:            重排序后保留数
            file_filter:      按文件名过滤（None 表示不过滤）
            memory:           对话历史（None 表示单轮模式）
            use_history:      是否在生成时注入对话历史
            fallback_enabled: 启用通用知识兜底
        Yields:
            str: 回答的每个 token
        """
        # Step 0: 查询改写
        search_query = question
        if use_history and memory and memory.get_turn_count() > 0:
            search_query = self.rewrite_query(question, memory)

        docs_with_scores = self.vs.retrieve(search_query, k=k, top_n=top_n, file_filter=file_filter)

        if not docs_with_scores:
            if fallback_enabled:
                context = "（未检索到相关文献片段）"
                use_hist = use_history and memory and memory.get_turn_count() > 0
                if use_hist:
                    chain = self.fallback_prompt_with_history | self.llm | StrOutputParser()
                    stream = chain.stream({
                        "context": context,
                        "question": question,
                        "history": memory.get_history_text(),
                    })
                else:
                    chain = self.fallback_prompt | self.llm | StrOutputParser()
                    stream = chain.stream({
                        "context": context,
                        "question": question,
                    })
                for chunk in stream:
                    yield chunk
                return
            yield "未检索到相关文献片段，无法回答。"
            return

        context = self.format_context(docs_with_scores)
        use_hist = use_history and memory and memory.get_turn_count() > 0

        # 选择链
        if fallback_enabled:
            if use_hist:
                chain = self.fallback_prompt_with_history | self.llm | StrOutputParser()
                stream = chain.stream({
                    "context": context,
                    "question": question,
                    "history": memory.get_history_text(),
                })
            else:
                chain = self.fallback_prompt | self.llm | StrOutputParser()
                stream = chain.stream({
                    "context": context,
                    "question": question,
                })
        elif use_hist:
            chain = self.prompt_with_history | self.llm | StrOutputParser()
            stream = chain.stream({
                "context": context,
                "question": question,
                "history": memory.get_history_text(),
            })
        else:
            stream = self.chain.stream({
                "context": context,
                "question": question,
            })

        for chunk in stream:
            yield chunk
