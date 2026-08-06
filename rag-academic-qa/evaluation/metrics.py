"""
评估指标实现
==================
每个指标使用 LLM-as-Judge 方法打分（0-1 分）。

指标说明:
  1. Faithfulness: 回答是否完全基于检索上下文，没有编造信息
  2. Answer Relevancy: 回答是否直接针对用户问题，不跑题
  3. Context Precision: 检索到的上下文中有多少是真正相关的
  4. Context Recall: 检索到的上下文是否覆盖了回答问题所需的信息
"""

from __future__ import annotations

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.logger import get_logger

logger = get_logger(__name__)


# ======================== Prompt 模板 ========================

FAITHFULNESS_PROMPT = """你是一个严格的评估员。请判断以下回答是否完全忠实于给定的上下文。

【评估标准】
- 回答中的每个事实陈述都必须能在上下文中找到对应
- 如果回答包含上下文中没有的信息（编造/幻觉），扣分
- 如果回答与上下文矛盾，扣分
- 回答可以对上下文信息进行合理总结和组织，但不能添加新信息

【上下文】
{context}

【回答】
{answer}

【问题】
{question}

请只输出一个 0 到 1 之间的分数（保留两位小数），1 表示完全忠实，0 表示完全不忠实。
不要输出任何其他内容。

分数："""


ANSWER_RELEVANCY_PROMPT = """你是一个专业的评估员。请评估以下回答与问题的相关程度。

【评估标准】
- 回答是否直接针对问题，没有跑题
- 回答是否提供了有价值的信息
- 回答是否完整地回应了问题的各个方面
- 回答不包含与问题无关的冗余信息

【问题】
{question}

【回答】
{answer}

请只输出一个 0 到 1 之间的分数（保留两位小数），1 表示完全相关，0 表示完全不相关。
不要输出任何其他内容。

分数："""


CONTEXT_PRECISION_PROMPT = """你是一个检索质量评估员。请评估以下检索到的上下文片段对回答问题的精确度。

【评估标准】
- 检索到的上下文中有多少片段真正与问题相关
- 相关片段在排序列表中的位置（越靠前越好）
- 不相关片段是否占据了高位

【问题】
{question}

【检索到的上下文片段】（按相关性排序）
{context}

请只输出一个 0 到 1 之间的分数（保留两位小数），1 表示所有片段都高度相关且排序正确，0 表示完全不相关。
不要输出任何其他内容。

分数："""


CONTEXT_RECALL_PROMPT = """你是一个检索质量评估员。请评估检索到的上下文是否覆盖了回答问题所需的信息。

【评估标准】
- 上下文是否包含回答问题所需的关键信息
- 是否有重要的信息遗漏
- 上下文的信息是否足以支撑一个完整的回答

【问题】
{question}

【检索到的上下文】
{context}

【参考回答】（用于判断需要哪些信息）
{answer}

请只输出一个 0 到 1 之间的分数（保留两位小数），1 表示完全覆盖，0 表示完全未覆盖。
不要输出任何其他内容。

分数："""


# ======================== 指标类 ========================

class Metric:
    """单个评估指标基类。"""

    name: str = ""
    description: str = ""

    def __init__(self, llm: ChatTongyi) -> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    template: str = ""

    def score(self, **kwargs) -> float:
        """打分（0-1）。"""
        try:
            result = self.chain.invoke(kwargs)
            # 提取分数
            text = result.strip()
            # 尝试解析浮点数
            for token in text.split():
                try:
                    score = float(token)
                    if 0 <= score <= 1:
                        return score
                except ValueError:
                    continue
            # 如果无法解析，尝试提取数字
            import re
            numbers = re.findall(r'[\d.]+', text)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            logger.warning("%s 无法解析分数: %s", self.name, text)
            return 0.0
        except Exception as e:
            logger.error("%s 评估失败: %s", self.name, e)
            return 0.0


class FaithfulnessMetric(Metric):
    """忠实度: 回答是否忠实于检索上下文。"""
    name = "faithfulness"
    description = "回答对上下文的忠实度"
    template = FAITHFULNESS_PROMPT

    def score(self, question: str, answer: str, context: str) -> float:
        return super().score(question=question, answer=answer, context=context)


class AnswerRelevancyMetric(Metric):
    """答案相关性: 回答与问题的相关程度。"""
    name = "answer_relevancy"
    description = "回答与问题的相关性"
    template = ANSWER_RELEVANCY_PROMPT

    def score(self, question: str, answer: str) -> float:
        return super().score(question=question, answer=answer)


class ContextPrecisionMetric(Metric):
    """上下文精确率: 检索结果的精确度。"""
    name = "context_precision"
    description = "检索上下文的精确率"
    template = CONTEXT_PRECISION_PROMPT

    def score(self, question: str, context: str) -> float:
        return super().score(question=question, context=context)


class ContextRecallMetric(Metric):
    """上下文召回率: 检索结果覆盖问题的程度。"""
    name = "context_recall"
    description = "检索上下文的召回率"
    template = CONTEXT_RECALL_PROMPT

    def score(self, question: str, context: str, answer: str) -> float:
        return super().score(question=question, context=context, answer=answer)
