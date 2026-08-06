"""
对话历史管理模块
==================
实现滑动窗口式对话历史管理，支持多轮上下文感知问答。

核心功能:
  1. add_user_message()    — 记录用户问题
  2. add_ai_message()      — 记录 AI 回答
  3. get_history()         — 获取格式化的对话历史
  4. clear()               — 清空历史
  5. get_token_estimate()  — 估算历史 token 数量
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Message:
    """单条对话消息。"""
    role: str          # "user" 或 "assistant"
    content: str       # 消息内容
    timestamp: float = 0.0  # 时间戳（可选）


class ConversationMemory:
    """
    对话历史管理器 — 滑动窗口策略。

    当对话轮数超过 max_turns 时，自动丢弃最早的对话，
    避免上下文过长导致 token 超限或性能下降。

    Attributes:
        max_turns:     保留的最大对话轮数（1 轮 = 1 问 + 1 答）
        max_token_est: 估算的历史 token 上限（粗略：1 中文字 ≈ 1 token）
    """

    def __init__(self, max_turns: int = 5, max_token_est: int = 3000) -> None:
        self.max_turns = max_turns
        self.max_token_est = max_token_est
        self._messages: deque[Message] = deque(maxlen=max_turns * 2)

    def add_user_message(self, content: str) -> None:
        """记录用户消息。"""
        import time
        self._messages.append(Message(role="user", content=content, timestamp=time.time()))
        self._trim_by_tokens()
        logger.debug("记录用户消息，当前历史 %d 条", len(self._messages))

    def add_ai_message(self, content: str) -> None:
        """记录 AI 回答。"""
        import time
        self._messages.append(Message(role="assistant", content=content, timestamp=time.time()))
        self._trim_by_tokens()
        logger.debug("记录 AI 回答，当前历史 %d 条", len(self._messages))

    def get_history(self) -> list[Message]:
        """
        获取当前对话历史列表。

        Returns:
            List[Message]，按时间顺序排列
        """
        return list(self._messages)

    def get_history_text(self) -> str:
        """
        将对话历史格式化为纯文本（用于注入 Prompt）。

        Returns:
            格式化后的历史字符串，如:
            [用户] 什么是 BM3D？
            [助手] BM3D 是一种...
        """
        if not self._messages:
            return ""

        lines = []
        for msg in self._messages:
            prefix = "[用户]" if msg.role == "user" else "[助手]"
            # 截断过长的历史消息
            content = msg.content[:500] if len(msg.content) > 500 else msg.content
            lines.append(f"{prefix} {content}")

        return "\n".join(lines)

    def get_history_messages(self) -> list[dict]:
        """
        将对话历史格式化为 LangChain 消息列表格式。

        Returns:
            List[dict]，每个 dict 包含 role 和 content
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self._messages
        ]

    def clear(self) -> None:
        """清空所有对话历史。"""
        self._messages.clear()
        logger.info("对话历史已清空")

    def get_turn_count(self) -> int:
        """返回当前对话轮数（1 轮 = 1 问 + 1 答）。"""
        return len(self._messages) // 2

    def get_token_estimate(self) -> int:
        """估算当前历史的 token 数量（粗略：字符数 / 2）。"""
        return sum(len(msg.content) for msg in self._messages) // 2

    def _trim_by_tokens(self) -> None:
        """当历史 token 估算超过上限时，从最早的消息开始丢弃。"""
        while self.get_token_estimate() > self.max_token_est and len(self._messages) > 2:
            removed = self._messages.popleft()
            logger.debug("丢弃旧消息以控制 token: %s...", removed.content[:50])

    def to_dict(self) -> dict:
        """序列化为字典（用于持久化）。"""
        return {
            "max_turns": self.max_turns,
            "max_token_est": self.max_token_est,
            "messages": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in self._messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConversationMemory:
        """从字典反序列化。"""
        mem = cls(
            max_turns=data.get("max_turns", 5),
            max_token_est=data.get("max_token_est", 3000),
        )
        for msg_data in data.get("messages", []):
            mem._messages.append(Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp", 0.0),
            ))
        return mem
