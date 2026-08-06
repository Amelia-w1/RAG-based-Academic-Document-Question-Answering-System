"""
统一日志模块
================
提供控制台 + 文件双通道日志，所有模块通过 get_logger(__name__) 获取 logger。

日志级别:
  DEBUG   - 详细调试信息（分块详情、向量维度等）
  INFO    - 关键流程节点（加载文件、构建索引、检索完成）
  WARNING - 可恢复异常（rerank 降级、文件跳过）
  ERROR   - 不可恢复错误（API 认证失败、索引不存在）
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.getenv("LOG_DIR", os.path.join(_PROJECT_ROOT, "logs"))

# 全局标记，确保 handler 只初始化一次
_initialized = False


def _init_root_logger():
    """初始化根 logger 的 handler 与格式（仅执行一次）。"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    os.makedirs(_LOG_DIR, exist_ok=True)
    log_path = os.path.join(_LOG_DIR, "rag_qa.log")

    root = logging.getLogger("rag_academic_qa")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    # 格式器
    fmt_console = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler（INFO 级别）
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt_console)
    root.addHandler(ch)

    # 文件 handler（DEBUG 级别，5MB 轮转，保留 3 份）
    fh = RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。

    Args:
        name: 通常传 __name__（如 modules.document_loader）
    Returns:
        logging.Logger 实例，命名空间挂在 rag_academic_qa 下
    """
    _init_root_logger()
    # 统一前缀，方便过滤
    if not name.startswith("rag_academic_qa"):
        name = f"rag_academic_qa.{name}"
    return logging.getLogger(name)
