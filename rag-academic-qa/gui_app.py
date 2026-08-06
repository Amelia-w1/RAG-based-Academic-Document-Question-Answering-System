#!/usr/bin/env python
"""
RAG 学术文献智能问答工具 — GUI 入口
======================================
启动 PySide6 桌面可视化界面。

用法:
  python gui_app.py

首次使用前:
  1. pip install -r requirements.txt  (含 PySide6)
  2. 在界面左侧填入 DashScope API Key
  3. 将 PDF 论文放入 data/papers/
  4. 点击「构建索引」
  5. 输入问题开始问答
"""

import os
import sys
import traceback
import threading
import logging

# ---- 重定向 stdout/stderr 到空设备 ----
# GUI 模式下静默标准输出/错误，避免控制台窗口打印干扰桌面体验。
class _NullStream:
    """静默流：丢弃所有写入，防止控制台窗口闪烁。"""
    def write(self, data):
        pass
    def flush(self):
        pass
    def isatty(self):
        return False

sys.stdout = _NullStream()
sys.stderr = _NullStream()

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QFont

from gui.main_window import MainWindow
from gui.styles import GLOBAL_QSS


# ---- 全局异常钩子 ----
# 捕获未处理的异常，记录到日志文件而非让 Windows 弹出错误对话框

def _global_excepthook(exc_type, exc_value, exc_tb):
    """主线程未捕获异常钩子。"""
    logging.getLogger("rag_academic_qa.gui_app").error(
        "未捕获异常: %s: %s", exc_type.__name__, exc_value, exc_info=True
    )


def _threading_excepthook(args):
    """工作线程未捕获异常钩子。"""
    logging.getLogger("rag_academic_qa.gui_app").error(
        "工作线程未捕获异常: %s: %s", args.exc_type.__name__, args.exc_value,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )


sys.excepthook = _global_excepthook
threading.excepthook = _threading_excepthook


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("RAG 学术文献问答")

    # 设置全局字体
    font = QFont("Microsoft YaHei UI", 10)
    app.setFont(font)

    # 应用全局样式
    app.setStyleSheet(GLOBAL_QSS)

    # 创建主窗口
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
