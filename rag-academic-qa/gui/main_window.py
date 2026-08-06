"""
GUI 主窗口
==================
三栏布局：左侧配置面板 | 中间问答区 | 右侧引用面板
顶部工具栏管理向量库构建，所有耗时操作通过 QThread 执行。

入口: python gui_app.py
"""

import os
import sys
import json
import shutil
import re
import logging

from PySide6.QtCore import Qt, QTimer, QMimeData, Signal
from PySide6.QtGui import QFont, QColor, QIcon, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QProgressBar, QPlainTextEdit,
    QScrollArea, QFrame, QMessageBox, QFileDialog, QListWidget, QListWidgetItem
)

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import Config
from gui.styles import GLOBAL_QSS
from gui.workers import BuildWorker, AskWorker

logger = logging.getLogger(__name__)


class _ChatMessageWidget(QWidget):
    """单条聊天消息气泡。

    完全基于 QLabel + QFrame，不使用任何 QTextEdit/QTextBrowser 控件，
    从而彻底规避 Windows 上 Qt 富文本引擎产生 640×480 临时辅助窗口的问题。
    """
    citationClicked = Signal(int)
    externalLinkClicked = Signal(str)

    _STYLE_MAP = {
        "user": ("#2f81f7", "#dbeafe", "#bfdbfe"),
        "assistant": ("#1a7f37", "#ffffff", "#e5e7eb"),
        "error": ("#cf222e", "#fee2e2", "#fecaca"),
    }

    def __init__(self, sender: str, html: str, is_user: bool = False,
                 is_error: bool = False, parent=None):
        super().__init__(parent)
        kind = "error" if is_error else ("user" if is_user else "assistant")
        title_color, bg_color, border_color = self._STYLE_MAP[kind]

        self.sender_label = QLabel(sender)
        self.sender_label.setStyleSheet(
            f"color: {title_color}; font-weight: bold; font-size: 13px; padding: 2px 8px;"
        )

        self.content_label = QLabel()
        self.content_label.setWordWrap(True)
        self.content_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse
        )
        self.content_label.setOpenExternalLinks(False)
        self.content_label.setStyleSheet("""
            QLabel {
                color: #1f2328;
                font-size: 14px;
                line-height: 140%;
                padding: 6px 10px;
                background: transparent;
            }
        """)
        self.content_label.linkActivated.connect(self._on_link)

        vbox = QVBoxLayout()
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(4)
        vbox.addWidget(self.sender_label)
        vbox.addWidget(self.content_label)

        bubble = QFrame()
        bubble.setLayout(vbox)
        bubble.setStyleSheet(f"""
            QFrame {{
                background: {bg_color};
                border: 1px solid {border_color};
                border-radius: 10px;
            }}
        """)
        # 限制气泡最大宽度，避免过宽；窄窗口时 QLabel 自动换行
        bubble.setMaximumWidth(800)

        hbox = QHBoxLayout(self)
        hbox.setContentsMargins(4, 4, 4, 4)
        hbox.setSpacing(0)
        if is_user:
            hbox.addStretch(1)
            hbox.addWidget(bubble)
        else:
            hbox.addWidget(bubble)
            hbox.addStretch(1)

        self.set_html(html)

    def set_html(self, html: str):
        self.content_label.setText(html)

    def append_html(self, html: str):
        self.content_label.setText(self.content_label.text() + html)

    def _on_link(self, url: str):
        if url.startswith("#frag-"):
            try:
                frag_id = int(url.split("-")[1])
                self.citationClicked.emit(frag_id)
            except (ValueError, IndexError):
                pass
        elif url.startswith(("http://", "https://")):
            self.externalLinkClicked.emit(url)


class _ChatDisplay(QWidget):
    """聊天显示区：QScrollArea 内的 QWidget 列表。

    每条消息是一个 _ChatMessageWidget，完全不用 QTextEdit/QTextBrowser。
    """
    citationClicked = Signal(int)
    externalLinkClicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)

        self._placeholder = QLabel(
            "问答结果将显示在此处 ...\n\n"
            "使用步骤:\n"
            "1. 在左侧填入 DashScope API Key\n"
            "2. 将 PDF 论文放入 data/papers/ 目录\n"
            "3. 点击顶部「构建索引」\n"
            "4. 在下方输入问题，开始问答"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: #8c959f; font-size: 14px; background: transparent; padding: 40px;"
        )
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)
        self._layout.addStretch(1)

    def _message_count(self):
        count = 0
        for i in range(self._layout.count()):
            item = self._layout.itemAt(i)
            if item.widget() and item.widget() is not self._placeholder:
                count += 1
        return count

    def add_message(self, sender: str, html: str, is_user: bool = False,
                    is_error: bool = False):
        # 隐藏占位提示（防御：若已被意外销毁则忽略）
        if self._placeholder is not None:
            try:
                self._placeholder.hide()
            except RuntimeError:
                self._placeholder = None
        # 移除末尾的 stretch，插入新消息后再把 stretch 加回去
        stretch = self._layout.takeAt(self._layout.count() - 1)
        widget = _ChatMessageWidget(sender, html, is_user=is_user,
                                    is_error=is_error)
        widget.citationClicked.connect(self.citationClicked)
        widget.externalLinkClicked.connect(self.externalLinkClicked)
        self._layout.addWidget(widget)
        self._layout.addStretch(1)
        if stretch is not None and stretch.spacerItem():
            del stretch
        return widget

    def last_message_widget(self):
        for i in range(self._layout.count() - 1, -1, -1):
            item = self._layout.itemAt(i)
            w = item.widget()
            if w and w is not self._placeholder:
                return w
        return None

    def clear(self):
        """清空所有消息，但保留占位标签（不删除，避免悬空引用）。"""
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is self._placeholder:
                # 占位标签不删除，仅从布局中移除，稍后重新加入
                continue
            if w:
                w.deleteLater()
            elif item.spacerItem():
                del item
        # 重新加入占位标签（若已被移除）并设为可见
        if self._layout.indexOf(self._placeholder) == -1:
            self._layout.addWidget(self._placeholder)
        self._placeholder.show()
        self._layout.addStretch(1)


class CollapsibleBox(QWidget):
    """可折叠面板：点击标题展开/收缩内容区域。"""

    def __init__(self, title: str, parent=None, expanded: bool = True):
        super().__init__(parent)
        self._expanded = expanded

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._toggle_btn = QPushButton(self._arrow(expanded) + " " + title)
        self._toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 6px 4px;
                border: none;
                background: transparent;
                color: #2f81f7;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #eaeef2;
                border-radius: 4px;
            }
        """)
        self._toggle_btn.setCursor(Qt.PointingHandCursor)
        self._toggle_btn.clicked.connect(self._toggle)
        layout.addWidget(self._toggle_btn)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 4, 8, 8)
        self._content_layout.setSpacing(6)
        self._content.setVisible(expanded)
        layout.addWidget(self._content)

    @staticmethod
    def _arrow(expanded: bool) -> str:
        return "▼" if expanded else "▶"

    def _toggle(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        text = self._toggle_btn.text()[2:]  # 去掉箭头
        self._toggle_btn.setText(self._arrow(self._expanded) + " " + text)

    def content_layout(self) -> QVBoxLayout:
        """返回内容区域的布局，用于添加控件。"""
        return self._content_layout


class CitationCard(QFrame):
    """单条引用卡片，展示文件名、页码、相关性、原文片段。

    点击标题行可展开/折叠正文内容。
    """

    # 全局默认展开状态，影响新创建的卡片
    _global_expanded: bool = True

    def __init__(self, citation: dict, parent=None):
        super().__init__(parent)
        self.citation = citation
        self._expanded = CitationCard._global_expanded
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #d8dee4;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题行（可点击折叠/展开）
        self._header = QFrame()
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setStyleSheet("""
            QFrame {
                background-color: #f6f7f9;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            QFrame:hover { background-color: #eaeef2; }
        """)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(8)

        self._arrow_label = QLabel(self._arrow(self._expanded))
        self._arrow_label.setStyleSheet("color: #2f81f7; font-size: 11px;")
        header_layout.addWidget(self._arrow_label)

        header_text = QLabel(
            f"[片段{citation['fragment_id']}]  {citation['file_name']}"
        )
        header_text.setStyleSheet("font-weight: bold; color: #2f81f7; font-size: 13px;")
        header_layout.addWidget(header_text, 1)

        # 页码 + 相关性（始终显示在标题行右侧）
        meta = QLabel(f"第 {citation['page']} 页")
        meta.setStyleSheet("color: #9a6700; font-size: 11px;")
        header_layout.addWidget(meta)

        rel = QLabel(f"相关度: {citation['relevance_score']:.4f}")
        rel.setStyleSheet("color: #9a6700; font-size: 11px;")
        header_layout.addWidget(rel)

        self._header.mousePressEvent = lambda _event: self._toggle()
        layout.addWidget(self._header)

        # 正文区域（可折叠）
        # 注意：必须指定父对象 self，否则 QWidget() 无父，调用 setVisible(True)
        # 时 Qt 在 Windows 上会先创建一个 640x480 的临时顶层窗口（就是那个"想弹没弹出来的框"）
        self._content_widget = QWidget(self)
        self._content_widget.setVisible(self._expanded)
        content_layout = QVBoxLayout(self._content_widget)
        content_layout.setContentsMargins(12, 8, 12, 10)
        content_layout.setSpacing(4)

        # 原文片段（完整显示，超长时面板自动滚动）
        content_text = citation["content"]
        content = QLabel(content_text)
        content.setWordWrap(True)
        content.setStyleSheet("color: #59636e; font-size: 12px; line-height: 1.4;")
        content.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content.setToolTip(content_text)  # 鼠标悬停显示完整内容
        content_layout.addWidget(content)

        layout.addWidget(self._content_widget)

    @staticmethod
    def _arrow(expanded: bool) -> str:
        return "▼" if expanded else "▶"

    def _toggle(self):
        self._expanded = not self._expanded
        self._content_widget.setVisible(self._expanded)
        self._arrow_label.setText(self._arrow(self._expanded))
        # 同步全局状态，让后续新建卡片保持当前习惯
        CitationCard._global_expanded = self._expanded

    def highlight(self):
        """临时高亮卡片（点击引用链接时调用），3 秒后恢复。"""
        self._header.setStyleSheet("""
            QFrame {
                background-color: #fff8c5;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
        """)
        # 确保卡片可见（展开）
        if not self._expanded:
            self._toggle()
        QTimer.singleShot(3000, self._reset_highlight)

    def _reset_highlight(self):
        """恢复卡片标题行默认样式。"""
        self._header.setStyleSheet("""
            QFrame {
                background-color: #f6f7f9;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            QFrame:hover { background-color: #eaeef2; }
        """)

    @classmethod
    def set_global_expanded(cls, expanded: bool):
        """设置全局默认展开状态。"""
        cls._global_expanded = expanded


class DropInputEdit(QPlainTextEdit):
    """支持拖拽文件导入的输入框。

    将 PDF / Word 文件拖到输入框上松手，会自动复制到论文目录；
    普通文本拖拽仍按默认逻辑粘贴。

    技术细节：QPlainTextEdit 的拖拽事件实际由内部 viewport 接收，
    仅重写 dragEnterEvent 等方法可能无法拦截，因此在 viewport 上
    安装事件过滤器作为主路径，同时保留重写方法作为双重保障。
    """

    SUPPORTED_DROP_EXTS = (".pdf", ".docx", ".doc")

    def __init__(self, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self._main_window = main_window
        self.setAcceptDrops(True)
        # 关键：viewport 才是实际接收鼠标/拖拽事件的 widget
        self.viewport().setAcceptDrops(True)
        self.viewport().installEventFilter(self)

    # ============ viewport 事件过滤器（主拖拽处理路径）============

    def eventFilter(self, obj, event):
        """拦截 viewport 的拖拽事件，确保文件拖拽能被正确捕获。"""
        from PySide6.QtCore import QEvent
        if obj is self.viewport():
            t = event.type()
            if t == QEvent.DragEnter:
                return self._viewport_drag_enter(event)
            elif t == QEvent.DragMove:
                return self._viewport_drag_move(event)
            elif t == QEvent.Drop:
                return self._viewport_drop(event)
        return super().eventFilter(obj, event)

    def _viewport_drag_enter(self, event):
        """viewport 拖拽进入：检查是否包含支持的文件。"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and os.path.splitext(path)[1].lower() in self.SUPPORTED_DROP_EXTS:
                    event.acceptProposedAction()
                    return True
        event.ignore()
        return False

    def _viewport_drag_move(self, event):
        """viewport 拖拽移动：保持接受状态。"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return True
        return False

    def _viewport_drop(self, event):
        """viewport 放下文件 → 导入到论文目录。"""
        if not event.mimeData().hasUrls():
            return False
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        supported = [
            p for p in paths
            if os.path.splitext(p)[1].lower() in self.SUPPORTED_DROP_EXTS
        ]
        if supported:
            event.acceptProposedAction()
            self._main_window._import_dropped_files(supported)
            return True
        return False

    # ============ 重写方法（双重保障）============

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入时检查是否包含支持的文件。"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            for url in urls:
                path = url.toLocalFile()
                if path and os.path.splitext(path)[1].lower() in self.SUPPORTED_DROP_EXTS:
                    event.acceptProposedAction()
                    return
        # 非文件拖拽（纯文本等）交给父类默认处理
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        """拖拽移动时保持接受状态。"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        """放下文件 → 导入到论文目录；普通文本按默认逻辑处理。"""
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return

        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile()]
        supported = [
            p for p in paths
            if os.path.splitext(p)[1].lower() in self.SUPPORTED_DROP_EXTS
        ]
        if supported:
            event.acceptProposedAction()
            self._main_window._import_dropped_files(supported)
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event):
        """Enter / Ctrl+Enter 发送；Shift+Enter 换行。

        QPlainTextEdit 的键盘事件由内部 viewport 接收，重写 keyPressEvent
        是最可靠的处理方式（事件过滤器对 viewport 事件的路由在部分场景下不可靠）。
        """
        from PySide6.QtCore import Qt
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) \
                and not (event.modifiers() & Qt.ShiftModifier):
            self._main_window._on_ask()
            return
        super().keyPressEvent(event)


class MainWindow(QMainWindow):
    """RAG 学术文献问答系统 — 主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG 学术文献智能问答工具")
        self.resize(1280, 800)
        self.setMinimumSize(1000, 650)

        # 工作线程引用
        self._build_worker = None
        self._ask_worker = None

        # 问答历史 HTML（用于持久化，不再用于 setHtml 显示）
        self._chat_html = ""
        # 当前正在流式输出的助手消息控件
        self._current_answer_widget = None

        # 对话历史（用于多轮对话，存储 Q-A 对）
        self._conversation_history: list[tuple[str, str]] = []
        # 多轮对话开关
        self._multi_turn_enabled = True
        # 最近一次提问（用于回答完成后保存对话历史）
        self._last_question: str = ""

        # 会话管理
        self._sessions_dir = os.path.join(_PROJECT_ROOT, "data", "chats")
        os.makedirs(self._sessions_dir, exist_ok=True)
        self._sessions_meta: list[dict] = []  # 会话元数据列表
        self._current_session_id: str | None = None
        self._current_session_name: str = "新会话"

        # 引用卡片索引（fragment_id → CitationCard），用于引用联动
        self._citation_cards: dict[int, CitationCard] = {}

        # 初始化 UI
        self._init_ui()
        self._apply_style()
        self._refresh_index_status()
        self._refresh_pdf_list()
        self._load_sessions()

    # ======================== 文件导入 ========================

    SUPPORTED_DROP_EXTS = (".pdf", ".docx", ".doc")

    def _import_dropped_files(self, paths: list):
        """将拖拽的文件复制到论文目录并刷新列表。"""
        target_dir = Config.PDF_DIR
        os.makedirs(target_dir, exist_ok=True)

        copied = []
        skipped = []
        for path in paths:
            filename = os.path.basename(path)
            dest = os.path.join(target_dir, filename)

            if os.path.abspath(path) == os.path.abspath(dest):
                skipped.append(filename)
                continue

            try:
                shutil.copy2(path, dest)
                copied.append(filename)
            except Exception as e:
                skipped.append(f"{filename} (复制失败: {e})")

        if copied:
            self._refresh_pdf_list()
            self.status_bar.showMessage(f"已导入 {len(copied)} 个文件: {', '.join(copied)}", 8000)
            files_list = chr(10).join(f"  ✓ {f}" for f in copied)
            QMessageBox.information(
                self, "导入成功",
                "已将以下文件复制到论文目录:" + chr(10) + target_dir + chr(10) + chr(10) +
                files_list + chr(10) + chr(10) + "如需生效，请重新「构建索引」"
            )

        if skipped and not copied:
            QMessageBox.information(self, "提示", "文件已存在于论文目录中，无需重复导入。")

    # ======================== 文件选择管理 ========================

    def _on_select_all_files(self):
        """切换全选 / 全不选。"""
        total = self.list_pdf_files.count()
        if total == 0:
            return
        all_checked = all(
            self.list_pdf_files.item(i).checkState() == Qt.Checked
            for i in range(total)
        )
        new_state = Qt.Unchecked if all_checked else Qt.Checked
        for i in range(total):
            self.list_pdf_files.item(i).setCheckState(new_state)
        self._update_pdf_count_label()

    def _on_clear_file_selection(self):
        """取消所有文档勾选。"""
        for i in range(self.list_pdf_files.count()):
            self.list_pdf_files.item(i).setCheckState(Qt.Unchecked)
        self._update_pdf_count_label()

    def _get_selected_file_filter(self) -> list:
        """获取当前勾选的原始文件名列表（用于传给检索模块）。"""
        selected = []
        for i in range(self.list_pdf_files.count()):
            item = self.list_pdf_files.item(i)
            if item.checkState() == Qt.Checked:
                # 优先使用 UserRole 中存储的原始文件名
                name = item.data(Qt.UserRole)
                if not name:
                    # 兼容旧数据：从显示文本中去掉前缀
                    text = item.text()
                    if text.startswith("[PDF] "):
                        name = text[6:]
                    elif text.startswith("[DOC] "):
                        name = text[6:]
                    else:
                        name = text
                selected.append(name)
        return selected

    def _refresh_select_all_button_text(self):
        """根据当前勾选状态刷新「全选」按钮文字。"""
        total = self.list_pdf_files.count()
        if total == 0:
            self.btn_select_all.setText("全选")
            return
        all_checked = all(
            self.list_pdf_files.item(i).checkState() == Qt.Checked
            for i in range(total)
        )
        self.btn_select_all.setText("全不选" if all_checked else "全选")

    def _update_pdf_count_label(self):
        """更新文档数量标签（总数 + 已选数）并刷新按钮文字。"""
        total = self.list_pdf_files.count()
        selected = len(self._get_selected_file_filter())
        self.label_pdf_count.setText(f"共 {total} 篇文档，已选 {selected} 篇")
        self._refresh_select_all_button_text()

    # ======================== UI 初始化 ========================

    def _init_ui(self):
        """构建完整 UI 布局。"""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 顶部工具栏
        main_layout.addWidget(self._build_toolbar())

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(3)
        main_layout.addWidget(self.progress_bar)

        # 三栏分割器
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._build_config_panel())    # 左
        self._splitter.addWidget(self._build_chat_panel())      # 中
        self._citation_panel = self._build_citation_panel()     # 右
        self._splitter.addWidget(self._citation_panel)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([280, 700, 280])
        main_layout.addWidget(self._splitter)

        # 底部状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")
        self.status_bar.setStyleSheet("QStatusBar { background: #e4e7ea; color: #59636e; min-height: 24px; }")
        self.status_bar.setMinimumHeight(24)

    def _build_toolbar(self) -> QWidget:
        """顶部工具栏：构建索引、强制重建、状态指示。"""
        toolbar = QFrame()
        toolbar.setFixedHeight(56)
        toolbar.setStyleSheet("QFrame { background-color: #ffffff; border-bottom: 1px solid #d8dee4; }")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        # 标题
        title = QLabel("RAG 学术文献问答")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2f81f7;")
        layout.addWidget(title)

        layout.addStretch()

        # 多轮对话开关
        from PySide6.QtWidgets import QCheckBox
        self.chk_multi_turn = QCheckBox("多轮对话")
        self.chk_multi_turn.setChecked(True)
        self.chk_multi_turn.setToolTip("启用后，问答时会参考之前的对话历史")
        self.chk_multi_turn.stateChanged.connect(self._on_multi_turn_toggled)
        self.chk_multi_turn.setStyleSheet("font-size: 13px; color: #59636e;")
        layout.addWidget(self.chk_multi_turn)

        # 通用知识兜底开关
        self.chk_fallback = QCheckBox("通用知识兜底")
        self.chk_fallback.setChecked(False)
        self.chk_fallback.setToolTip("检索不到相关文献时，允许模型用通用知识回答（会标注来源类型）")
        self.chk_fallback.setStyleSheet("font-size: 13px; color: #59636e;")
        layout.addWidget(self.chk_fallback)

        # 清空对话按钮
        self.btn_clear_chat = QPushButton("清空对话")
        self.btn_clear_chat.setToolTip("清空对话历史和聊天记录")
        self.btn_clear_chat.clicked.connect(self._on_clear_chat)
        layout.addWidget(self.btn_clear_chat)

        # 保存对话按钮
        self.btn_save_chat = QPushButton("保存对话")
        self.btn_save_chat.setToolTip("将当前对话保存为文件")
        self.btn_save_chat.clicked.connect(self._on_save_chat)
        layout.addWidget(self.btn_save_chat)

        # 打开对话按钮
        self.btn_load_chat = QPushButton("打开对话")
        self.btn_load_chat.setToolTip("从文件加载历史对话")
        self.btn_load_chat.clicked.connect(self._on_load_chat)
        layout.addWidget(self.btn_load_chat)

        # 引用面板切换按钮
        self.btn_toggle_citation = QPushButton("引用面板")
        self.btn_toggle_citation.setCheckable(True)
        self.btn_toggle_citation.setChecked(True)
        self.btn_toggle_citation.setToolTip("显示/隐藏右侧引用来源面板")
        self.btn_toggle_citation.clicked.connect(self._on_toggle_citation)
        layout.addWidget(self.btn_toggle_citation)

        # 索引状态标签
        self.index_status_label = QLabel("检查中 ...")
        self.index_status_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.index_status_label)

        # 构建索引按钮
        self.btn_build = QPushButton("构建索引")
        self.btn_build.setObjectName("primaryBtn")
        self.btn_build.clicked.connect(self._on_build)
        layout.addWidget(self.btn_build)

        # 强制重建按钮
        self.btn_rebuild = QPushButton("强制重建")
        self.btn_rebuild.setObjectName("dangerBtn")
        self.btn_rebuild.clicked.connect(lambda: self._on_build(force=True))
        layout.addWidget(self.btn_rebuild)

        return toolbar

    def _build_config_panel(self) -> QWidget:
        """左侧配置面板。"""
        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setMinimumWidth(280)
        panel.setFrameShape(QFrame.NoFrame)
        panel.setStyleSheet("QScrollArea { border: none; background: #f6f7f9; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ---- 标题 ----
        title = QLabel("配置")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # ---- 会话管理 ----
        session_group = QGroupBox("历史会话")
        session_layout = QVBoxLayout(session_group)
        session_layout.setSpacing(6)

        self.list_sessions = QListWidget()
        self.list_sessions.setFixedHeight(120)
        self.list_sessions.setToolTip("点击切换历史会话")
        self.list_sessions.itemClicked.connect(self._on_session_clicked)
        session_layout.addWidget(self.list_sessions)

        session_btn_layout = QHBoxLayout()
        self.btn_new_session = QPushButton("新建")
        self.btn_new_session.setToolTip("创建一个新会话")
        self.btn_new_session.clicked.connect(self._on_new_session)
        session_btn_layout.addWidget(self.btn_new_session)

        self.btn_rename_session = QPushButton("重命名")
        self.btn_rename_session.setToolTip("修改当前会话名称")
        self.btn_rename_session.clicked.connect(self._on_rename_session)
        session_btn_layout.addWidget(self.btn_rename_session)

        self.btn_delete_session = QPushButton("删除")
        self.btn_delete_session.setToolTip("删除选中的历史会话")
        self.btn_delete_session.clicked.connect(self._on_delete_session)
        session_btn_layout.addWidget(self.btn_delete_session)
        session_layout.addLayout(session_btn_layout)

        layout.addWidget(session_group)

        # ---- API 配置组 ----
        api_group = QGroupBox("DashScope API")
        api_layout = QVBoxLayout(api_group)
        api_layout.setSpacing(6)

        api_layout.addWidget(QLabel("API Key:"))
        self.input_apikey = QLineEdit(Config.DASHSCOPE_API_KEY)
        self.input_apikey.setEchoMode(QLineEdit.Password)
        self.input_apikey.setPlaceholderText("sk-xxxxxxxx")
        api_layout.addWidget(self.input_apikey)

        api_layout.addWidget(QLabel("对话模型:"))
        self.combo_llm = QComboBox()
        self.combo_llm.addItems(["qwen-turbo", "qwen-plus", "qwen-max"])
        self.combo_llm.setCurrentText(Config.LLM_MODEL)
        api_layout.addWidget(self.combo_llm)

        api_layout.addWidget(QLabel("向量模型:"))
        self.combo_embed = QComboBox()
        self.combo_embed.addItems(["text-embedding-v2", "text-embedding-v1"])
        self.combo_embed.setCurrentText(Config.EMBEDDING_MODEL)
        api_layout.addWidget(self.combo_embed)

        api_layout.addWidget(QLabel("重排序模型:"))
        self.combo_rerank = QComboBox()
        self.combo_rerank.addItems(["gte-rerank"])
        self.combo_rerank.setCurrentText(Config.RERANK_MODEL)
        api_layout.addWidget(self.combo_rerank)

        layout.addWidget(api_group)

        # ---- 文档处理组（可折叠）----
        doc_box = CollapsibleBox("文档处理", expanded=True)
        doc_layout = doc_box.content_layout()

        doc_layout.addWidget(QLabel("切片大小 (字符):"))
        self.spin_chunk_size = QSpinBox()
        self.spin_chunk_size.setRange(100, 5000)
        self.spin_chunk_size.setSingleStep(50)
        self.spin_chunk_size.setValue(Config.CHUNK_SIZE)
        doc_layout.addWidget(self.spin_chunk_size)

        doc_layout.addWidget(QLabel("重叠长度 (字符):"))
        self.spin_chunk_overlap = QSpinBox()
        self.spin_chunk_overlap.setRange(0, 1000)
        self.spin_chunk_overlap.setSingleStep(10)
        self.spin_chunk_overlap.setValue(Config.CHUNK_OVERLAP)
        doc_layout.addWidget(self.spin_chunk_overlap)

        layout.addWidget(doc_box)

        # ---- 检索参数组（可折叠）----
        ret_box = CollapsibleBox("检索参数", expanded=True)
        ret_layout = ret_box.content_layout()

        ret_layout.addWidget(QLabel("检索候选数 K:"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 50)
        self.spin_k.setValue(Config.RETRIEVAL_K)
        ret_layout.addWidget(self.spin_k)

        ret_layout.addWidget(QLabel("重排序 Top-N:"))
        self.spin_topn = QSpinBox()
        self.spin_topn.setRange(1, 50)
        self.spin_topn.setValue(Config.RERANK_TOP_N)
        ret_layout.addWidget(self.spin_topn)

        ret_layout.addWidget(QLabel("模型温度:"))
        self.spin_temp = QDoubleSpinBox()
        self.spin_temp.setRange(0.0, 2.0)
        self.spin_temp.setSingleStep(0.05)
        self.spin_temp.setDecimals(2)
        self.spin_temp.setValue(Config.TEMPERATURE)
        ret_layout.addWidget(self.spin_temp)

        ret_layout.addWidget(QLabel("检索策略:"))
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItems(["vector", "hyde", "multi_query", "hybrid"])
        self.combo_strategy.setCurrentText("vector")
        self.combo_strategy.setToolTip(
            "vector: 标准向量检索+重排序\n"
            "hyde: HyDE 假设性文档嵌入检索\n"
            "multi_query: 多查询检索(LLM改写+RRF融合)\n"
            "hybrid: 混合检索(向量+BM25+RRF融合)"
        )
        ret_layout.addWidget(self.combo_strategy)

        layout.addWidget(ret_box)

        # ---- 保存按钮 ----
        self.btn_save_config = QPushButton("保存配置")
        self.btn_save_config.setObjectName("primaryBtn")
        self.btn_save_config.clicked.connect(self._on_save_config)
        layout.addWidget(self.btn_save_config)

        # ---- PDF 目录选择 ----
        pdf_group = QGroupBox("论文目录")
        pdf_layout = QVBoxLayout(pdf_group)
        pdf_layout.setSpacing(6)

        self.label_pdf_dir = QLabel(Config.PDF_DIR)
        self.label_pdf_dir.setWordWrap(True)
        self.label_pdf_dir.setStyleSheet("color: #59636e; font-size: 12px;")
        pdf_layout.addWidget(self.label_pdf_dir)

        self.btn_browse_pdf = QPushButton("选择目录 ...")
        self.btn_browse_pdf.clicked.connect(self._on_browse_pdf)
        pdf_layout.addWidget(self.btn_browse_pdf)

        pdf_layout.addWidget(QLabel("已发现文档（勾选参与检索）:"))
        self.list_pdf_files = QListWidget()
        self.list_pdf_files.setFixedHeight(160)
        self.list_pdf_files.setToolTip("勾选后，问答时只从选中的文档里检索；不勾选任何文档则无法问答")
        pdf_layout.addWidget(self.list_pdf_files)

        # 全选 / 清空 按钮
        file_select_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.setToolTip("勾选所有文档")
        self.btn_select_all.clicked.connect(self._on_select_all_files)
        file_select_layout.addWidget(self.btn_select_all)

        self.btn_clear_selection = QPushButton("清空")
        self.btn_clear_selection.setToolTip("取消所有勾选")
        self.btn_clear_selection.clicked.connect(self._on_clear_file_selection)
        file_select_layout.addWidget(self.btn_clear_selection)
        pdf_layout.addLayout(file_select_layout)

        self.label_pdf_count = QLabel("共 0 篇文档")
        self.label_pdf_count.setStyleSheet("color: #8b949e; font-size: 11px;")
        pdf_layout.addWidget(self.label_pdf_count)

        # 拖拽提示
        drop_hint = QLabel("💡 提示: 可直接拖拽 PDF / Word 文件到下方输入框导入")
        drop_hint.setStyleSheet("color: #8b949e; font-size: 11px; padding: 4px 0;")
        drop_hint.setWordWrap(True)
        pdf_layout.addWidget(drop_hint)

        layout.addWidget(pdf_group)

        layout.addStretch()
        panel.setWidget(container)
        return panel

    def _build_chat_panel(self) -> QWidget:
        """中间问答面板。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # 问答显示区（基于 QLabel 列表，彻底规避 QTextBrowser/QTextEdit 的 640x480 弹窗）
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)
        self.chat_scroll.setStyleSheet("QScrollArea { background: #f6f7f9; border: none; }")
        self.chat_display = _ChatDisplay()
        self.chat_display.setFont(QFont("Microsoft YaHei UI", 11))
        # 引用联动：点击 [片段N] 高亮对应卡片；外部链接用浏览器打开
        self.chat_display.citationClicked.connect(self._on_citation_clicked)
        self.chat_display.externalLinkClicked.connect(self._on_external_link_clicked)
        self.chat_scroll.setWidget(self.chat_display)
        layout.addWidget(self.chat_scroll, 1)

        # 输入区域
        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background: transparent; }")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        self.input_question = DropInputEdit(self)
        self.input_question.setPlaceholderText("输入你的问题，按 Enter 发送；Shift+Enter 换行；或拖拽 PDF / Word 文件到此处导入 ...")
        self.input_question.setFixedHeight(60)
        self.input_question.installEventFilter(self)
        input_layout.addWidget(self.input_question, 1)

        self.btn_send = QPushButton("发送")
        self.btn_send.setObjectName("primaryBtn")
        self.btn_send.setFixedSize(80, 60)
        self.btn_send.clicked.connect(self._on_ask)
        input_layout.addWidget(self.btn_send)

        layout.addWidget(input_frame)
        return panel

    def _build_citation_panel(self) -> QWidget:
        """右侧引用面板。"""
        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setMinimumWidth(280)
        panel.setFrameShape(QFrame.NoFrame)
        panel.setStyleSheet("QScrollArea { border: none; background: #f6f7f9; }")

        self._citation_container = QWidget()
        self._citation_layout = QVBoxLayout(self._citation_container)
        self._citation_layout.setContentsMargins(12, 12, 12, 12)
        self._citation_layout.setSpacing(8)

        # 标题行 + 全部折叠/展开按钮（固定在 index 0，不被 _clear_citations 清理）
        self._citation_header = QWidget()
        header_layout = QHBoxLayout(self._citation_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        title = QLabel("引用来源")
        title.setObjectName("sectionTitle")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #2f81f7; padding: 4px 0;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.btn_toggle_all_citations = QPushButton("全部折叠")
        self.btn_toggle_all_citations.setToolTip("折叠/展开所有引用片段")
        self.btn_toggle_all_citations.setStyleSheet("""
            QPushButton {
                font-size: 11px;
                padding: 3px 8px;
                border: 1px solid #d8dee4;
                border-radius: 4px;
                background: #ffffff;
                color: #59636e;
            }
            QPushButton:hover { background: #f6f7f9; }
        """)
        self.btn_toggle_all_citations.clicked.connect(self._on_toggle_all_citations)
        header_layout.addWidget(self.btn_toggle_all_citations)
        self._citation_layout.addWidget(self._citation_header)

        self._citation_placeholder = QLabel("回答中的引用文献片段将显示在此处")
        self._citation_placeholder.setWordWrap(True)
        self._citation_placeholder.setStyleSheet("color: #8b949e; font-size: 13px; padding: 20px;")
        self._citation_layout.addWidget(self._citation_placeholder)

        self._citation_layout.addStretch()
        panel.setWidget(self._citation_container)
        return panel

    # ======================== 样式 ========================

    def _apply_style(self):
        self.setStyleSheet(GLOBAL_QSS)

    # ======================== 事件处理 ========================

    def eventFilter(self, obj, event):
        """Enter / Ctrl+Enter 发送问题；Shift+Enter 保留换行。

        注意：QPlainTextEdit 的键盘事件实际发给其内部 viewport 子控件，
        因此需同时匹配 input_question 及其 viewport，否则 Enter 失效。
        """
        from PySide6.QtCore import QEvent
        if obj in (self.input_question, self.input_question.viewport()) \
                and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if event.modifiers() & Qt.ShiftModifier:
                    # Shift+Enter 插入换行，不拦截
                    return False
                # Enter / Ctrl+Enter 发送
                self._on_ask()
                return True
        return super().eventFilter(obj, event)

    # ======================== 索引管理 ========================

    def _refresh_index_status(self):
        """刷新索引状态显示。"""
        try:
            from modules import VectorStore
        except ImportError:
            self.index_status_label.setText("依赖未安装")
            self.index_status_label.setStyleSheet("color: #cf222e; font-size: 13px; font-weight: bold;")
            return
        if VectorStore.has_index():
            self.index_status_label.setText("向量库: 已就绪")
            self.index_status_label.setObjectName("statusOk")
            self.index_status_label.setStyleSheet("color: #1a7f37; font-size: 13px; font-weight: bold;")
        else:
            self.index_status_label.setText("向量库: 未构建")
            self.index_status_label.setObjectName("statusWarn")
            self.index_status_label.setStyleSheet("color: #9a6700; font-size: 13px; font-weight: bold;")

    def _on_build(self, force=False):
        """触发索引构建。"""
        try:
            from modules import VectorStore
        except ImportError:
            QMessageBox.critical(self, "依赖缺失",
                "核心依赖未安装，请在终端运行:\npip install -r requirements.txt")
            return

        # 先保存配置（应用最新参数）
        self._apply_config_from_ui()

        ok, msg = Config.validate()
        if not ok:
            QMessageBox.warning(self, "配置错误", msg)
            return

        if not force and VectorStore.has_index():
            reply = QMessageBox.question(
                self, "确认",
                "向量库已存在，是否重新构建？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # 禁用按钮，显示进度
        self.btn_build.setEnabled(False)
        self.btn_rebuild.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("正在构建向量库 ...")

        # 启动工作线程
        self._build_worker = BuildWorker(
            chunk_size=self.spin_chunk_size.value(),
            chunk_overlap=self.spin_chunk_overlap.value(),
            force=force,
        )
        self._build_worker.progress.connect(self._on_build_progress)
        self._build_worker.finished_ok.connect(self._on_build_ok)
        self._build_worker.finished_err.connect(self._on_build_err)
        self._build_worker.start()

    def _on_build_progress(self, msg):
        """构建进度回调。"""
        self.status_bar.showMessage(msg.strip())

    def _on_build_ok(self, msg):
        """构建成功。"""
        self.progress_bar.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_rebuild.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.status_bar.showMessage("向量库构建完成", 5000)
        self._refresh_index_status()
        QMessageBox.information(self, "构建完成", msg)

    def _on_build_err(self, msg):
        """构建失败。"""
        self.progress_bar.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_rebuild.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.status_bar.showMessage("构建失败", 5000)
        QMessageBox.critical(self, "构建失败", msg)

    # ======================== 问答 ========================

    def _on_multi_turn_toggled(self, state):
        """多轮对话开关切换。"""
        self._multi_turn_enabled = bool(state)
        if self._multi_turn_enabled:
            self.status_bar.showMessage("多轮对话已启用", 3000)
        else:
            self.status_bar.showMessage("多轮对话已关闭", 3000)

    def _on_clear_chat(self):
        """清空当前会话的对话内容（不删除会话本身）。"""
        self._conversation_history.clear()
        self._chat_html = ""
        self.chat_display.clear()
        self._clear_citations()
        self._save_current_session()
        self._refresh_session_list()
        self.status_bar.showMessage("当前会话已清空", 3000)

    def _on_save_chat(self):
        """将当前会话导出为 JSON 文件（兼容导入格式）。"""
        if not self._conversation_history and not self._chat_html:
            QMessageBox.information(self, "提示", "当前没有对话内容可保存。")
            return

        from datetime import datetime
        default_name = f"{self._current_session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        default_path = os.path.join(_PROJECT_ROOT, "data", "chats", default_name)

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出对话", default_path, "JSON 文件 (*.json)"
        )
        if not file_path:
            return

        import json as _json
        data = {
            "version": 2,
            "session_id": self._current_session_id or "",
            "name": self._current_session_name,
            "timestamp": datetime.now().isoformat(),
            "history": [[q, a] for q, a in self._conversation_history],
            "chat_html": self._chat_html,
        }
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                _json.dump(data, f, ensure_ascii=False, indent=2)
            self.status_bar.showMessage(f"对话已导出至 {file_path}", 5000)
            QMessageBox.information(self, "导出成功", f"对话已导出至:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _on_load_chat(self):
        """从文件导入历史对话，作为新会话打开。"""
        default_dir = os.path.join(_PROJECT_ROOT, "data", "chats")
        if not os.path.isdir(default_dir):
            default_dir = _PROJECT_ROOT

        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开对话", default_dir, "JSON 文件 (*.json)"
        )
        if not file_path:
            return

        import json as _json
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"无法读取文件: {e}")
            return

        # 先保存当前会话
        self._save_current_session()

        # 创建新会话并导入数据
        from datetime import datetime
        sid = self._create_new_session(data.get("name", "导入的会话"))
        self._current_session_id = sid
        self._current_session_name = data.get("name", "导入的会话")
        history = data.get("history", [])
        self._conversation_history = [(q, a) for q, a in history]

        # 恢复聊天显示（从历史重建，避免 setHtml 闪烁）
        self._rebuild_chat_from_history()

        # 持久化新会话并刷新列表
        self._save_current_session()
        self._refresh_session_list()

        # 清空引用面板（加载的对话不恢复引用）
        self._clear_citations()

        ts = data.get("timestamp", "未知时间")
        self.status_bar.showMessage(f"已导入对话为「{self._current_session_name}」，共 {len(self._conversation_history)} 轮", 5000)

    # ======================== 会话管理 ========================

    def _session_file_path(self, session_id: str) -> str:
        """返回会话数据的持久化文件路径。"""
        return os.path.join(self._sessions_dir, f"{session_id}.json")

    def _load_sessions(self):
        """加载所有历史会话元数据，并确保至少有一个当前会话。"""
        self._sessions_meta = []
        if os.path.isdir(self._sessions_dir):
            for fname in sorted(os.listdir(self._sessions_dir)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(self._sessions_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # 跳过已标记删除的会话
                    if data.get("deleted"):
                        continue
                    sid = data.get("session_id") or fname[:-5]
                    meta = {
                        "session_id": sid,
                        "name": data.get("name", "未命名会话"),
                        "timestamp": data.get("timestamp", ""),
                        "turn_count": len(data.get("history", [])),
                    }
                    self._sessions_meta.append(meta)
                except Exception:
                    continue

        if not self._sessions_meta:
            # 没有任何会话时自动创建一个
            self._create_new_session("新会话")
        else:
            # 默认加载最新的会话
            latest = max(self._sessions_meta, key=lambda m: m["timestamp"] or "")
            self._switch_to_session(latest["session_id"])

        self._refresh_session_list()

    def _create_new_session(self, name: str) -> str:
        """创建新会话并返回 session_id。"""
        from datetime import datetime
        sid = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._sessions_meta.append({
            "session_id": sid,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "turn_count": 0,
        })
        # 持久化空会话
        self._save_session_data(sid, name, [], "")
        return sid

    def _save_session_data(self, session_id: str, name: str,
                           history: list, chat_html: str):
        """将会话数据写入磁盘。"""
        from datetime import datetime
        data = {
            "version": 2,
            "session_id": session_id,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "history": [[q, a] for q, a in history],
            "chat_html": chat_html,
        }
        try:
            with open(self._session_file_path(session_id), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("保存会话失败: %s", e)

    def _save_current_session(self):
        """保存当前会话到磁盘，并更新元数据。"""
        if not self._current_session_id:
            return
        self._save_session_data(
            self._current_session_id,
            self._current_session_name,
            self._conversation_history,
            self._chat_html,
        )
        # 更新元数据
        for meta in self._sessions_meta:
            if meta["session_id"] == self._current_session_id:
                meta["turn_count"] = len(self._conversation_history)
                break

    def _auto_name_session(self, question: str) -> str:
        """根据首个问题自动生成会话名称。

        规则:
        - 去除首尾空白，过滤换行
        - 中文/混合内容取前 12 个字符，超出加省略号
        - 纯 ASCII 内容取前 20 个字符，超出加省略号
        - 空内容返回默认名称
        """
        question = question.strip().replace("\n", " ").replace("\r", "")
        if not question:
            return "新会话"

        if question.isascii():
            max_len = 20
        else:
            max_len = 12

        if len(question) <= max_len:
            return question
        return question[:max_len] + "…"

    def _switch_to_session(self, session_id: str):
        """切换到指定会话（先保存当前，再加载目标）。"""
        # 保存当前（如果存在且不是已删除的）
        if self._current_session_id:
            self._save_current_session()

        # 加载目标
        fpath = self._session_file_path(session_id)
        if not os.path.exists(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        self._current_session_id = session_id
        self._current_session_name = data.get("name", "未命名会话")
        history = data.get("history", [])
        self._conversation_history = [(q, a) for q, a in history]

        # 恢复聊天显示（从历史重建，避免 setHtml 闪烁）
        self._rebuild_chat_from_history()

        self._clear_citations()

    def _refresh_session_list(self):
        """刷新左侧会话列表显示。"""
        self.list_sessions.clear()
        for meta in self._sessions_meta:
            item = QListWidgetItem(meta["name"])
            item.setData(Qt.UserRole, meta["session_id"])
            item.setToolTip(f"创建于 {meta.get('timestamp', '')}\n共 {meta['turn_count']} 轮对话")
            self.list_sessions.addItem(item)
            if meta["session_id"] == self._current_session_id:
                item.setSelected(True)

    def _on_session_clicked(self, item: QListWidgetItem):
        """用户点击会话列表项时切换会话。"""
        sid = item.data(Qt.UserRole)
        if sid == self._current_session_id:
            return
        self._switch_to_session(sid)
        self._refresh_session_list()
        self.status_bar.showMessage(f"已切换到会话: {item.text()}", 3000)

    def _on_new_session(self):
        """新建会话。"""
        # 先保存当前
        self._save_current_session()
        sid = self._create_new_session("新会话")
        self._current_session_id = sid
        self._current_session_name = "新会话"
        self._conversation_history = []
        self._chat_html = ""
        self.chat_display.clear()
        self._clear_citations()
        self._refresh_session_list()
        self.status_bar.showMessage("已创建新会话", 3000)

    def _on_rename_session(self):
        """重命名当前会话。"""
        if not self._current_session_id:
            return
        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "重命名会话", "请输入新名称:",
            text=self._current_session_name
        )
        if not ok or not new_name.strip():
            return
        self._current_session_name = new_name.strip()
        self._save_current_session()
        for meta in self._sessions_meta:
            if meta["session_id"] == self._current_session_id:
                meta["name"] = self._current_session_name
                break
        self._refresh_session_list()
        self.status_bar.showMessage(f"已重命名为: {self._current_session_name}", 3000)

    def _on_delete_session(self):
        """删除选中的会话。"""
        item = self.list_sessions.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先选中要删除的会话")
            return
        sid = item.data(Qt.UserRole)
        name = item.text()
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除会话「{name}」吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # 标记删除（沙箱环境可能禁止真实删除文件，改为写入 deleted 标记）
        fpath = self._session_file_path(sid)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["deleted"] = True
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                QMessageBox.critical(self, "删除失败", str(e))
                return

        # 更新元数据
        self._sessions_meta = [m for m in self._sessions_meta if m["session_id"] != sid]

        if sid == self._current_session_id:
            # 删除的是当前会话：先清除 current_session_id 防止 _switch_to_session 保存已删除的会话
            self._current_session_id = None
            if self._sessions_meta:
                latest = max(self._sessions_meta, key=lambda m: m["timestamp"] or "")
                # 直接加载目标会话，不保存当前（已删除）
                new_fpath = self._session_file_path(latest["session_id"])
                if os.path.exists(new_fpath):
                    try:
                        with open(new_fpath, "r", encoding="utf-8") as f:
                            new_data = json.load(f)
                        self._current_session_id = latest["session_id"]
                        self._current_session_name = new_data.get("name", "未命名会话")
                        history = new_data.get("history", [])
                        self._conversation_history = [(q, a) for q, a in history]
                        self._rebuild_chat_from_history()
                        self._clear_citations()
                    except Exception:
                        pass
            else:
                sid_new = self._create_new_session("新会话")
                self._current_session_id = sid_new
                self._current_session_name = "新会话"
                self._conversation_history = []
                self._chat_html = ""
                self.chat_display.clear()
                self._clear_citations()

        self._refresh_session_list()
        self.status_bar.showMessage(f"已删除会话: {name}", 3000)

    def _on_toggle_citation(self):
        """显示/隐藏右侧引用面板。"""
        visible = self.btn_toggle_citation.isChecked()
        self._citation_panel.setVisible(visible)
        if visible:
            self.status_bar.showMessage("引用面板已显示", 2000)
        else:
            self.status_bar.showMessage("引用面板已隐藏", 2000)

    def _on_toggle_all_citations(self):
        """切换所有引用卡片的展开/折叠状态。"""
        # 根据当前按钮文字判断下一步动作
        collapse = self.btn_toggle_all_citations.text() == "全部折叠"
        CitationCard.set_global_expanded(not collapse)

        # 遍历所有 CitationCard 并同步状态
        changed = 0
        for i in range(self._citation_layout.count()):
            item = self._citation_layout.itemAt(i)
            if item is None:
                continue
            widget = item.widget()
            if isinstance(widget, CitationCard):
                if collapse:
                    if widget._expanded:
                        widget._toggle()
                        changed += 1
                else:
                    if not widget._expanded:
                        widget._toggle()
                        changed += 1

        if collapse:
            self.btn_toggle_all_citations.setText("全部展开")
            self.status_bar.showMessage(f"已折叠 {changed} 个引用片段", 2000)
        else:
            self.btn_toggle_all_citations.setText("全部折叠")
            self.status_bar.showMessage(f"已展开 {changed} 个引用片段", 2000)

    def _on_ask(self):
        """触发问答。"""
        try:
            from modules import VectorStore
        except ImportError:
            QMessageBox.critical(self, "依赖缺失",
                "核心依赖未安装，请在终端运行:\npip install -r requirements.txt")
            return

        question = self.input_question.toPlainText().strip()
        if not question:
            return

        if self._ask_worker and self._ask_worker.isRunning():
            return

        # 检查向量库
        if not VectorStore.has_index():
            QMessageBox.warning(self, "提示", "向量库未构建，请先点击「构建索引」。")
            return

        # 获取选中的文件过滤条件
        file_filter = self._get_selected_file_filter()
        if not file_filter:
            QMessageBox.information(self, "提示", "请先在左侧文件列表中至少勾选一篇文献。")
            return

        # 清空引用面板
        self._clear_citations()

        # 在聊天区追加问题
        self._append_question(question)
        self._last_question = question  # 保存用于对话历史
        self.input_question.clear()

        # 禁用发送按钮
        self.btn_send.setEnabled(False)
        self.btn_build.setEnabled(False)
        self.btn_rebuild.setEnabled(False)
        self.progress_bar.setVisible(True)

        scope_text = f"仅检索 {len(file_filter)} 篇文献"
        self.status_bar.showMessage(f"正在检索并生成回答 ({scope_text}) ...")

        # 追加回答占位
        self._answering = True
        self._current_answer = ""

        # 构建历史文本（如果启用多轮对话且有历史）
        history_text = None
        if self._multi_turn_enabled and self._conversation_history:
            lines = []
            for q, a in self._conversation_history[-5:]:  # 保留最近 5 轮
                lines.append(f"[用户] {q}")
                lines.append(f"[助手] {a[:500]}")  # 截断过长的历史回答
            history_text = "\n".join(lines)

        # 启动问答线程
        self._ask_worker = AskWorker(
            question,
            k=self.spin_k.value(),
            top_n=self.spin_topn.value(),
            file_filter=file_filter,
            history_text=history_text,
            use_history=self._multi_turn_enabled,
            strategy_name=self.combo_strategy.currentText(),
            fallback_enabled=self.chk_fallback.isChecked(),
        )
        self._ask_worker.retrieved.connect(self._on_retrieved)
        self._ask_worker.chunk.connect(self._on_chunk)
        self._ask_worker.finished_ok.connect(self._on_ask_ok)
        self._ask_worker.finished_err.connect(self._on_ask_err)
        self._ask_worker.start()

    def _on_retrieved(self, citations):
        """检索完成，展示引用。"""
        try:
            self._display_citations(citations)
        except Exception as e:
            logger.error("_on_retrieved 出错: %s", e, exc_info=True)

    def _on_chunk(self, text):
        """流式回答片段。"""
        try:
            self._current_answer += text
            self._on_chunk_insert(text)
        except Exception as e:
            logger.error("_on_chunk 出错: %s", e, exc_info=True)

    def _on_ask_ok(self, full_answer):
        """回答完成，保存对话历史并自动持久化当前会话。"""
        self._answering = False
        self.progress_bar.setVisible(False)
        self.btn_send.setEnabled(True)
        self.btn_build.setEnabled(True)
        self.btn_rebuild.setEnabled(True)

        try:
            # 将流式纯文本替换为带引用链接的 HTML
            self._finalize_answer()
            # 保存 Q-A 对到对话历史
            if full_answer and full_answer != "未检索到相关文献片段，无法回答。":
                self._conversation_history.append((self._last_question, full_answer))

                # 首次问答后自动命名会话
                if self._current_session_name == "新会话":
                    new_name = self._auto_name_session(self._last_question)
                    if new_name != "新会话":
                        self._current_session_name = new_name
                        for meta in self._sessions_meta:
                            if meta["session_id"] == self._current_session_id:
                                meta["name"] = new_name
                                break
                        self.status_bar.showMessage(f"会话已自动命名为: {new_name}", 5000)

                # 自动持久化当前会话
                self._save_current_session()
                self._refresh_session_list()

            self.status_bar.showMessage("回答完成", 5000)
        except Exception as e:
            logger.error("_on_ask_ok 出错: %s", e, exc_info=True)
            self.status_bar.showMessage(f"保存对话出错: {e}", 5000)

    def _on_ask_err(self, msg):
        """回答失败。"""
        self._answering = False
        self.progress_bar.setVisible(False)
        self.btn_send.setEnabled(True)
        self.btn_build.setEnabled(True)
        self.btn_rebuild.setEnabled(True)
        # 清理未完成的回答控件
        if self._current_answer_widget is not None:
            self._finalize_answer()
        self.status_bar.showMessage("回答失败", 5000)
        self._append_error(msg)

    # ======================== 聊天显示 ========================

    def _html_text(self, text: str) -> str:
        """将纯文本转为 QLabel 可用的 HTML（转义 + 换行）。"""
        return _escape_html(text).replace("\n", "<br>")

    def _append_question(self, question):
        """追加用户问题并创建助手回答占位气泡。"""
        user_html = self._html_text(question)
        self.chat_display.add_message("你", user_html, is_user=True)
        self._current_answer_widget = self.chat_display.add_message(
            "RAG 助手", "", is_user=False
        )
        self._chat_html += (
            f"<div><b style='color:#2f81f7'>你</b><br>{_escape_html(question)}"
            f"<br><br><b style='color:#1a7f37'>RAG 助手</b><br>"
        )
        self._scroll_chat_to_bottom()

    def _on_chunk_insert(self, text):
        """流式追加回答文本片段。"""
        if self._current_answer_widget is None:
            return
        self._current_answer_widget.append_html(self._html_text(text))
        self._scroll_chat_to_bottom()

    def _finalize_answer(self):
        """流式回答完成后，渲染带引用链接的最终文本。"""
        if self._current_answer_widget is None:
            return
        rendered = _make_citation_links(
            _escape_html(self._current_answer).replace("\n", "<br>")
        )
        self._current_answer_widget.set_html(rendered)
        self._chat_html += rendered + "</div>"
        self._current_answer_widget = None
        self._scroll_chat_to_bottom()

    def _restore_answer(self, answer_text):
        """恢复历史会话时直接插入完整回答。"""
        if self._current_answer_widget is None:
            return
        rendered = _make_citation_links(
            _escape_html(answer_text).replace("\n", "<br>")
        )
        self._current_answer_widget.set_html(rendered)
        self._chat_html += rendered + "</div>"
        self._current_answer_widget = None
        self._scroll_chat_to_bottom()

    def _append_error(self, msg):
        """追加错误信息到聊天区。"""
        self.chat_display.add_message(
            "错误", self._html_text(msg), is_user=False, is_error=True
        )
        self._chat_html += f"<div><b style='color:#cf222e'>错误</b><br>{_escape_html(msg)}</div>"
        self._scroll_chat_to_bottom()

    def _rebuild_chat_from_history(self):
        """从对话历史重建聊天显示。"""
        self.chat_display.clear()
        self._chat_html = ""
        for q, a in self._conversation_history:
            self._append_question(q)
            self._current_answer = a
            self._restore_answer(a)

    # ======================== 引用显示 ========================

    def _clear_citations(self):
        """清空引用面板的动态内容（保留 index 0 的标题行）。"""
        # 清空卡片索引
        self._citation_cards = {}
        # 跳过 index 0（self._citation_header），只清理后面的动态内容
        while self._citation_layout.count() > 1:
            item = self._citation_layout.takeAt(1)  # 始终取第二个（index 1）
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                while item.layout().count() > 0:
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()

    def _display_citations(self, citations):
        """展示引用卡片列表。"""
        if not citations:
            placeholder = QLabel("本次回答无引用文献")
            placeholder.setStyleSheet("color: #8b949e; font-size: 13px; padding: 20px;")
            self._citation_layout.addWidget(placeholder)
            self._citation_layout.addStretch()
            return

        for cite in citations:
            card = CitationCard(cite)
            self._citation_cards[cite["fragment_id"]] = card
            self._citation_layout.addWidget(card)

        self._citation_layout.addStretch()

    def _on_citation_clicked(self, frag_id: int):
        """点击回答中的 [片段N] 文本 → 高亮并滚动到对应引用卡片。"""
        try:
            card = self._citation_cards.get(frag_id)
            if card is not None:
                # 滚动引用面板到该卡片
                self._citation_panel.ensureWidgetVisible(card)
                # 高亮卡片
                card.highlight()
        except Exception as e:
            logger.error("_on_citation_clicked 出错: %s", e, exc_info=True)

    def _on_external_link_clicked(self, url: str):
        """点击外部 http/https 链接 → 用系统默认浏览器打开。"""
        try:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(url))
        except Exception as e:
            logger.error("_on_external_link_clicked 出错: %s", e, exc_info=True)

    def _scroll_chat_to_bottom(self):
        """滚动聊天区到底部（延迟一帧，等待布局完成）。"""
        def _do_scroll():
            vsb = self.chat_scroll.verticalScrollBar()
            vsb.setValue(vsb.maximum())
        QTimer.singleShot(0, _do_scroll)

    # ======================== 配置管理 ========================

    def _apply_config_from_ui(self):
        """从 UI 控件读取值，更新到 Config 类属性。"""
        Config.DASHSCOPE_API_KEY = self.input_apikey.text().strip()
        Config.LLM_MODEL = self.combo_llm.currentText()
        Config.EMBEDDING_MODEL = self.combo_embed.currentText()
        Config.RERANK_MODEL = self.combo_rerank.currentText()
        Config.CHUNK_SIZE = self.spin_chunk_size.value()
        Config.CHUNK_OVERLAP = self.spin_chunk_overlap.value()
        Config.RETRIEVAL_K = self.spin_k.value()
        Config.RERANK_TOP_N = self.spin_topn.value()
        Config.TEMPERATURE = self.spin_temp.value()

    def _on_save_config(self):
        """保存配置到 .env 文件。"""
        self._apply_config_from_ui()

        env_path = os.path.join(_PROJECT_ROOT, ".env")
        lines = [
            f"# RAG 学术文献问答系统配置",
            f"# 由 GUI 自动生成 — {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"DASHSCOPE_API_KEY={Config.DASHSCOPE_API_KEY}",
            f"LLM_MODEL={Config.LLM_MODEL}",
            f"EMBEDDING_MODEL={Config.EMBEDDING_MODEL}",
            f"RERANK_MODEL={Config.RERANK_MODEL}",
            f"CHUNK_SIZE={Config.CHUNK_SIZE}",
            f"CHUNK_OVERLAP={Config.CHUNK_OVERLAP}",
            f"RETRIEVAL_K={Config.RETRIEVAL_K}",
            f"RERANK_TOP_N={Config.RERANK_TOP_N}",
            f"TEMPERATURE={Config.TEMPERATURE}",
            f"PDF_DIR={Config.PDF_DIR}",
            f"FAISS_INDEX_DIR={Config.FAISS_INDEX_DIR}",
        ]

        try:
            with open(env_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self.status_bar.showMessage(f"配置已保存至 {env_path}", 5000)
            QMessageBox.information(self, "保存成功", f"配置已保存至:\n{env_path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _on_browse_pdf(self):
        """选择 PDF 目录。"""
        current_dir = Config.PDF_DIR
        chosen = QFileDialog.getExistingDirectory(
            self, "选择 PDF 论文目录", current_dir
        )
        if chosen:
            Config.PDF_DIR = chosen
            self.label_pdf_dir.setText(chosen)
            self.label_pdf_dir.setStyleSheet("color: #1a7f37; font-size: 12px;")
            self._refresh_pdf_list()

    def _refresh_pdf_list(self):
        """刷新已发现的文档列表。"""
        self.list_pdf_files.clear()
        pdf_dir = Config.PDF_DIR
        if not os.path.isdir(pdf_dir):
            self.label_pdf_count.setText("目录不存在")
            return

        try:
            doc_files = sorted([
                f for f in os.listdir(pdf_dir)
                if f.lower().endswith((".pdf", ".docx", ".doc"))
                and os.path.isfile(os.path.join(pdf_dir, f))
            ])
        except Exception:
            doc_files = []
            self.label_pdf_count.setText("无法读取目录")
            return

        for name in doc_files:
            # 根据文件类型显示图标标记
            ext = os.path.splitext(name)[1].lower()
            prefix = "[PDF] " if ext == ".pdf" else "[DOC] "
            item = QListWidgetItem(prefix + name)
            item.setToolTip(name)
            item.setData(Qt.UserRole, name)  # 存储原始文件名
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)  # 默认全部勾选
            self.list_pdf_files.addItem(item)

        # 勾选状态变化时更新计数
        self.list_pdf_files.itemChanged.connect(self._update_pdf_count_label)
        self._update_pdf_count_label()


# ======================== 工具函数 ========================

def _escape_html(text: str) -> str:
    """转义 HTML 特殊字符。"""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _make_citation_links(text: str) -> str:
    """将 [片段N] 标记转换为可点击的 HTML 锚点链接。

    必须在 _escape_html 之后调用，因为注入的是原始 HTML。
    """
    return re.sub(
        r'\[片段(\d+)\]',
        r'<a href="#frag-\1" style="color:#2f81f7;text-decoration:none;font-weight:bold;">[片段\1]</a>',
        text,
    )
