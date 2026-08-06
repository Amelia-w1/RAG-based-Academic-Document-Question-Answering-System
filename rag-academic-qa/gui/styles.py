"""
GUI 全局样式表
=================
PySide6 QSS 浅色主题，统一控件视觉风格。
配色灵感参考 GitHub Light，柔和耐看。
"""

# ======================== 全局 QSS 样式 ========================
GLOBAL_QSS = """
/* ===== 全局 ===== */
QWidget {
    background-color: #f6f7f9;
    color: #1f2328;
    font-family: "Microsoft YaHei UI", "Segoe UI", "PingFang SC", sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #f6f7f9;
}

/* ===== 滚动条 ===== */
QScrollBar:vertical {
    background: #f0f2f5;
    width: 10px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #c4c9d0;
    min-height: 30px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #a8afb8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #f0f2f5;
    height: 10px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #c4c9d0;
    min-width: 30px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #a8afb8;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ===== 分组框 ===== */
QGroupBox {
    border: 1px solid #d8dee4;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: bold;
    color: #2f81f7;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* ===== 标签 ===== */
QLabel {
    background: transparent;
    color: #59636e;
}
QLabel#sectionTitle {
    font-size: 15px;
    font-weight: bold;
    color: #2f81f7;
    padding: 4px 0;
}
QLabel#statusOk {
    color: #1a7f37;
    font-weight: bold;
}
QLabel#statusWarn {
    color: #9a6700;
    font-weight: bold;
}
QLabel#statusErr {
    color: #cf222e;
    font-weight: bold;
}
QLabel#citationFile {
    font-weight: bold;
    color: #2f81f7;
}
QLabel#citationMeta {
    color: #9a6700;
    font-size: 12px;
}

/* ===== 输入框 ===== */
QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    border: 1px solid #d8dee4;
    border-radius: 6px;
    padding: 6px 10px;
    color: #1f2328;
    selection-background-color: #2f81f7;
    selection-color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #2f81f7;
}
QTextEdit[readOnly="true"], QPlainTextEdit[readOnly="true"] {
    background-color: #f0f2f5;
}

/* ===== 下拉框 ===== */
QComboBox {
    background-color: #ffffff;
    border: 1px solid #d8dee4;
    border-radius: 6px;
    padding: 5px 10px;
    color: #1f2328;
}
QComboBox:hover {
    border: 1px solid #2f81f7;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #59636e;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #d8dee4;
    selection-background-color: #2f81f7;
    selection-color: #ffffff;
    outline: none;
}

/* ===== 按钮 ===== */
QPushButton {
    background-color: #ffffff;
    border: 1px solid #d8dee4;
    border-radius: 6px;
    padding: 7px 18px;
    color: #1f2328;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #f0f2f5;
    border: 1px solid #2f81f7;
}
QPushButton:pressed {
    background-color: #e4e7ea;
}
QPushButton:disabled {
    background-color: #f0f2f5;
    color: #a8afb8;
    border: 1px solid #d8dee4;
}

QPushButton#primaryBtn {
    background-color: #2f81f7;
    border: none;
    color: #ffffff;
}
QPushButton#primaryBtn:hover {
    background-color: #1c6fdd;
}
QPushButton#primaryBtn:pressed {
    background-color: #1860c5;
}
QPushButton#primaryBtn:disabled {
    background-color: #b0c8ef;
    color: #ffffff;
}

QPushButton#dangerBtn {
    background-color: #cf222e;
    border: none;
    color: #ffffff;
}
QPushButton#dangerBtn:hover {
    background-color: #b41d28;
}
QPushButton#dangerBtn:pressed {
    background-color: #9e1a24;
}

/* ===== 进度条 ===== */
QProgressBar {
    background-color: #e4e7ea;
    border: none;
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: #1f2328;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #2f81f7;
    border-radius: 5px;
}

/* ===== 分割线 ===== */
QFrame#separator {
    background-color: #d8dee4;
    max-height: 1px;
    min-height: 1px;
    border: none;
}

/* ===== 文本浏览器（问答显示区） ===== */
QTextBrowser {
    background-color: #ffffff;
    border: 1px solid #d8dee4;
    border-radius: 8px;
    padding: 12px;
    color: #1f2328;
}

/* ===== 工具提示 ===== */
QToolTip {
    background-color: #ffffff;
    color: #1f2328;
    border: 1px solid #d8dee4;
    border-radius: 4px;
    padding: 4px 8px;
}
"""
