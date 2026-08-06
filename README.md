# RAG 学术文献问答系统

基于 RAG（Retrieval-Augmented Generation）的本地论文问答工具，支持 PDF/Word 文档建库、语义检索、重排序、引用溯源，以及 CLI / GUI 两种交互方式。
## 特性

- 支持批量导入论文并构建 FAISS 向量库
- 支持向量检索 + gte-rerank 重排序
- 支持 HyDE、Multi-Query、Hybrid 等高级检索策略
- 支持多轮对话与查询改写
- 支持流式输出回答
- 支持引用片段、文件名、页码溯源
- 支持 PySide6 桌面端界面

## 技术栈

- LangChain
- LangGraph
- FAISS
- DashScope
- pypdf / docx2txt
- PySide6

## 目录结构

```text
rag-academic-qa/
├── main.py
├── gui_app.py
├── config.py
├── modules/
├── gui/
├── evaluation/
├── experiments/
├── data/papers/
└── faiss_index/
```

## 环境要求

- Python 3.10+
- DashScope API Key

## 安装

```bash
git clone <your-repo-url>
cd rag-academic-qa
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置

复制环境变量模板并填写 API Key：

```bash
Copy-Item .env.example .env
```

如果你在 Bash / Git Bash 下，也可以用：

```bash
cp .env.example .env
```

至少需要配置：

```env
DASHSCOPE_API_KEY=sk-your-api-key
```

可选参数包括：

- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `RERANK_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `RETRIEVAL_K`
- `RERANK_TOP_N`
- `TEMPERATURE`

## 准备文档

把论文放到 `data/papers/` 目录下，支持：

- `.pdf`
- `.docx`
- `.doc`

## 使用

### 1. 构建向量库

```bash
python main.py build
```

强制重建：

```bash
python main.py build --force
```

### 2. 单次问答

```bash
python main.py ask "什么是 Restormer？"
```

指定检索策略：

```bash
python main.py ask "什么是 Restormer？" --strategy hybrid
```

### 3. 多轮对话

```bash
python main.py chat
```

### 4. 查看配置

```bash
python main.py info
```

### 5. 启动 GUI

```bash
python gui_app.py
```

## 配置说明

默认配置来自 `.env`，常见调整项：

- `CHUNK_SIZE`：切片大小
- `CHUNK_OVERLAP`：切片重叠
- `RETRIEVAL_K`：召回候选数
- `RERANK_TOP_N`：重排序后保留数
- `TEMPERATURE`：模型温度

## 工作流程

```text
用户提问
  -> 查询改写
  -> 向量检索
  -> 重排序
  -> 生成回答
  -> 引用整理
```

## 测试

```bash
pytest
```

## 展示
![GUI Screenshot](assets/readme/pic.png)
