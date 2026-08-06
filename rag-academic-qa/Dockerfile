# ========================
# RAG 学术文献问答系统 — Dockerfile
# 支持两种模式:
#   CLI 模式: docker run rag-qa python main.py info
#   API 模式: docker run -p 7860:7860 rag-qa (Gradio Web 界面)
# ========================

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（PySide6 GUI 需要 libgl，CLI 模式不需要）
# --no-install-recommends 减小镜像体积
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单，利用 Docker 缓存层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p data/papers logs faiss_index

# 环境变量默认值（可通过 -e 覆盖）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_DIR=/app/logs \
    PDF_DIR=/app/data/papers \
    FAISS_INDEX_DIR=/app/faiss_index

# 默认入口：显示帮助信息
ENTRYPOINT ["python", "main.py"]
CMD ["info"]
