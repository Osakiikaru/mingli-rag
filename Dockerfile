# ─────────────────────────────────────────────────────────
# 命理测算RAG — Docker 镜像
# 暴露：FastAPI  http://localhost:8000
# 挂载：./data    → /app/data   (chroma_db / bm25_index)
#       ./models  → /app/models (bge-m3 / reranker)
# ─────────────────────────────────────────────────────────

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# ── 系统依赖 ─────────────────────────────────────────────
# build-essential  → hnswlib (chromadb) 编译
# poppler-utils    → pdfplumber PDF 解析（可选，仅跑 pdf_to_txt.py 时需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ──────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── 复制项目代码 ─────────────────────────────────────────
# data/ 和 models/ 通过 docker-compose volume 挂载，不打入镜像
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY app.py     .

# ── 环境变量默认值 ────────────────────────────────────────
# 真正的 KEY 通过 --env-file .env 或 docker-compose 注入
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
# 容器内无 GPU，强制 CPU 推理
ENV CUDA_VISIBLE_DEVICES=""
# HuggingFace 模型缓存目录（与 models/ volume 对齐）
ENV HF_HOME=/app/models/hf_cache

# ── 健康检查 ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── 默认启动 FastAPI ─────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Streamlit 启动（demo 用途，替换 CMD 即可）────────────
# CMD ["streamlit", "run", "app.py", "--server.port=8501", \
#      "--server.address=0.0.0.0", "--server.fileWatcherType=none"]
# EXPOSE 8501
