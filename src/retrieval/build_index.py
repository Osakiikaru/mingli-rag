"""
Day 6: ChromaDB + BGE-M3 向量索引构建
--------------------------------------
读取 data/processed/*_chunks.json（语义切分版，排除 baseline）
→ BGE-M3 对 original 字段做 embedding（annotation 为空时用 original 代替）
→ 存入 ChromaDB 持久化集合，带 metadata 支持过滤检索

Week 2 批量生成注解后，重新跑此脚本即可重建索引（幂等）
"""

import os
import json
from pathlib import Path

# 必须在所有 HuggingFace 相关 import 之前设置，否则不生效
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# ── 配置 ──────────────────────────────────────────────
PROCESSED_DIR   = Path("data/processed")
CHROMA_DIR      = Path("data/chroma_db")
MODEL_CACHE_DIR = Path("models")           # 模型缓存在项目文件夹内，不占 C 盘
COLLECTION_NAME = "mingli_chunks"
MODEL_NAME      = "BAAI/bge-m3"
BATCH_SIZE      = 64          # 显存不够时调小到 32
# ──────────────────────────────────────────────────────


def load_all_chunks() -> list[dict]:
    """读取所有语义切分 JSON，排除 baseline 对照组"""
    chunks = []
    for jf in sorted(PROCESSED_DIR.glob("*_chunks.json")):
        if "baseline" in jf.name:
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        chunks.extend(data)
        print(f"  {jf.name:<30} {len(data):>4} chunks")
    return chunks


def build_index():
    # ── Step 1: 加载 chunks ──
    print("── Step 1: 加载 chunks ──")
    chunks = load_all_chunks()
    print(f"  总计: {len(chunks)} chunks\n")

    # ── Step 2: 加载 BGE-M3 ──
    print("── Step 2: 加载 BGE-M3 模型 ──")
    model_local_path = MODEL_CACHE_DIR / "bge-m3"
    if not model_local_path.exists():
        print("  首次运行，从镜像下载 ~2.3GB，请耐心等待...")
        MODEL_CACHE_DIR.mkdir(exist_ok=True)
        snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=str(model_local_path),
            local_dir_use_symlinks=False,   # Windows 不用符号链接，直接下实体文件
        )
        print("  下载完成")
    else:
        print("  模型已存在，直接加载")
    model = SentenceTransformer(str(model_local_path))
    print("  模型加载完成\n")

    # ── Step 3: 初始化 ChromaDB ──
    print("── Step 3: 初始化 ChromaDB ──")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # 幂等：已存在则先删除重建
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  已删除旧集合 [{COLLECTION_NAME}]，重新构建")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # 余弦相似度
    )
    print(f"  集合 [{COLLECTION_NAME}] 创建完成\n")

    # ── Step 4: 批量 embedding + 写入 ──
    print("── Step 4: Embedding + 写入 ChromaDB ──")

    ids       = [c["id"] for c in chunks]
    # annotation 为空时用 original 代替，Week 2 生成注解后重跑本脚本即可
    texts     = [c["annotation"].strip() or c["original"] for c in chunks]
    metadatas = [
        {
            "source":  c["source"],
            "chapter": c.get("chapter", ""),
            "section": c.get("section", ""),
            "type":    c.get("type", "理论"),
        }
        for c in chunks
    ]
    # 原文单独存一份，检索命中后可以直接取用
    originals = [c["original"] for c in chunks]

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="  Embedding"):
        batch_slice = slice(i, i + BATCH_SIZE)

        embeddings = model.encode(
            texts[batch_slice],
            normalize_embeddings=True,   # 余弦相似度必须归一化
            show_progress_bar=False,
        ).tolist()

        collection.add(
            ids        = ids[batch_slice],
            embeddings = embeddings,
            documents  = originals[batch_slice],   # 存原文，方便检索后直接展示
            metadatas  = metadatas[batch_slice],
        )

    total = collection.count()
    print(f"\n✅ 索引构建完成，共写入 {total} 条向量")
    print(f"   持久化路径：{CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    build_index()
