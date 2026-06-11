"""
Day 7: 检索通路验证
-------------------
输入一个命理查询 → 从 ChromaDB 召回 Top-K 相关 chunk
用于验证 Day 6 构建的向量索引是否正常工作

用法：python scripts/test_retrieval.py
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_DIR      = Path("data/chroma_db")
MODEL_CACHE_DIR = Path("models")
COLLECTION_NAME = "mingli_chunks"
MODEL_NAME      = "BAAI/bge-m3"
TOP_K           = 5

# ── 测试查询（可以自由修改）──
TEST_QUERIES = [
    "甲木生于冬月如何取用神",
    "伤官佩印格的条件是什么",
    "七杀格如何制化",
    "日主身弱喜用什么",
]


def retrieve(query: str, collection, model, top_k: int = TOP_K):
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def main():
    print("加载模型中（已下载则秒开）...")
    model = SentenceTransformer(str(MODEL_CACHE_DIR / "bge-m3"))

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print(f"向量库已加载，共 {collection.count()} 条\n")
    print("=" * 60)

    for query in TEST_QUERIES:
        print(f"\n🔍 查询：{query}")
        print("-" * 50)

        results = retrieve(query, collection, model)

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = 1 - dist          # ChromaDB cosine distance → similarity
            source  = meta.get("source", "")
            chapter = meta.get("chapter", "")
            section = meta.get("section", "")
            loc     = f"{source} · {chapter}" + (f" · {section}" if section else "")

            print(f"\n  [{rank}] 相似度 {similarity:.3f} | {loc}")
            # 只展示前 120 字，避免刷屏
            preview = doc[:120].replace("\n", " ")
            print(f"      {preview}{'…' if len(doc) > 120 else ''}")

        print()

    print("=" * 60)
    print("✅ 检索验证完成，如果结果内容与查询相关，Day 6-7 目标达成")


if __name__ == "__main__":
    main()
