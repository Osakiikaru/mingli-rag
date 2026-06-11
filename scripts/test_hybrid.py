"""
混合检索对比测试
---------------
同一批查询，并排对比三种模式的 Top-5 结果：
  纯向量 / 纯BM25 / 混合RRF

用法：python scripts/test_hybrid.py
"""

import sys
from pathlib import Path

# 让 src 包可以被 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever

TEST_QUERIES = [
    "甲木生于冬月如何取用神",
    "伤官佩印格的条件是什么",
    "七杀格如何制化",
    "日主身弱喜用什么",
]

TOP_K = 5


def print_results(mode: str, query: str, results: list[dict]):
    print(f"\n  ── {mode} ──")
    for i, r in enumerate(results, 1):
        loc = r["source"]
        if r.get("chapter"):
            loc += f" · {r['chapter']}"
        if r.get("section"):
            loc += f" · {r['section']}"

        # 显示对应分数
        score_str = ""
        if "rrf_score" in r:
            score_str = f"RRF={r['rrf_score']:.5f}"
        elif "bm25_score" in r:
            score_str = f"BM25={r['bm25_score']:.3f}"

        preview = r["original"][:80].replace("\n", " ")
        print(f"  [{i}] {loc}  {score_str}")
        print(f"       {preview}…")


def main():
    retriever = HybridRetriever()

    print("=" * 70)
    for query in TEST_QUERIES:
        print(f"\n🔍 查询：{query}")
        print("-" * 60)

        vec_results    = retriever.search(query, top_k=TOP_K, mode="vector")
        bm25_results   = retriever.search(query, top_k=TOP_K, mode="bm25")
        hybrid_results = retriever.search(query, top_k=TOP_K, mode="hybrid")

        print_results("纯向量", query, vec_results)
        print_results("纯BM25", query, bm25_results)
        print_results("混合RRF", query, hybrid_results)

    print("\n" + "=" * 70)
    print("✅ 对比完成")


if __name__ == "__main__":
    main()
