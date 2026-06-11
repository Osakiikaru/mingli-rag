"""
Reranker 对比测试
-----------------
同一批查询，并排对比两种模式：
  混合RRF  vs  混合RRF + Reranker

用法：python scripts/test_rerank.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.hybrid_retriever import HybridRetriever

TEST_QUERIES = [
    "甲木生于冬月如何取用神",
    "伤官佩印格的条件是什么",
    "七杀格如何制化",
    "日主身弱喜用什么",
]

TOP_K = 5


def print_results(mode: str, results: list[dict]):
    print(f"\n  ── {mode} ──")
    for i, r in enumerate(results, 1):
        loc = r["source"]
        if r.get("chapter"):
            loc += f" · {r['chapter']}"
        if r.get("section"):
            loc += f" · {r['section']}"

        score_str = ""
        if "rerank_score" in r:
            score_str = f"rerank={r['rerank_score']:.4f}"
        elif "rrf_score" in r:
            score_str = f"rrf={r['rrf_score']:.5f}"

        preview = r["original"][:80].replace("\n", " ")
        print(f"  [{i}] {loc}  {score_str}")
        print(f"       {preview}…")


def main():
    retriever = HybridRetriever()

    print("=" * 70)
    for query in TEST_QUERIES:
        print(f"\n🔍 查询：{query}")
        print("-" * 60)

        hybrid_results = retriever.search(query, top_k=TOP_K, mode="hybrid")
        rerank_results = retriever.search(query, top_k=TOP_K, mode="hybrid", rerank=True)

        print_results("混合RRF", hybrid_results)
        print_results("混合RRF + Reranker", rerank_results)

    print("\n" + "=" * 70)
    print("✅ 对比完成")


if __name__ == "__main__":
    main()
