import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from src.retrieval.bm25_retriever import BM25Retriever, _expand_query

# 1. 验证辰月的同义词扩展
tokens = _expand_query("丁火辰月生")
print("=== 辰月扩展结果 ===")
print(tokens)

# 2. 用新关键词检索
print()
print("=== 检索结果（丁火辰月生 三月丁火）===")
r = BM25Retriever.load()
results = r.search("丁火辰月生 三月丁火 调候用神", top_k=5)
for i, c in enumerate(results, 1):
    print(f"{i}. 《{c['source']}》{c['chapter']} | {c['section']} (score={c['bm25_score']:.3f})")
