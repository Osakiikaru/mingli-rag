"""
scripts/debug_imports.py
逐步测试各个模块导入，定位静默崩溃的来源
用法：python scripts/debug_imports.py
"""
# ⚠️ 必须在所有其他 import 之前设置，禁用 tokenizers 并行（Windows fork 问题）
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""        # 禁用 CUDA 检测
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ""
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "models"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test(label, fn):
    print(f"  测试: {label} ...", flush=True)
    fn()
    print(f"  ✅  {label} OK", flush=True)

print("=== 导入诊断 ===", flush=True)

test("pathlib",         lambda: __import__("pathlib"))
test("json",            lambda: __import__("json"))
test("chromadb",          lambda: __import__("chromadb"))
test("torch",             lambda: __import__("torch"))
test("numpy",             lambda: __import__("numpy"))
test("scipy",             lambda: __import__("scipy"))
test("sklearn",           lambda: __import__("sklearn"))
test("huggingface_hub",   lambda: __import__("huggingface_hub"))
test("tqdm",              lambda: __import__("tqdm"))
test("BM25Retriever",   lambda: __import__("src.retrieval.bm25_retriever", fromlist=["BM25Retriever"]))
test("DirectEncoder",   lambda: __import__("src.retrieval.direct_encoder", fromlist=["DirectEncoder"]))
test("HybridRetriever", lambda: __import__("src.retrieval.hybrid_retriever", fromlist=["HybridRetriever"]))
test("ragas",           lambda: __import__("ragas"))
test("ragas.metrics",   lambda: __import__("ragas.metrics"))
test("datasets",        lambda: __import__("datasets"))
test("jieba",           lambda: __import__("jieba"))
test("rank_bm25",       lambda: __import__("rank_bm25"))
test("BM25Retriever",   lambda: __import__("src.retrieval.bm25_retriever", fromlist=["BM25Retriever"]))
test("HybridRetriever", lambda: __import__("src.retrieval.hybrid_retriever", fromlist=["HybridRetriever"]))
test("ragas",           lambda: __import__("ragas"))
test("ragas.metrics",   lambda: __import__("ragas.metrics"))
test("datasets",        lambda: __import__("datasets"))

print("\n=== 全部导入成功 ===", flush=True)
