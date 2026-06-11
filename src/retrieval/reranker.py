"""
Reranker：本地运行 bge-reranker-v2-m3 对候选文档精排
----------------------------------------------------
首次运行自动从镜像下载模型（~570MB），之后直接加载本地缓存。

用法：
  from src.retrieval.reranker import Reranker
  r = Reranker()
  results = r.rerank(query="甲木冬月取用神", candidates=[...], top_n=5)
"""

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from src.retrieval.direct_encoder import DirectCrossEncoder

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_CACHE_DIR = Path("models")
MODEL_NAME      = "BAAI/bge-reranker-v2-m3"
LOCAL_PATH      = MODEL_CACHE_DIR / "bge-reranker-v2-m3"


class Reranker:
    def __init__(self):
        if not LOCAL_PATH.exists():
            print("  首次运行，下载 bge-reranker-v2-m3（~570MB）...")
            MODEL_CACHE_DIR.mkdir(exist_ok=True)
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=str(LOCAL_PATH),
                local_dir_use_symlinks=False,
            )
            print("  下载完成")
        else:
            print("  bge-reranker-v2-m3 已存在，直接加载")

        self.model = DirectCrossEncoder(str(LOCAL_PATH), max_length=512)
        print("  ✅ Reranker 加载完成")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = 5,
        text_field: str = "original",
    ) -> list[dict]:
        """
        对候选列表精排，返回 top_n 条，每条附加 rerank_score 字段。
        """
        if not candidates:
            return []

        pairs = [(query, c[text_field]) for c in candidates]
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )[:top_n]

        results = []
        for score, chunk in scored:
            r = chunk.copy()
            r["rerank_score"] = round(float(score), 4)
            results.append(r)

        return results
