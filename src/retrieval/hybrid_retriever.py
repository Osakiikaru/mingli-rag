"""
混合检索器：BGE-M3 向量检索 + BM25 关键词检索 + RRF 融合
----------------------------------------------------------

RRF（Reciprocal Rank Fusion）公式：
  score(d) = Σ_i  1 / (k + rank_i(d))
  k=60 是经典默认值（来自 Cormack et al. 2009），
  作用是平滑排名靠后的文档的贡献，避免低排名主导最终分数。

用法：
  from src.retrieval.hybrid_retriever import HybridRetriever
  r = HybridRetriever()
  results = r.search("冬月甲木取用神", top_k=5)
"""

from pathlib import Path

import chromadb

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.direct_encoder import DirectEncoder

# ── 配置 ──────────────────────────────────────────────
CHROMA_DIR      = Path("data/chroma_db")
MODEL_CACHE_DIR = Path("models")
COLLECTION_NAME = "mingli_chunks"
CANDIDATE_K     = 20    # 每路各取 20 个候选，融合后取 top_k
RRF_K           = 60    # RRF 平滑常数
# ──────────────────────────────────────────────────────


class HybridRetriever:
    def __init__(self):
        print("── 初始化混合检索器 ──")

        # 向量检索组件（使用 DirectEncoder，不依赖 sentence_transformers）
        print("  加载 BGE-M3...")
        self.model = DirectEncoder(MODEL_CACHE_DIR / "bge-m3")
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        print(f"  ChromaDB 已加载，{self.collection.count()} 条向量")

        # BM25 组件
        print("  加载 BM25 索引...")
        self.bm25 = BM25Retriever.load()

        print("  ✅ 混合检索器就绪\n")

    # ── 各路检索 ──────────────────────────────────────

    def _vector_search(self, query: str, top_k: int) -> list[str]:
        """返回 chunk_id 列表，按向量相似度排序"""
        q_emb = self.model.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        return res["ids"][0]   # list[str]

    def _bm25_search(self, query: str, top_k: int) -> list[str]:
        """返回 chunk_id 列表，按 BM25 分数排序"""
        results = self.bm25.search(query, top_k=top_k)
        return [r["chunk_id"] for r in results]

    # ── RRF 融合 ──────────────────────────────────────

    @staticmethod
    def _rrf_fusion(
        ranked_lists: list[list[str]],
        k: int = RRF_K,
    ) -> list[tuple[str, float]]:
        """
        输入：多路 chunk_id 排名列表
        输出：[(chunk_id, rrf_score), ...] 降序排列
        """
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, chunk_id in enumerate(ranked):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── 主接口 ────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = CANDIDATE_K,
        mode: str = "hybrid",   # "hybrid" | "vector" | "bm25"
        rerank: bool = False,   # True = 在 hybrid 基础上加 Reranker 精排
    ) -> list[dict]:
        """
        mode:
          "hybrid" — 向量 + BM25 + RRF（默认）
          "vector" — 仅向量检索
          "bm25"   — 仅 BM25 检索
        rerank:
          True — 先 hybrid 取 candidate_k 候选，再用 Reranker 精排取 top_k
        """
        if mode == "vector":
            ids = self._vector_search(query, top_k)
            return self._ids_to_results(ids, query)

        if mode == "bm25":
            return self.bm25.search(query, top_k=top_k)

        # hybrid：两路各取候选，RRF 融合
        vec_ids  = self._vector_search(query, candidate_k)
        bm25_ids = self._bm25_search(query, candidate_k)

        # rerank 模式：先多取候选再精排；普通模式：直接取 top_k
        n_fused = candidate_k if rerank else top_k
        fused = self._rrf_fusion([vec_ids, bm25_ids])[:n_fused]
        top_ids = [chunk_id for chunk_id, _ in fused]
        rrf_scores = {chunk_id: score for chunk_id, score in fused}

        results = self._ids_to_results(top_ids, query)
        for r in results:
            r["rrf_score"] = round(rrf_scores.get(r["chunk_id"], 0.0), 6)

        if rerank:
            from src.retrieval.reranker import Reranker
            if not hasattr(self, "_reranker"):
                self._reranker = Reranker()
            results = self._reranker.rerank(query, results, top_n=top_k)

        return results

    def _ids_to_results(self, chunk_ids: list[str], query: str) -> list[dict]:
        """根据 chunk_id 列表从 BM25 索引中取回完整数据"""
        # 建立 chunk_id → chunk 的快速查找表（用 BM25Retriever 的 chunks 列表）
        id_map = {c["id"]: c for c in self.bm25.chunks}
        results = []
        for cid in chunk_ids:
            c = id_map.get(cid)
            if c:
                results.append({
                    "chunk_id":   c["id"],
                    "source":     c["source"],
                    "chapter":    c.get("chapter", ""),
                    "section":    c.get("section", ""),
                    "original":   c["original"],
                    "annotation": c.get("annotation", ""),
                })
        return results
