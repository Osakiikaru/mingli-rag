"""
direct_encoder.py
-----------------
用 transformers 直接加载 BGE-M3 / bge-reranker-v2-m3，
完全绕开 sentence_transformers（在某些 Windows 环境下会崩溃）。

接口与 SentenceTransformer / CrossEncoder 完全兼容，
hybrid_retriever.py 和 reranker.py 可无缝替换。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class DirectEncoder:
    """
    替代 SentenceTransformer，直接用 transformers 加载编码器。
    适用于 BGE-M3 等双编码器模型。
    """

    def __init__(self, model_path: str | Path, device: str | None = None):
        model_path = str(model_path)
        if device is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        # GPU 用 float16 节省显存并加速；CPU 保持 float32 保证精度
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        print(f"  DirectEncoder 运行于 {self.device.upper()}（{dtype}）")

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 16,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                # CLS token 表示（BGE 系列模型用法）
                emb = outputs.last_hidden_state[:, 0, :]
                if normalize_embeddings:
                    emb = F.normalize(emb, p=2, dim=1)

            all_embeddings.append(emb.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)


class DirectCrossEncoder:
    """
    替代 CrossEncoder，直接用 transformers 加载重排序模型。
    适用于 bge-reranker-v2-m3。
    """

    def __init__(self, model_path: str | Path, max_length: int = 512, device: str | None = None):
        model_path = str(model_path)
        self.max_length = max_length
        if device is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, torch_dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"  DirectCrossEncoder 运行于 {self.device.upper()}（{dtype}）")

    def predict(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 8,
        **kwargs,
    ) -> list[float]:
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            queries = [p[0] for p in batch_pairs]
            texts   = [p[1] for p in batch_pairs]

            encoded = self.tokenizer(
                queries,
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                # bge-reranker 输出 logits shape: (batch, 1)
                batch_scores = outputs.logits.squeeze(-1).cpu().float().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)

        return scores
