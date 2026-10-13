"""向量语义召回器：sentence-transformers 编码 + FAISS 内积检索。

解决关键词检索对同义词、口语化表达失效的问题。
默认使用 bge-small-zh-v1.5，可按需替换为任意 HuggingFace 嵌入模型。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import faiss


class VectorRetriever:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.docs: List[str] = []

    def build_index(self, docs: List[str], batch_size: int = 64) -> None:
        """对文档列表编码并构建 FAISS 内积索引。"""
        self.docs = docs
        embeddings = self.model.encode(docs, batch_size=batch_size, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.asarray(embeddings, dtype=np.float32))

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """返回 [{"index": int, "score": float, "text": str}]。"""
        if self.index is None:
            raise RuntimeError("请先调用 build_index()")
        q_vec = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.asarray(q_vec, dtype=np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({"index": int(idx), "score": float(score), "text": self.docs[idx]})
        return results
