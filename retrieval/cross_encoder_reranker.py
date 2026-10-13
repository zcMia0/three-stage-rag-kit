"""交叉编码器精排器：对候选对 (query, doc) 联合编码打分。

相比双塔向量检索，Cross-Encoder 精度更高，是三级 RAG 的最后一环。
默认使用 bge-reranker-base。
"""
from __future__ import annotations

from typing import List


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[dict], top_k: int = 3) -> List[dict]:
        """对候选列表精排，返回重排后的 top_k。"""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]
