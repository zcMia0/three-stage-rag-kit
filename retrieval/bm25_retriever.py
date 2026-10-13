"""BM25 粗召回器：关键词匹配 + 电商商品属性加权。

解决长尾商品、同义词表述导致的检索失配问题。
- 对商品标题/描述做 jieba 分词后建立 BM25Okapi 索引
- 支持按商品属性（如类目、标签）加权，提升业务相关命中
"""
from __future__ import annotations

from typing import List

import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    """中文分词，过滤空白。"""
    return [w.strip() for w in jieba.lcut(text.lower()) if w.strip()]


class BM25Retriever:
    """基于 BM25Okapi 的关键词粗召回器。"""

    def __init__(self, docs: List[str], attr_weights: List[dict] | None = None):
        """
        Args:
            docs: 商品文档（标题+描述拼接后的文本）
            attr_weights: 与 docs 一一对应的属性权重（如 {"category": 1.2, "tags": 0.8}）
        """
        self.docs = docs
        self.attr_weights = attr_weights or [{} for _ in docs]
        self.tokenized = [tokenize(d) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 20) -> List[dict]:
        """返回 [{"index": int, "score": float, "text": str}]，分数已含属性加权。"""
        query_tokens = tokenize(query)
        raw_scores = self.bm25.get_scores(query_tokens)
        scored = []
        for i, score in enumerate(raw_scores):
            weight = 1.0
            for w in self.attr_weights[i].values():
                weight *= float(w)
            scored.append({"index": i, "score": score * weight, "text": self.docs[i]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
