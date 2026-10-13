"""三级 RAG 完整流水线：BM25粗召回 → 向量语义召回 → 交叉编码器精排 → 分层记忆。

这是本仓库的入口组件，将 retrieval 与 memory 两个模块串联起来。
"""
from __future__ import annotations

from typing import List

from retrieval.bm25_retriever import BM25Retriever
from retrieval.cross_encoder_reranker import CrossEncoderReranker
from retrieval.vector_retriever import VectorRetriever
from memory.hierarchical_memory import HierarchicalMemory

# 默认模型（首次运行会自动下载，可替换）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"


class ThreeStageRAG:
    def __init__(
        self,
        docs: List[str],
        attr_weights: List[dict] | None = None,
        bm25_top_k: int = 20,
        vector_top_k: int = 10,
        rerank_top_k: int = 3,
        use_reranker: bool = True,
    ):
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.rerank_top_k = rerank_top_k
        self.memory = HierarchicalMemory()

        self.bm25 = BM25Retriever(docs, attr_weights)
        self.vector = VectorRetriever(EMBEDDING_MODEL)
        self.vector.build_index(docs)
        self.reranker = CrossEncoderReranker(RERANKER_MODEL) if use_reranker else None

    def retrieve(self, query: str) -> List[dict]:
        """执行三级召回，返回精排后的文档列表。"""
        bm25_hits = {h["index"] for h in self.bm25.search(query, self.bm25_top_k)}
        vec_hits = {h["index"] for h in self.vector.search(query, self.vector_top_k)}

        # 合并两级召回（取并集，向量召回结果附语义分）
        index_map = {}
        for h in self.bm25.search(query, self.bm25_top_k):
            index_map[h["index"]] = {"index": h["index"], "text": h["text"], "bm25_score": h["score"]}
        for h in self.vector.search(query, self.vector_top_k):
            if h["index"] in index_map:
                index_map[h["index"]]["vector_score"] = h["score"]
            else:
                index_map[h["index"]] = {"index": h["index"], "text": h["text"], "vector_score": h["score"]}

        candidates = list(index_map.values())

        if self.reranker and len(candidates) > 1:
            candidates = self.reranker.rerank(query, candidates, self.rerank_top_k)
        else:
            # 无精排时按融合分排序
            candidates.sort(
                key=lambda c: c.get("vector_score", 0.0) + c.get("bm25_score", 0.0),
                reverse=True,
            )
            candidates = candidates[: self.rerank_top_k]
        return candidates

    def chat(self, query: str) -> dict:
        """单轮对话：检索 + 拼上下文 + （示意）生成回答 + 更新记忆。"""
        hits = self.retrieve(query)
        context = "\n".join(f"- {h['text']}" for h in hits)
        memory_ctx = self.memory.build_context()

        # 生产环境此处调用大模型；Demo 用模板化回答示意
        answer = f"根据检索结果，为您推荐：{hits[0]['text'][:30]}...（完整回答由大模型基于上下文生成）"

        before = self.memory.estimate_tokens()
        self.memory.add_turn(query, answer)
        after = self.memory.estimate_tokens()

        return {
            "answer": answer,
            "top_docs": [h["text"] for h in hits],
            "memory_tokens_before": before,
            "memory_tokens_after": after,
            "memory_context": memory_ctx,
        }
