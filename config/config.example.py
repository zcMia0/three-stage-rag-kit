# Three-Stage RAG 配置示例（复制为 config.py 后按需修改）
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# 召回参数
BM25_TOP_K = 20
VECTOR_TOP_K = 10
RERANK_TOP_K = 3

# 记忆参数
MAX_RECENT_ROUNDS = 4
SUMMARY_TOKEN_BUDGET = 300

# 生产环境向量库（切换为 Milvus/Qdrant 时使用）
VECTOR_DB = "faiss"  # faiss | milvus | qdrant
