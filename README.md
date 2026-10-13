# Three-Stage RAG Kit
电商场景三级检索增强生成工具包，针对长尾商品、同义词表述、长对话上下文丢失等痛点，实现「粗召回-语义召回-精排」三级架构，配套分层对话记忆优化。 

## ✨ 核心特性
- **三级召回架构**：BM25关键词粗召回 → 向量语义召回 → 交叉编码器（Cross-Encoder）精排，兼顾召回率与准确率
- **业务增强策略**：内置商品属性加权、类目匹配校验，解决电商场景长尾商品检索失配问题
- **分层对话记忆**：实现三级分层会话记忆策略，自动压缩历史上下文，显著降低长对话Token开销
- **可插拔组件**：支持替换向量数据库、Embedding模型、交叉编码器，兼容LangChain生态

## 🏗️ 架构设计
```
用户Query → 查询改写与扩展 → BM25粗召回（Top20）
                          ↓
                    向量语义召回（Top10）
                          ↓
                    交叉编码器精排（Top3）
                          ↓
                    上下文拼接 → 大模型生成回答
                          ↓
                    分层记忆更新（摘要/关键信息/最新轮）
```

## 🚀 快速开始
### 环境要求
- Python 3.10+
- 建议使用 conda / venv 虚拟环境

### 安装
```bash
git clone https://github.com/zcMia0/three-stage-rag-kit.git
cd three-stage-rag-kit
pip install -r requirements.txt
```

### 运行示例（内置电商商品测试数据）
```bash
python examples/run_demo.py
```

示例输入：
> "有没有适合夏天用的轻薄款防晒衣，要大码的"

输出将展示三级召回的每一步结果对比、最终回答、Token消耗统计。

## 📊 性能基准
**测试集**：电商商品知识库1000条SKU + 200条用户真实咨询Query
| 指标 | 纯向量检索 | 三级RAG架构 | 提升幅度 |
|------|------------|-------------|----------|
| 检索准确率 | 72% | 89% | +17pp |
| 召回率 | 71% | 93% | +22pp |
| 10轮长对话单会话Token | 1.8k | 990 | -45% |
| 长对话上下文准确率 | 61% | 96% | +35pp |

> 测试环境：bge-small-zh-v1.5 嵌入模型 + bge-reranker-base 交叉编码器

## 📁 目录结构
```
three-stage-rag-kit/
├── retrieval/          # 三级召回核心模块
│   ├── bm25_retriever.py
│   ├── vector_retriever.py
│   └── cross_encoder_reranker.py
├── memory/             # 分层对话记忆模块
│   └── hierarchical_memory.py
├── examples/           # 示例脚本与测试数据
├── config/             # 配置文件
└── requirements.txt
```

## 🛠️ 技术栈
- 核心框架：LangChain
- 向量检索：FAISS / Chroma
- 关键词检索：BM25Okapi
- 精排模型：BGE Cross-Encoder
- 嵌入模型：BGE Embedding

## 📌 扩展说明
实际生产环境中，可将向量库替换为Milvus/Qdrant，接入Redis会话缓存，支持高并发调用。
