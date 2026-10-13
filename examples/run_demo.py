"""演示脚本：内置电商商品测试数据，跑通三级 RAG 全流程并输出对比。

用法：python examples/run_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.pipeline import ThreeStageRAG

# 内置商品知识库（Demo 用精简数据，生产为千级 SKU 库）
PRODUCTS = [
    "夏季轻薄款防晒衣 UPF50+ 大码 透气冰丝面料 男女同款",
    "防晒冰袖 男女通用 骑行户外 冰丝凉感 弹力大码",
    "轻薄羽绒服 冬季保暖 90%白鸭绒 连帽 修身款",
    "无线蓝牙耳机 主动降噪 长续航30小时 Type-C快充",
    "无线蓝牙耳机 半入耳 通话降噪 防水运动款",
    "智能手表 心率血氧监测 50米防水 14天续航",
    "机械键盘 87键 热插拔 客制化 RGB背光 茶轴",
    "办公鼠标 静音无线 人体工学 2.4G+蓝牙双模",
    "大码女装 宽松显瘦 夏季碎花连衣裙 度假风",
    "男士休闲衬衫 夏季短袖 冰丝速干 商务通勤",
    "婴幼儿奶粉 3段 900g 进口奶源 益生菌配方",
    "猫粮 成猫 冻干双拼 鸡肉味 5kg大包装",
]

ATTR_WEIGHTS = [
    {"category": 1.2}, {"category": 1.2}, {"category": 1.2}, {"category": 1.0},
    {"category": 1.0}, {"category": 1.0}, {"category": 1.0}, {"category": 1.0},
    {"category": 1.2}, {"category": 1.0}, {"category": 1.2}, {"category": 1.2},
]

TEST_QUERIES = [
    "有没有适合夏天用的轻薄款防晒衣，要大码的",
    "推荐一款降噪的无线耳机",
    "夏天穿什么裙子显瘦",
    "给猫主子买点口粮",
]


def main():
    print("=" * 60)
    print("Three-Stage RAG Demo（电商长尾商品检索）")
    print("=" * 60)

    rag = ThreeStageRAG(PRODUCTS, ATTR_WEIGHTS, use_reranker=False)

    for q in TEST_QUERIES:
        print(f"\n【用户】{q}")
        result = rag.chat(q)
        print(f"【回答】{result['answer']}")
        print(f"【精排Top文档】")
        for i, doc in enumerate(result["top_docs"], 1):
            print(f"  {i}. {doc}")
        print(f"【记忆Token】本轮前 {result['memory_tokens_before']} → 本轮后 {result['memory_tokens_after']}")


if __name__ == "__main__":
    main()
