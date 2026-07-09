#!/usr/bin/env python3
"""
测试 LLM 评分功能

演示如何使用 LLM 评分（0-100分）替代金钱逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.item_base import Item
from src.bidder_base import Bidder

def test_llm_scoring():
    """测试 LLM 评分功能"""

    print("=" * 60)
    print("LLM 评分功能测试")
    print("=" * 60)

    # 测试 1: 创建图像评估模式的物品
    print("\n【测试 1】创建图像评估物品")
    item = Item(
        id=1,
        name="Abstract Artwork #001",
        price=100,
        image_path="data/images/001.jpg",
        context="评估该作品在现代办公空间的适配度",
        style_tags=["abstract", "minimalist"],
        llm_score=None  # 初始时为 None，会由 LLM 自动评分
    )
    print(f"✓ 创建物品: {item.name}")
    print(f"  is_image_mode: {item.is_image_mode}")
    print(f"  initial llm_score: {item.llm_score}")

    # 测试 2: 创建 bidder 并获取 LLM 评分
    print("\n【测试 2】创建 Bidder 并测试评分")
    bidder = Bidder.create(
        name="Test Bidder",
        model_name="gpt-4-0613",
        budget=1000,
        desire="aesthetic_alignment",
        plan_strategy="adaptive",
        correct_belief=True,
        auction_hash="test_001"
    )
    print(f"✓ 创建 Bidder: {bidder.name}")

    # 注意: 实际调用 _get_llm_score 需要有效的 API key
    # 这里仅演示接口
    print(f"  准备调用 LLM 评分...")
    # score = bidder._get_llm_score(item)
    # print(f"  LLM 评分: {score:.1f}/100")

    # 测试 3: 模拟中标并计算"利润"（实际是评分）
    print("\n【测试 3】模拟中标流程")
    # 手动设置评分进行演示
    item.llm_score = 75.5

    bid_price = 150
    initial_profit = bidder.profit

    win_msg = bidder.win_bid(item, bid_price)
    print(f"✓ {win_msg}")
    print(f"  预算变化: ${bidder.budget + bid_price} → ${bidder.budget}")
    print(f"  美学评分变化: {initial_profit} → {bidder.profit}")
    print(f"  该物品的 LLM 评分: {item.llm_score}/100")

    # 测试 4: 生成报告
    print("\n【测试 4】生成拍卖报告")
    report = bidder.profit_report()
    print(report)

    print("\n" + "=" * 60)
    print("功能说明")
    print("=" * 60)
    print("""
1. 图像评估模式 (is_image_mode=True):
   - 物品不再有 true_value 字段
   - 使用 llm_score (0-100) 代替价值

2. 利润计算逻辑改变:
   - 原版模式: profit = true_value - bid_price
   - 图像模式: profit = 累积的 LLM 评分总和

3. 评分缓存机制:
   - 首次调用 _get_llm_score() 时，调用 LLM 评分
   - 评分存储在 item.llm_score 中
   - 后续调用直接返回缓存值，避免重复 API 调用

4. 特殊处理:
   - rule bidder: 使用默认评分 50.0
   - human bidder: 使用默认评分 50.0
   - LLM bidder: 真实调用 LLM 进行评分

5. 报告显示:
   - 原版模式: "total profit of $XXX"
   - 图像模式: "total aesthetic score of XX.X/100"
    """)

    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    print("""
# 在 items_demo.jsonl 中定义物品:
{
    "id": 1,
    "name": "Artwork #001",
    "price": 100,
    "image_path": "data/images/001.jpg",
    "context": "评估该作品的美学价值",
    "style_tags": ["abstract", "modern"]
}

# 运行拍卖:
python auction_workflow.py --input_dir data/example/ --threads 2 --monte_carlo --num_runs 30

# 结果解读:
- 每个 bidder 的 profit 字段 = 获得的所有物品的 LLM 评分之和
- Monte Carlo 统计会显示平均美学评分、标准差等
- 可以用来比较不同 bidder 的审美偏好一致性
    """)


if __name__ == '__main__':
    test_llm_scoring()
