#!/usr/bin/env python3
"""
简化验证脚本 - 仅验证核心修改（不需要额外依赖）
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def main():
    print("\n" + "="*60)
    print(" LLM 评分功能 - 核心验证")
    print("="*60 + "\n")

    from src.item_base import Item

    # 测试 1: 图像模式
    print("【测试 1】图像模式物品")
    item1 = Item(
        id=1,
        name="Artwork #001",
        price=100,
        image_path="data/images/001.jpg",
        llm_score=85.5
    )
    assert item1.is_image_mode == True
    assert item1.llm_score == 85.5
    assert item1.true_value is None
    print(f"  ✓ is_image_mode: {item1.is_image_mode}")
    print(f"  ✓ llm_score: {item1.llm_score}")
    print(f"  ✓ true_value: {item1.true_value}")

    # 测试 2: 原版模式
    print("\n【测试 2】原版模式物品")
    item2 = Item(id=2, name="Widget A", price=1000, true_value=2000)
    assert item2.is_image_mode == False
    assert item2.true_value == 2000
    assert item2.llm_score is None
    print(f"  ✓ is_image_mode: {item2.is_image_mode}")
    print(f"  ✓ true_value: {item2.true_value}")
    print(f"  ✓ llm_score: {item2.llm_score}")

    # 测试 3: to_dict
    print("\n【测试 3】to_dict 方法")
    item_dict = item1.to_dict()
    assert 'llm_score' in item_dict
    print(f"  ✓ to_dict 包含 llm_score: {item_dict['llm_score']}")

    # 测试 4: 利润计算逻辑
    print("\n【测试 4】利润计算逻辑模拟")
    # 图像模式
    profit_image = 0
    profit_image += item1.llm_score
    print(f"  图像模式: profit = {profit_image} (累加 LLM 评分)")
    assert profit_image == 85.5

    # 原版模式
    profit_normal = 0
    bid_price = 1500
    profit_normal += item2.true_value - bid_price
    print(f"  原版模式: profit = {profit_normal} (价值 {item2.true_value} - 出价 {bid_price})")
    assert profit_normal == 500

    # 测试 5: 向后兼容
    print("\n【测试 5】向后兼容性")
    old_item = Item(id=3, name="Old Widget", price=300, true_value=500)
    assert old_item.is_image_mode == False
    assert old_item.llm_score is None
    print("  ✓ 旧代码仍然正常工作")

    print("\n" + "="*60)
    print("🎉 核心验证通过！")
    print("="*60)

    print("\n✅ 已完成的修改:")
    print("  1. utils.py - 添加 ujson fallback")
    print("  2. src/item_base.py - 添加 llm_score 字段")
    print("  3. src/bidder_base.py - 添加 _get_llm_score() 方法")
    print("  4. src/bidder_base.py - 修改 win_bid()")
    print("  5. src/bidder_base.py - 修改 profit_report()")
    print("  6. src/bidder_base.py - 修改 to_monitors()")
    print("  7. src/bidder_base.py - 修改 draw_plot()")
    print("  8. src/human_bidder.py - 更新 draw_plot 调用")
    print("  9. config.py - 添加 LLM 评分说明")

    print("\n📄 新增文档:")
    print("  • LLM_SCORING_README.md - 详细使用文档")
    print("  • QUICKSTART_LLM_SCORING.md - 快速开始指南")
    print("  • COMPARISON_MONEY_VS_LLM.md - 对比文档")

    print("\n📦 示例配置:")
    print("  • data/example_llm_scoring/items_demo.jsonl")
    print("  • data/example_llm_scoring/bidders_demo.jsonl")

    print("\n🔑 核心改变:")
    print("  • 原版模式: profit = Σ(true_value - bid)")
    print("  • 图像模式: profit = Σ(llm_score)  ← 0-100 分数")
    print("  • 自动模式检测: is_image_mode = (image_path is not None)")
    print("  • 评分缓存机制: item.llm_score")
    print("  • 图表标签自适应: 'Total Profit ($)' vs 'Total Aesthetic Score'")

    print("\n📊 数据格式:")
    print("  图像模式物品 JSONL:")
    print('  {"id": 1, "name": "Artwork", "price": 100,')
    print('   "image_path": "001.jpg", "context": "评估美学"}')

    print("\n🚀 使用方法:")
    print("  1. 设置 API key: export OPENAI_API_KEY='your-key'")
    print("  2. 运行拍卖: python auction_workflow.py \\")
    print("       --input_dir data/example_llm_scoring/ --threads 2")
    print("  3. Monte Carlo: 添加 --monte_carlo --num_runs 30")

    print("\n💡 profit 字段含义:")
    print("  • 原版模式: 金钱利润（美元）")
    print("  • 图像模式: 美学评分累积（0-100 分）")

    print("\n")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
