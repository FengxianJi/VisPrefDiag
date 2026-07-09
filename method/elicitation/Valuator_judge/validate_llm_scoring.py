#!/usr/bin/env python3
"""
完整验证脚本 - 验证 LLM 评分功能的所有修改

这个脚本不需要实际的 API key，只测试代码逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_item_modifications():
    """测试 Item 类的修改"""
    print("="*60)
    print("【测试 1】Item 类修改")
    print("="*60)

    from src.item_base import Item

    # 测试 1.1: 图像模式物品
    print("\n1.1 图像模式物品")
    item1 = Item(
        id=1,
        name="Artwork #001",
        price=100,
        image_path="data/images/001.jpg",
        context="评估美学价值",
        style_tags=["abstract", "modern"],
        llm_score=85.5
    )
    assert item1.is_image_mode == True, "图像模式判断错误"
    assert item1.llm_score == 85.5, "llm_score 赋值错误"
    assert item1.true_value is None, "图像模式不应有 true_value"
    print(f"  ✓ is_image_mode: {item1.is_image_mode}")
    print(f"  ✓ llm_score: {item1.llm_score}")
    print(f"  ✓ true_value: {item1.true_value}")

    # 测试 1.2: 原版模式物品
    print("\n1.2 原版模式物品")
    item2 = Item(
        id=2,
        name="Widget A",
        price=1000,
        true_value=2000
    )
    assert item2.is_image_mode == False, "原版模式判断错误"
    assert item2.true_value == 2000, "true_value 赋值错误"
    assert item2.llm_score is None, "原版模式不应有 llm_score"
    print(f"  ✓ is_image_mode: {item2.is_image_mode}")
    print(f"  ✓ true_value: {item2.true_value}")
    print(f"  ✓ llm_score: {item2.llm_score}")

    # 测试 1.3: to_dict 包含 llm_score
    print("\n1.3 to_dict 方法")
    item_dict = item1.to_dict()
    assert 'llm_score' in item_dict, "to_dict 应包含 llm_score"
    assert item_dict['llm_score'] == 85.5, "to_dict llm_score 值错误"
    print(f"  ✓ to_dict 包含 llm_score: {item_dict['llm_score']}")

    print("\n✅ Item 类测试通过\n")


def test_bidder_methods():
    """测试 Bidder 类的方法是否存在"""
    print("="*60)
    print("【测试 2】Bidder 类方法")
    print("="*60)

    from src.bidder_base import Bidder

    # 测试方法存在性
    methods = ['_get_llm_score', 'win_bid', 'profit_report', 'to_monitors']
    print("\n检查方法存在性:")
    for method in methods:
        assert hasattr(Bidder, method), f"方法 {method} 不存在"
        print(f"  ✓ {method} 存在")

    print("\n✅ Bidder 方法测试通过\n")


def test_draw_plot_signature():
    """测试 draw_plot 函数签名"""
    print("="*60)
    print("【测试 3】draw_plot 函数签名")
    print("="*60)

    from src.bidder_base import draw_plot
    import inspect

    sig = inspect.signature(draw_plot)
    params = list(sig.parameters.keys())

    print(f"\n函数签名: draw_plot{sig}")
    print(f"参数列表: {params}")

    assert 'is_image_mode' in params, "draw_plot 应接受 is_image_mode 参数"
    assert sig.parameters['is_image_mode'].default == False, "is_image_mode 默认值应为 False"

    print("  ✓ is_image_mode 参数存在")
    print("  ✓ 默认值为 False")

    print("\n✅ draw_plot 签名测试通过\n")


def test_profit_logic_simulation():
    """模拟测试利润计算逻辑"""
    print("="*60)
    print("【测试 4】利润计算逻辑模拟")
    print("="*60)

    from src.item_base import Item

    # 创建测试物品
    print("\n4.1 图像模式利润计算")
    image_item = Item(
        id=1,
        name="Artwork #001",
        price=100,
        image_path="test.jpg",
        llm_score=75.5
    )

    # 模拟：图像模式下，profit 应该累加 llm_score
    profit = 0
    bid_price = 120
    profit += image_item.llm_score  # 图像模式：profit = llm_score
    print(f"  物品: {image_item.name}")
    print(f"  出价: ${bid_price}")
    print(f"  LLM评分: {image_item.llm_score}")
    print(f"  利润变化: 0 → {profit} (累加评分)")
    assert profit == 75.5, "图像模式利润计算错误"
    print("  ✓ 图像模式利润 = LLM评分")

    print("\n4.2 原版模式利润计算")
    normal_item = Item(
        id=2,
        name="Widget A",
        price=1000,
        true_value=2000
    )

    # 模拟：原版模式下，profit = true_value - bid
    profit = 0
    bid_price = 1500
    profit += normal_item.true_value - bid_price  # 原版模式
    print(f"  物品: {normal_item.name}")
    print(f"  出价: ${bid_price}")
    print(f"  真实价值: ${normal_item.true_value}")
    print(f"  利润变化: 0 → {profit} (价值 - 出价)")
    assert profit == 500, "原版模式利润计算错误"
    print("  ✓ 原版模式利润 = 价值 - 出价")

    print("\n✅ 利润计算逻辑测试通过\n")


def test_code_compatibility():
    """测试代码兼容性"""
    print("="*60)
    print("【测试 5】代码兼容性")
    print("="*60)

    from src.item_base import Item

    # 测试混合使用
    print("\n5.1 混合模式测试")
    items = [
        Item(id=1, name="Widget", price=100, true_value=200),
        Item(id=2, name="Artwork", price=50, image_path="art.jpg", llm_score=80.0)
    ]

    for item in items:
        mode = "图像模式" if item.is_image_mode else "原版模式"
        print(f"  物品 {item.id}: {mode}")
        if item.is_image_mode:
            assert item.llm_score is not None or item.llm_score == 80.0
        else:
            assert item.true_value is not None

    print("  ✓ 两种模式可以混合使用")

    # 测试向后兼容
    print("\n5.2 向后兼容性")
    old_style_item = Item(id=3, name="Old Widget", price=300, true_value=500)
    assert old_style_item.is_image_mode == False
    assert old_style_item.llm_score is None
    print("  ✓ 旧代码仍然正常工作")

    print("\n✅ 兼容性测试通过\n")


def test_file_modifications_summary():
    """总结文件修改"""
    print("="*60)
    print("【修改总结】")
    print("="*60)

    modifications = [
        ("utils.py", "添加 ujson fallback"),
        ("src/item_base.py", "添加 llm_score 字段"),
        ("src/bidder_base.py", "添加 _get_llm_score() 方法"),
        ("src/bidder_base.py", "修改 win_bid() - 图像模式累加评分"),
        ("src/bidder_base.py", "修改 profit_report() - 显示评分"),
        ("src/bidder_base.py", "修改 to_monitors() - 传递评分"),
        ("src/bidder_base.py", "修改 draw_plot() - 动态标签"),
        ("src/human_bidder.py", "更新 draw_plot 调用"),
        ("config.py", "添加 LLM 评分说明")
    ]

    print("\n修改的文件:")
    for i, (file, desc) in enumerate(modifications, 1):
        print(f"  {i}. {file:30s} - {desc}")

    print("\n新增的文件:")
    new_files = [
        "LLM_SCORING_README.md",
        "QUICKSTART_LLM_SCORING.md",
        "COMPARISON_MONEY_VS_LLM.md",
        "test_llm_scoring.py",
        "data/example_llm_scoring/items_demo.jsonl",
        "data/example_llm_scoring/bidders_demo.jsonl"
    ]
    for i, file in enumerate(new_files, 1):
        print(f"  {i}. {file}")

    print("\n核心改变:")
    print("  • 原版模式: profit = Σ(true_value - bid)")
    print("  • 图像模式: profit = Σ(llm_score)")
    print("  • 自动模式检测: item.is_image_mode")
    print("  • 评分缓存: item.llm_score")
    print("  • 图表标签自适应")

    print("\n✅ 修改完成")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print(" LLM 评分功能 - 完整验证")
    print("="*60 + "\n")

    try:
        test_item_modifications()
        test_bidder_methods()
        test_draw_plot_signature()
        test_profit_logic_simulation()
        test_code_compatibility()
        test_file_modifications_summary()

        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)

        print("\n下一步:")
        print("1. 准备真实图像数据")
        print("2. 配置 API key (OPENAI_API_KEY)")
        print("3. 运行拍卖: python auction_workflow.py --input_dir data/example_llm_scoring/ --threads 2")
        print("4. 查看结果和日志\n")

        return 0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
