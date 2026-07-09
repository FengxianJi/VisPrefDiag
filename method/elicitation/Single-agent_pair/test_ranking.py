#!/usr/bin/env python3
"""
测试排序功能（不需要API key）
验证代码逻辑的正确性
"""

import sys
import os
sys.path.append('/mnt/data/user/fengxian.ji/AUCTION/method/Single-agent_pair')

def test_ranking_logic():
    """测试排序逻辑（模拟数据）"""
    print("="*60)
    print("测试排序逻辑")
    print("="*60)

    from sigelLLM import SigelLLMAuction

    # 创建agent
    agent = SigelLLMAuction(
        api_key="dummy-key",  # 测试用，不会真正调用API
        model_name="gpt-4o",
        temperature=0.7
    )

    print(f"\n✓ Agent创建成功")
    print(f"  模型: {agent.model_name}")
    print(f"  温度: {agent.temperature}")

    # 检查方法存在性
    assert hasattr(agent, 'rank_items'), "缺少 rank_items 方法"
    assert hasattr(agent, 'ranking_history'), "缺少 ranking_history 属性"
    assert hasattr(agent, 'get_ranking_summary'), "缺少 get_ranking_summary 方法"

    print(f"\n✓ 必要的方法和属性都存在")
    print(f"  - rank_items()")
    print(f"  - ranking_history")
    print(f"  - get_ranking_summary()")

    # 测试reset
    agent.ranking_history = [{'test': 'data'}]
    agent.reset()
    assert len(agent.ranking_history) == 0, "reset() 未清空历史"

    print(f"\n✓ reset() 方法正常工作")

    # 测试get_ranking_summary
    summary = agent.get_ranking_summary()
    assert summary == {}, "空历史应返回空字典"

    print(f"\n✓ get_ranking_summary() 方法正常工作")

    print(f"\n{'='*60}")
    print("✅ 所有基础测试通过！")
    print("="*60)


def test_episode_structure():
    """测试episode数据结构"""
    print("\n" + "="*60)
    print("测试Episode数据结构")
    print("="*60)

    from dataset import load_dataset

    # 使用示例数据
    json_path = '/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data/Low-price.json'

    if not os.path.exists(json_path):
        print(f"\n⚠️  测试数据不存在: {json_path}")
        return

    dataset = load_dataset(json_path, budget_ratio=1.2, seed=42)
    episode = dataset.create_episode()

    print(f"\n✓ Episode创建成功")
    print(f"  Group ID: {episode['group_id']}")
    print(f"  Items数量: {episode['num_items']}")
    print(f"  总价格: €{episode['total_price']:.2f}")

    # 检查items结构
    items = episode['items']
    print(f"\n✓ Items结构检查:")
    for idx, item in enumerate(items[:3], 1):  # 只显示前3个
        print(f"  Item {idx}:")
        print(f"    folder_id: {item['folder_id']}")
        print(f"    price: €{item['price']:.2f}")
        print(f"    images数量: {item['num_images']}")

    print(f"\n{'='*60}")
    print("✅ Episode数据结构正常！")
    print("="*60)


def show_code_changes():
    """显示代码改动总结"""
    print("\n" + "="*60)
    print("代码改动总结")
    print("="*60)

    changes = [
        ("类说明", "预测价格 → 对图片排序"),
        ("decision_history", "→ ranking_history"),
        ("predict_price()", "→ rank_items()"),
        ("get_decision_summary()", "→ get_ranking_summary()"),
        ("_call_llm_api()", "→ _call_llm_api_multi_images()"),
        ("run_auction_episode()", "预测逻辑 → 排序逻辑")
    ]

    print("\n主要改动:")
    for old, new in changes:
        print(f"  • {old:30s} {new}")

    print("\n新增功能:")
    print("  • 支持多图片同时输入")
    print("  • 返回排序结果 (ranking)")
    print("  • 返回评分 (scores)")
    print("  • 计算排名与价格的相关性")

    print("\n输出格式:")
    print("  {")
    print("    'ranking': [2, 5, 1, 3, 4],  # 从最佳到最差")
    print("    'reasoning': '...',           # 排序理由")
    print("    'scores': [85, 92, 78, ...]   # 每个item的评分")
    print("  }")

    print(f"\n{'='*60}")


def main():
    print("\n" + "="*60)
    print(" Single-agent_pair 排序功能测试")
    print("="*60)

    try:
        # 测试1: 基础逻辑
        test_ranking_logic()

        # 测试2: Episode结构
        test_episode_structure()

        # 显示改动总结
        show_code_changes()

        print("\n" + "="*60)
        print("🎉 所有测试通过！")
        print("="*60)

        print("\n下一步:")
        print("  1. 设置API key: export OPENAI_API_KEY='your-key'")
        print("  2. 运行实际测试: python sigelLLM.py")
        print("  3. 查看排序结果和分析\n")

        return 0

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
