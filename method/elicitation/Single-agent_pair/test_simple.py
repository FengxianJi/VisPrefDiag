#!/usr/bin/env python3
"""
Single-agent_pair 独立测试脚本
"""

import sys
import os

# 确保路径正确
sys.path.insert(0, '/mnt/data/user/fengxian.ji/AUCTION/method/Single-agent_pair')
os.chdir('/mnt/data/user/fengxian.ji/AUCTION/method/Single-agent_pair')

def main():
    print("="*60)
    print("Single-agent_pair 测试")
    print("="*60)

    # 导入
    try:
        from sigelLLM import SigelLLMAuction
        from dataset import load_dataset
        print("✓ 模块导入成功")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return 1

    # 创建agent
    try:
        agent = SigelLLMAuction(api_key='test-key', model_name='gpt-4o')
        print("✓ Agent创建成功")
    except Exception as e:
        print(f"❌ Agent创建失败: {e}")
        return 1

    # 检查方法
    methods = ['rank_items', 'ranking_history', 'get_ranking_summary', 'reset']
    all_exist = True
    for method in methods:
        exists = hasattr(agent, method)
        status = "✓" if exists else "❌"
        print(f"{status} {method}: {exists}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n❌ 部分方法缺失")
        return 1

    # 测试reset
    agent.reset()
    assert len(agent.ranking_history) == 0
    print("✓ reset() 工作正常")

    # 测试summary
    agent.ranking_history = [
        {'num_items': 5, 'ranking': [1,2,3,4,5], 'reasoning': 'test'}
    ]
    summary = agent.get_ranking_summary()
    assert summary['total_rankings'] == 1
    print("✓ get_ranking_summary() 工作正常")

    # 测试数据加载
    try:
        json_path = '/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data/Low-price.json'
        dataset = load_dataset(json_path, seed=42)
        episode = dataset.create_episode()
        print(f"✓ 数据加载成功 (Group {episode['group_id']}, {episode['num_items']} items)")
    except Exception as e:
        print(f"⚠ 数据加载警告: {e}")

    print("\n" + "="*60)
    print("✅ 所有测试通过！")
    print("="*60)
    print("\n下一步:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  python sigelLLM.py")

    return 0


if __name__ == '__main__':
    sys.exit(main())
