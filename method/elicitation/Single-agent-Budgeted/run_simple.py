#!/usr/bin/env python3
"""
Simple 评估脚本 - 不使用多进程，方便调试
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime

if not os.environ.get('OPENROUTER_API_KEY'):
    raise RuntimeError('Set OPENROUTER_API_KEY before running this script')

# 导入本地模块
from dataset import load_dataset
from sigelLLM import SigelLLMAuction

print("=" * 80)
print("Single-agent-Budgeted 简化评估（无多进程）")
print("=" * 80)

# 加载数据集
print("\n加载Low-price数据集...")
dataset = load_dataset(
    json_path='/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data/Low-price.json',
    budget_ratio=1.2,
    seed=42
)

stats = dataset.get_statistics()
print(f"  总items数: {stats['total_items']}")
print(f"  Groups数: {stats['num_groups']}")

# 创建episodes
print("\n创建episodes...")
episodes = dataset.create_all_episodes(
    shuffle_groups=False,
    shuffle_items=False,
    shuffle_images=False
)
print(f"  创建了 {len(episodes)} 个episodes")

# 创建agent
print("\n创建LLM agent...")
agent = SigelLLMAuction(
    api_key=os.environ['OPENROUTER_API_KEY'],
    api_base='https://openrouter.ai/api/v1',
    model_name='google/gemini-3-flash-preview',
    temperature=0.7
)
print("✓ Agent创建成功")

# 处理每个episode
all_predictions = []
for scene_id, episode in enumerate(episodes):
    print(f"\n{'='*60}")
    print(f"处理Scene {scene_id} (Group {episode['group_id']})")
    print(f"{'='*60}")

    agent.reset()
    budget = episode['budget']
    items = episode['items']
    num_items = len(items)

    print(f"  Budget: €{budget:.2f}, Items: {num_items}")

    # 步骤1: 创建contact sheet
    print("\n  步骤1: 创建contact sheet...")
    all_first_images = [item['images'][0] for item in items if item['images']]
    try:
        contact_sheet = agent.create_contact_sheet(
            all_first_images,
            thumbnail_size=(300, 300)
        )
        print("  ✓ Contact sheet创建成功")
    except Exception as e:
        print(f"  ✗ 创建失败: {e}")
        continue

    # 步骤2: 获取偏好排序
    print("\n  步骤2: 获取偏好排序（可能需要10-60秒）...")
    try:
        preference_result = agent.get_preference_ranking(contact_sheet, budget, num_items)
        print(f"  ✓ 偏好排序: {preference_result.get('preference_ranking', [])}")
    except Exception as e:
        print(f"  ✗ 获取失败: {e}")
        import traceback
        traceback.print_exc()
        continue

    # 步骤3: 逐个预测价格
    print("\n  步骤3: 预测每个item的价格...")
    remaining_budget = budget

    for idx, item in enumerate(items, 1):
        print(f"\n    Item {idx}/{num_items} (Folder {item['folder_id']})...")
        try:
            prediction = agent.predict_price(
                item_images=item['images'],
                item_index=idx,
                budget=budget,
                remaining_budget=remaining_budget,
                num_remaining_items=num_items - idx + 1
            )

            bid_price = prediction['bid_price']
            actual_price = item['price']

            print(f"      实际: €{actual_price:.2f}, 预测: €{bid_price:.2f}")

            remaining_budget -= bid_price

            all_predictions.append({
                'scene': scene_id,
                'folder_id': item['folder_id'],
                'actual_price': actual_price,
                'predicted_price': bid_price
            })

        except Exception as e:
            print(f"      ✗ 预测失败: {e}")
            import traceback
            traceback.print_exc()

print(f"\n{'='*80}")
print(f"评估完成! 共 {len(all_predictions)} 个预测")
print(f"{'='*80}")

# 保存结果
output_file = "/tmp/simple_evaluation_result.json"
with open(output_file, 'w') as f:
    json.dump({
        'predictions': all_predictions,
        'total_predictions': len(all_predictions)
    }, f, indent=2)

print(f"\n结果已保存: {output_file}")
