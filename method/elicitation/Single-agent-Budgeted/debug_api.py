#!/usr/bin/env python3
"""
测试API返回的实际内容
"""
import os
import sys

if not os.environ.get('UNIAPI_API_KEY'):
    raise RuntimeError('Set UNIAPI_API_KEY before running this script')

from sigelLLM import SigelLLMAuction
from dataset import load_dataset
from PIL import Image

print("测试API返回内容")
print("=" * 60)

# 加载数据
dataset = load_dataset(
    json_path='/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data/Low-price.json',
    budget_ratio=1.2,
    seed=42
)

episode = dataset.create_episode(group_id=0, shuffle_items=False, shuffle_images=False)

# 创建agent
agent = SigelLLMAuction(
    api_key=os.environ['UNIAPI_API_KEY'],
    api_base='https://api.uniapi.io/v1',
    model_name='deepseek-chat',
    temperature=0.7
)

# 测试单个图片的价格预测
print("\n测试单个item的价格预测...")
item = episode['items'][0]
print(f"Item folder: {item['folder_id']}")
print(f"实际价格: €{item['price']:.2f}")
print(f"图片数量: {len(item['images'])}")

print("\n发送API请求...")
try:
    prediction = agent.predict_price(
        item_images=item['images'],
        item_index=1,
        budget=episode['budget'],
        remaining_budget=episode['budget'],
        num_remaining_items=6
    )

    print("\n收到响应:")
    print(f"  预测价格: €{prediction.get('bid_price', 'N/A')}")
    print(f"  理由: {prediction.get('reasoning', 'N/A')[:200]}")

    if prediction.get('bid_price', 0) == 0:
        print("\n⚠️ 预测价格为0，可能是API响应有问题")
        print("完整响应:")
        print(prediction)

except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
