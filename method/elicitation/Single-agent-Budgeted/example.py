#!/usr/bin/env python3
"""
示例脚本：运行LLM拍卖agent

使用方法:
    python3 example.py --json_path <path> --api_key <key> [options]

示例:
    # 运行Low-price测试集
    python3 example.py --json_path ../../../benchmark/EmergingArt_test100/data/Low-price.json

    # 使用自定义预算比例
    python3 example.py --json_path ../../../benchmark/EmergingArt_test100/data/Overall.json --budget_ratio 1.5

    # 运行所有episodes
    python3 example.py --json_path ../../../benchmark/EmergingArt_test100/data/Overall.json --run_all
"""

import argparse
import os
import sys
from dataset import load_dataset
from sigelLLM import SigelLLMAuction, run_auction_episode


def main():
    parser = argparse.ArgumentParser(description='运行LLM拍卖agent')

    # 必需参数
    parser.add_argument('--json_path', type=str, required=True,
                       help='测试集JSON文件路径')

    # LLM配置
    parser.add_argument('--api_key', type=str, default=None,
                       help='OpenAI API key (默认从环境变量OPENAI_API_KEY读取)')
    parser.add_argument('--api_base', type=str, default=None,
                       help='API base URL')
    parser.add_argument('--model_name', type=str, default='gpt-4o',
                       help='模型名称 (默认: gpt-4o)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='温度参数 (默认: 0.7)')

    # Dataset配置
    parser.add_argument('--budget_ratio', type=float, default=1.2,
                       help='预算比例 (默认: 1.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')

    # 运行配置
    parser.add_argument('--group_id', type=int, default=None,
                       help='指定运行的group_id (默认: 随机选择)')
    parser.add_argument('--run_all', action='store_true',
                       help='运行所有groups')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录 (默认: ./output)')

    # 随机化配置
    parser.add_argument('--no_shuffle_items', action='store_true',
                       help='不随机打乱items顺序')
    parser.add_argument('--no_shuffle_images', action='store_true',
                       help='不随机打乱图片顺序')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.json_path):
        print(f"错误: 文件不存在: {args.json_path}")
        sys.exit(1)

    # 检查API key
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("警告: 未设置OPENAI_API_KEY，API调用可能会失败")
        print("请使用 --api_key 参数或设置环境变量: export OPENAI_API_KEY='your-key'")

    print("="*60)
    print("LLM拍卖Agent - 运行配置")
    print("="*60)
    print(f"数据集: {os.path.basename(args.json_path)}")
    print(f"模型: {args.model_name}")
    print(f"预算比例: {args.budget_ratio}")
    print(f"随机种子: {args.seed}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)

    # 加载数据集
    print("\n加载数据集...")
    dataset = load_dataset(
        json_path=args.json_path,
        budget_ratio=args.budget_ratio,
        seed=args.seed
    )

    # 显示统计信息
    stats = dataset.get_statistics()
    print(f"  总items数: {stats['total_items']}")
    print(f"  Groups数: {stats['num_groups']}")
    print(f"  价格范围: €{stats['min_price']:.2f} - €{stats['max_price']:.2f}")
    print(f"  平均价格: €{stats['avg_price']:.2f}")
    print(f"  总价格: €{stats['total_price']:.2f}")
    print(f"  默认Budget: €{stats['default_budget']:.2f}")

    # 创建LLM agent
    print("\n初始化LLM agent...")
    agent = SigelLLMAuction(
        api_key=api_key,
        api_base=args.api_base,
        model_name=args.model_name,
        temperature=args.temperature
    )

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行episodes
    if args.run_all:
        # 运行所有episodes
        print(f"\n运行所有{stats['num_groups']}个groups...")
        episodes = dataset.create_all_episodes(
            shuffle_items=not args.no_shuffle_items,
            shuffle_images=not args.no_shuffle_images
        )

        results = []
        for ep in episodes:
            result = run_auction_episode(agent, ep, args.output_dir)
            results.append(result)

        # 总体统计
        print("\n"+"="*60)
        print("所有Episodes统计:")
        print("="*60)

        total_budget = sum(r['episode_info']['budget'] for r in results)
        total_spent = sum(r['summary']['total_spent'] for r in results)
        total_actual = sum(r['episode_info']['total_price'] for r in results)

        print(f"总Episodes数: {len(results)}")
        print(f"总预算: €{total_budget:.2f}")
        print(f"总花费: €{total_spent:.2f}")
        print(f"实际总价: €{total_actual:.2f}")
        print(f"预算利用率: {(total_spent/total_budget)*100:.1f}%")
        print("="*60)

    else:
        # 运行单个episode
        episode = dataset.create_episode(
            group_id=args.group_id,
            shuffle_items=not args.no_shuffle_items,
            shuffle_images=not args.no_shuffle_images
        )

        result = run_auction_episode(agent, episode, args.output_dir)

    print("\n完成!")
    print(f"结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
