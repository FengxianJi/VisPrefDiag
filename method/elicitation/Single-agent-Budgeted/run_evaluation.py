#!/usr/bin/env python3
"""
Single-agent-Budgeted 评估脚本
支持多进程并行评估，按数据集划分结果

功能：
1. 加载 EmergingArt 数据集（Low-price, Mid-price, High-price, Overall）
2. 使用 deepseek-v3-chat 模型进行价格预测
3. 多进程并行加速评估（默认20进程）
4. 按数据集划分保存结果
5. 调用 calculate_matrix.py 计算评估指标
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback
from datetime import datetime

# 导入本地模块
from dataset import load_dataset
from sigelLLM import SigelLLMAuction


def process_single_scene(args_tuple: Tuple) -> Dict:
    """
    处理单个 scene（一个拍卖场景）

    Args:
        args_tuple: (api_key, api_base, model_name, temperature, episode, scene_id)

    Returns:
        包含预测结果的字典
    """
    api_key, api_base, model_name, temperature, episode, scene_id = args_tuple

    try:
        # 创建 agent（每个进程独立创建）
        agent = SigelLLMAuction(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            temperature=temperature
        )

        # 重置 agent
        agent.reset()

        budget = episode['budget']
        items = episode['items']
        num_items = len(items)

        # 步骤1: 创建 contact sheet
        all_first_images = [item['images'][0] for item in items if item['images']]
        contact_sheet = agent.create_contact_sheet(
            all_first_images,
            thumbnail_size=(300, 300)
        )

        # 步骤2: 获取偏好排序
        print(f"[Scene {scene_id}] 获取偏好排序...")
        preference_result = agent.get_preference_ranking(contact_sheet, budget, num_items)
        print(f"[Scene {scene_id}] 偏好排序获取成功")

        # 步骤3: 逐个 item 获取价格预测
        remaining_budget = budget
        predictions = []

        for idx, item in enumerate(items, 1):
            print(f"[Scene {scene_id}] 预测 Item {idx}/{num_items}...")
            # 获取价格预测
            prediction = agent.predict_price(
                item_images=item['images'],
                item_index=idx,
                budget=budget,
                remaining_budget=remaining_budget,
                num_remaining_items=num_items - idx + 1
            )

            bid_price = prediction['bid_price']
            actual_price = item['price']

            # 更新剩余预算
            remaining_budget -= bid_price

            # 获取偏好排名（如果有的话）
            preference_rank = -1
            if 'preference_ranking' in preference_result and preference_result['preference_ranking']:
                try:
                    preference_rank = preference_result['preference_ranking'].index(idx) + 1
                except (ValueError, TypeError):
                    preference_rank = -1

            predictions.append({
                'scene': scene_id,
                'folder_id': item['folder_id'],
                'actual_price': actual_price,
                'predicted_price': bid_price,
                'reasoning': prediction.get('reasoning', ''),
                'preference_rank': preference_rank
            })

        print(f"✓ Completed scene {scene_id} (Group {episode['group_id']})")

        return {
            'success': True,
            'scene_id': scene_id,
            'group_id': episode['group_id'],
            'predictions': predictions,
            'preference_result': preference_result
        }

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"✗ Error processing scene {scene_id}: {e}")
        print(error_msg)
        return {
            'success': False,
            'scene_id': scene_id,
            'error': str(e),
            'traceback': error_msg
        }


def evaluate_dataset(
    json_path: str,
    dataset_name: str,
    api_key: str,
    api_base: str,
    model_name: str,
    temperature: float,
    budget_ratio: float,
    num_processes: int,
    seed: int = 42
) -> Dict:
    """
    评估单个数据集

    Args:
        json_path: 数据集JSON文件路径
        dataset_name: 数据集名称（Low-price, Mid-price, High-price, Overall）
        api_key: API密钥
        api_base: API基础URL
        model_name: 模型名称
        temperature: 温度参数
        budget_ratio: 预算比例
        num_processes: 进程数量
        seed: 随机种子

    Returns:
        评估结果字典
    """
    print("=" * 80)
    print(f"评估数据集: {dataset_name}")
    print("=" * 80)

    # 加载数据集
    print(f"加载数据集: {json_path}")
    dataset = load_dataset(
        json_path=json_path,
        budget_ratio=budget_ratio,
        seed=seed
    )

    # 显示统计信息
    stats = dataset.get_statistics()
    print(f"  总items数: {stats['total_items']}")
    print(f"  Groups数: {stats['num_groups']}")
    print(f"  价格范围: €{stats['min_price']:.2f} - €{stats['max_price']:.2f}")
    print(f"  平均价格: €{stats['avg_price']:.2f}")
    print(f"  总价格: €{stats['total_price']:.2f}")
    print(f"  预算: €{stats['default_budget']:.2f}")

    # 创建所有 episodes
    print(f"\n创建所有 episodes...")
    episodes = dataset.create_all_episodes(
        shuffle_groups=False,
        shuffle_items=False,
        shuffle_images=False
    )
    print(f"  创建了 {len(episodes)} 个 episodes")

    # 准备多进程参数
    print(f"\n使用 {num_processes} 个进程进行并行评估...")
    args_list = []
    for scene_id, episode in enumerate(episodes):
        args_list.append((
            api_key,
            api_base,
            model_name,
            temperature,
            episode,
            scene_id
        ))

    # 多进程并行处理
    all_predictions = []
    failed_scenes = []

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_scene, args_list)

    # 收集结果
    for result in results:
        if result['success']:
            all_predictions.extend(result['predictions'])
        else:
            failed_scenes.append(result)

    print(f"\n评估完成:")
    print(f"  成功: {len(episodes) - len(failed_scenes)}/{len(episodes)} scenes")
    print(f"  失败: {len(failed_scenes)} scenes")
    print(f"  总预测数: {len(all_predictions)}")

    # 构建结果
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'budget_ratio': budget_ratio,
        'num_scenes': len(episodes),
        'num_predictions': len(all_predictions),
        'predictions': all_predictions,
        'failed_scenes': failed_scenes,
        'stats': stats
    }

    return result


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """
    根据预测结果计算评估指标（使用 calculate_matrix.py 的逻辑）

    Args:
        predictions: 预测结果列表

    Returns:
        评估指标字典
    """
    import numpy as np
    from scipy.stats import spearmanr
    from collections import defaultdict

    # 按 scene 分组
    episodes = defaultdict(list)
    for pred in predictions:
        scene_id = pred['scene']
        episodes[scene_id].append(pred)

    # 构建 episodes_data 格式
    episodes_data = []
    for scene_id in sorted(episodes.keys()):
        items_in_scene = episodes[scene_id]

        items = []
        for pred in items_in_scene:
            folder_id = pred['folder_id']
            true_price = pred['actual_price']
            predicted_price = pred['predicted_price']
            rel = np.log(1 + true_price)  # rel = log(1 + y)

            items.append({
                'name': f"Artwork #{folder_id}",
                'true_price': true_price,
                'rel': rel,
                'pred_score': predicted_price
            })

        episodes_data.append({
            'scene': scene_id,
            'items': items
        })

    # 计算 Acc (Top-1 Accuracy)
    def calculate_acc(episodes_data):
        correct = 0
        total = len(episodes_data)
        for episode in episodes_data:
            true_ranking = sorted(episode['items'], key=lambda x: x['rel'], reverse=True)
            pred_ranking = sorted(episode['items'], key=lambda x: x['pred_score'], reverse=True)
            if true_ranking[0]['name'] == pred_ranking[0]['name']:
                correct += 1
        return correct / total if total > 0 else 0.0

    # 计算 NDCG@k
    def calculate_dcg_at_k(items, ranking, k):
        dcg = 0.0
        rel_map = {item['name']: item['rel'] for item in items}
        for r in range(min(k, len(ranking))):
            item_name = ranking[r]
            rel = rel_map[item_name]
            dcg += (2**rel - 1) / np.log2(r + 2)
        return dcg

    def calculate_ndcg_at_k(episodes_data, k):
        ndcg_scores = []
        for episode in episodes_data:
            items = episode['items']
            pred_ranking = [item['name'] for item in sorted(items, key=lambda x: x['pred_score'], reverse=True)]
            ideal_ranking = [item['name'] for item in sorted(items, key=lambda x: x['rel'], reverse=True)]
            dcg = calculate_dcg_at_k(items, pred_ranking, k)
            idcg = calculate_dcg_at_k(items, ideal_ranking, k)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    # 计算 Spearman's ρ
    def calculate_spearman_rho(episodes_data):
        rho_scores = []
        for episode in episodes_data:
            items = episode['items']
            if len(items) < 2:
                continue
            true_rels = [item['rel'] for item in items]
            pred_scores = [item['pred_score'] for item in items]
            rho, _ = spearmanr(true_rels, pred_scores)
            if not np.isnan(rho):
                rho_scores.append(rho)
        return np.mean(rho_scores) if rho_scores else 0.0

    # 计算 sMAPE
    def calculate_smape(episodes_data):
        errors = []
        for episode in episodes_data:
            for item in episode['items']:
                y_true = item['true_price']
                y_pred = item['pred_score']
                epsilon = 1e-10
                smape_value = 2 * abs(y_pred - y_true) / (abs(y_pred) + abs(y_true) + epsilon)
                errors.append(smape_value)
        return np.mean(errors) if errors else 0.0

    # 计算所有指标
    acc = calculate_acc(episodes_data)
    ndcg1 = calculate_ndcg_at_k(episodes_data, k=1)
    ndcg3 = calculate_ndcg_at_k(episodes_data, k=3)
    spearman = calculate_spearman_rho(episodes_data)
    smape = calculate_smape(episodes_data)

    print(f"\n评估指标:")
    print(f"  Acc: {acc:.4f}")
    print(f"  NDCG@1: {ndcg1:.4f}")
    print(f"  NDCG@3: {ndcg3:.4f}")
    print(f"  Spearman's ρ: {spearman:.4f}")
    print(f"  sMAPE: {smape:.4f}")

    return {
        'overall': {
            'Acc': round(acc, 4),
            'NDCG@1': round(ndcg1, 4),
            'NDCG@3': round(ndcg3, 4),
            'Spearman_rho': round(spearman, 4)
        },
        'price_error': {
            'sMAPE': round(smape, 4)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Single-agent-Budgeted 评估脚本')

    # API 配置
    parser.add_argument('--api_key', type=str, default=None,
                       help='API密钥（默认从环境变量读取）')
    parser.add_argument('--api_base', type=str,
                       default='https://api.uniapi.io/v1',
                       help='API基础URL')
    parser.add_argument('--model_name', type=str,
                       default='google/gemini-3-flash-preview',
                       help='模型名称（默认: google/gemini-3-flash-preview）')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='温度参数（默认: 0.7）')

    # 数据集配置
    parser.add_argument('--data_dir', type=str,
                       default='/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data',
                       help='数据集目录')
    parser.add_argument('--budget_ratio', type=float, default=1.2,
                       help='预算比例（默认: 1.2）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')

    # 运行配置
    parser.add_argument('--num_processes', type=int, default=20,
                       help='并行进程数（默认: 20）')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['Low-price', 'Mid-price', 'High-price', 'Overall'],
                       help='要评估的数据集')

    # 输出配置
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main/mini Main/test@1',
                       help='输出目录')

    args = parser.parse_args()

    # 获取 API key
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("错误: 未设置 API key")
        print("请使用 --api_key 参数或设置环境变量 OPENROUTER_API_KEY")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Single-agent-Budgeted 评估")
    print("=" * 80)
    print(f"模型: {args.model_name}")
    print(f"预算比例: {args.budget_ratio}")
    print(f"并行进程数: {args.num_processes}")
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)
    print()

    # 记录开始时间
    start_time = datetime.now()

    # 评估每个数据集
    all_results = {}
    for dataset_name in args.datasets:
        json_path = Path(args.data_dir) / f"{dataset_name}.json"

        if not json_path.exists():
            print(f"警告: 数据集文件不存在: {json_path}")
            continue

        # 评估数据集
        result = evaluate_dataset(
            json_path=str(json_path),
            dataset_name=dataset_name,
            api_key=api_key,
            api_base=args.api_base,
            model_name=args.model_name,
            temperature=args.temperature,
            budget_ratio=args.budget_ratio,
            num_processes=args.num_processes,
            seed=args.seed
        )

        # 计算指标
        print(f"\n计算 {dataset_name} 的评估指标...")
        metrics = calculate_metrics(result['predictions'])
        result['metrics'] = metrics

        all_results[dataset_name] = result

        print()

    # 保存结果
    print("=" * 80)
    print("保存结果...")
    print("=" * 80)

    # 保存 Single-agent-Budgeted.json（包含所有预测结果）
    result_file = output_dir / "Single-agent-Budgeted.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存预测结果: {result_file}")

    # 保存 Single-agent-Budgeted_matrix.json（只包含评估指标）
    matrix_file = output_dir / "Single-agent-Budgeted_matrix.json"
    matrix_results = {
        'method': 'Single-agent-Budgeted',
        'model': args.model_name,
        'budget_ratio': args.budget_ratio,
        'timestamp': start_time.isoformat(),
        'datasets': {}
    }

    for dataset_name, result in all_results.items():
        matrix_results['datasets'][dataset_name] = {
            'num_scenes': result['num_scenes'],
            'num_predictions': result['num_predictions'],
            'metrics': result['metrics']
        }

    with open(matrix_file, 'w', encoding='utf-8') as f:
        json.dump(matrix_results, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存评估指标: {matrix_file}")

    # 计算运行时间
    end_time = datetime.now()
    duration = end_time - start_time

    print()
    print("=" * 80)
    print("评估完成!")
    print("=" * 80)
    print(f"总运行时间: {duration}")
    print(f"结果保存在: {output_dir}")
    print()

    # 打印摘要
    print("评估摘要:")
    print("-" * 80)
    for dataset_name, result in all_results.items():
        metrics = result['metrics']['overall']
        print(f"{dataset_name}:")
        print(f"  Scenes: {result['num_scenes']}")
        print(f"  Predictions: {result['num_predictions']}")
        print(f"  Acc: {metrics['Acc']:.4f}")
        print(f"  NDCG@3: {metrics['NDCG@3']:.4f}")
        print(f"  Spearman ρ: {metrics['Spearman_rho']:.4f}")
        print()


if __name__ == '__main__':
    main()
