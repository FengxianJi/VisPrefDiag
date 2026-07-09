#!/usr/bin/python3
"""
Single-agent_pair ranking (Qwen3-VL-4B local).
Sends all 6 images per group and ranks them best-to-worst.
Replaces API-based run_Single-agent_pair.py.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time

INTERFACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTERFACE_DIR))
sys.path.insert(0, '/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main')

from sigelLLM import SigelLLMAuction
from dataset import AuctionDataset
import calculate_matrix

DATA_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data')
RESULT_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-4B/result/Single-agent_pair')

DEFAULT_TEMPERATURE = 0.0
DEFAULT_BUDGET_RATIO = 1.2


def evaluate_single_group(agent, json_file, group_id, budget_ratio, seed):
    try:
        dataset = AuctionDataset(json_path=str(json_file), budget_ratio=budget_ratio, seed=seed)
        episode = dataset.create_episode(
            group_id=group_id, shuffle_groups=False,
            shuffle_items=False, shuffle_images=False
        )

        items = episode['items']
        if not items:
            return {'group_id': group_id, 'items': [], 'success': False, 'error': 'No items'}

        # Prepare items for ranking
        all_items = [{'images': item['images'], 'folder_id': item['folder_id']} for item in items]

        # Rank all items in one call
        agent.reset()
        ranking_result = agent.rank_items(all_items)
        ranking = ranking_result['ranking']
        reasoning = ranking_result.get('reasoning', '')

        # Convert ranking to per-item results
        items_result = []
        for idx, item in enumerate(items):
            item_rank = ranking.index(idx + 1) + 1 if (idx + 1) in ranking else len(items)
            items_result.append({
                'name': item['folder_id'],
                'folder_id': item['folder_id'],
                'true_price': item['price'],
                'pred_score': item_rank,
                'rank': item_rank,
                'rel': calculate_matrix.calculate_rel(item['price']),
                'reasoning': f"Rank {item_rank}/{len(items)} | {reasoning[:100]}"
            })

        return {
            'group_id': group_id,
            'items': items_result,
            'ranking': ranking,
            'full_reasoning': reasoning,
            'success': True
        }

    except Exception as e:
        print(f"Error evaluating group {group_id}: {e}")
        import traceback; traceback.print_exc()
        return {'group_id': group_id, 'items': [], 'success': False, 'error': str(e)}


def evaluate_model(agent, temperature, output_dir, budget_ratio, seed):
    print(f"\n{'='*80}")
    print(f"Qwen3-VL-4B Single-agent_pair (Ranking)")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'model_name': 'qwen3-vl-4b-instruct',
        'method': 'Single-agent_pair',
        'temperature': temperature,
        'budget_ratio': budget_ratio,
        'datasets': {}
    }

    dataset_files = [
        ('Low-price', DATA_DIR / 'Low-price.json'),
        ('Mid-price', DATA_DIR / 'Mid-price.json'),
        ('High-price', DATA_DIR / 'High-price.json'),
        ('Near-Tie-low', DATA_DIR / 'Near-Tie-low.json'),
        ('Near-Tie-high', DATA_DIR / 'Near-Tie-high.json'),
        ('Overall', DATA_DIR / 'Overall.json')
    ]

    for dataset_name, json_file in dataset_files:
        if not json_file.exists():
            print(f"Warning: {json_file} not found, skipping")
            continue

        print(f"\nEvaluating on {json_file.name}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        group_ids = sorted(set(item['group_id'] for item in data))
        print(f"  {len(group_ids)} groups (sequential)")

        results = []
        for i, gid in enumerate(group_ids):
            print(f"  [{i+1}/{len(group_ids)}] Group {gid}...", end=" ", flush=True)
            r = evaluate_single_group(agent, json_file, gid, budget_ratio, seed)
            results.append(r)
            if r['success']:
                print(f"OK  ranking={r.get('ranking', [])}")
            else:
                print(f"FAIL: {r.get('error','')}")

            # Incremental save after each group
            all_results['datasets'][dataset_name] = [x for x in results if x['success']]
            result_file = output_dir / "Single-agent_pair.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        success = [r for r in results if r['success']]
        print(f"  Done: {len(success)}/{len(group_ids)} groups succeeded")
        all_results['datasets'][dataset_name] = success

    result_file = output_dir / "Single-agent_pair.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {result_file}")

    metrics_result = {}
    for ds_name in all_results['datasets']:
        episodes = all_results['datasets'][ds_name]
        if episodes:
            print(f"  {ds_name}:")
            metrics_result[ds_name] = calculate_matrix.calculate_all_metrics(episodes)

    matrix_file = output_dir / "Single-agent_pair_matrix.json"
    with open(matrix_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_result, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved: {matrix_file}")

    return all_results, metrics_result


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-4B Single-agent_pair Ranking')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--budget_ratio', type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    agent = SigelLLMAuction(temperature=args.temperature)

    start_time = time.time()
    evaluate_model(agent, args.temperature, RESULT_DIR, args.budget_ratio, args.seed)
    print(f"\nTotal time: {(time.time()-start_time)/60:.2f} min")


if __name__ == '__main__':
    main()
