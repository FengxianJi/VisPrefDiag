#!/usr/bin/python3
"""
Single-agent-Budgeted (Qwen3-VL-4B local).
Two-phase: contact sheet ranking + sequential budgeted pricing.
Replaces API-based run_Single-agent-Budgeted.py.

Supports parallel processing with multiple workers (each loads its own model).
"""

import os
import sys
import json
import argparse
from pathlib import Path
import time
from multiprocessing import Pool, set_start_method

INTERFACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTERFACE_DIR))
sys.path.insert(0, '/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main')

import calculate_matrix

DATA_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data')
RESULT_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-4B/result/Single-agent-Budgeted')

DEFAULT_TEMPERATURE = 0.0
DEFAULT_BUDGET_RATIO = 1.2


def evaluate_single_group(args):
    """Evaluate a single group (can be called from multiprocessing)."""
    json_path, group_id, budget_ratio, seed, temperature = args

    # Import inside worker process (each process loads its own model)
    sys.path.insert(0, str(INTERFACE_DIR))
    sys.path.insert(0, '/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main')
    from sigelLLM import SigelLLMAuction
    from dataset import load_dataset

    try:
        agent = SigelLLMAuction(temperature=temperature)
        dataset = load_dataset(json_path=str(json_path), budget_ratio=budget_ratio, seed=seed)
        episode = dataset.create_episode(
            group_id=group_id, shuffle_groups=False,
            shuffle_items=False, shuffle_images=False
        )

        agent.reset()
        budget = episode['budget']
        items = episode['items']
        num_items = len(items)

        if not items:
            return {'group_id': group_id, 'items': [], 'success': False, 'error': 'No items'}

        # Phase 1: contact sheet + preference ranking
        all_images = [img for item in items for img in item['images']]
        contact_sheet = agent.create_contact_sheet(all_images, thumbnail_size=(300, 300))
        preference_result = agent.get_preference_ranking(contact_sheet, budget, num_items)

        # Phase 2: sequential budgeted pricing
        remaining_budget = budget
        items_result = []

        for idx, item in enumerate(items, 1):
            prediction = agent.predict_price(
                item_images=item['images'],
                item_index=idx,
                budget=budget,
                remaining_budget=remaining_budget,
                num_remaining_items=num_items - idx + 1
            )
            bid_price = prediction['bid_price']
            remaining_budget -= bid_price

            items_result.append({
                'name': item['folder_id'],
                'folder_id': item['folder_id'],
                'true_price': item['price'],
                'pred_score': bid_price,
                'rel': calculate_matrix.calculate_rel(item['price']),
                'reasoning': prediction.get('reasoning', '')
            })

        # Clear KV cache
        from model_loader import get_qwen_model
        model = get_qwen_model()
        model.clear_cache()

        return {'group_id': group_id, 'items': items_result, 'success': True}

    except Exception as e:
        print(f"Error evaluating group {group_id}: {e}")
        import traceback; traceback.print_exc()
        return {'group_id': group_id, 'items': [], 'success': False, 'error': str(e)}


def evaluate_model(temperature, output_dir, budget_ratio, seed, num_workers=3):
    print(f"\n{'='*80}")
    print(f"Qwen3-VL-4B Single-agent-Budgeted")
    print(f"PARALLEL MODE: {num_workers} workers")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'model_name': 'qwen3-vl-4b-instruct',
        'method': 'Single-agent-Budgeted',
        'temperature': temperature,
        'budget_ratio': budget_ratio,
        'num_workers': num_workers,
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
        print(f"  {len(group_ids)} groups (parallel with {num_workers} workers)")

        args_list = [(json_file, gid, budget_ratio, seed, temperature) for gid in group_ids]

        results = []
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                for i, r in enumerate(pool.imap(evaluate_single_group, args_list)):
                    gid = group_ids[i]
                    status = "OK" if r['success'] else f"FAIL: {r.get('error','')}"
                    print(f"  [{i+1}/{len(group_ids)}] Group {gid}... {status}")
                    results.append(r)

                    # Incremental save
                    all_results['datasets'][dataset_name] = [x for x in results if x['success']]
                    result_file = output_dir / "Single-agent-Budgeted.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
        else:
            for i, gid in enumerate(group_ids):
                print(f"  [{i+1}/{len(group_ids)}] Group {gid}...", end=" ", flush=True)
                r = evaluate_single_group(args_list[i])
                results.append(r)
                status = "OK" if r['success'] else f"FAIL: {r.get('error','')}"
                print(status)

                # Incremental save
                all_results['datasets'][dataset_name] = [x for x in results if x['success']]
                result_file = output_dir / "Single-agent-Budgeted.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

        success = [r for r in results if r['success']]
        print(f"  Done: {len(success)}/{len(group_ids)} groups succeeded")
        all_results['datasets'][dataset_name] = success

    result_file = output_dir / "Single-agent-Budgeted.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {result_file}")

    metrics_result = {}
    for ds_name in all_results['datasets']:
        episodes = all_results['datasets'][ds_name]
        if episodes:
            print(f"  {ds_name}:")
            metrics_result[ds_name] = calculate_matrix.calculate_all_metrics(episodes)

    matrix_file = output_dir / "Single-agent-Budgeted_matrix.json"
    with open(matrix_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_result, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved: {matrix_file}")

    return all_results, metrics_result


def main():
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Qwen3-VL-4B Single-agent-Budgeted')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--budget_ratio', type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=3, help='Number of parallel workers (default: 3)')
    args = parser.parse_args()

    start_time = time.time()
    evaluate_model(args.temperature, RESULT_DIR, args.budget_ratio, args.seed, num_workers=args.num_workers)
    print(f"\nTotal time: {(time.time()-start_time)/60:.2f} min")


if __name__ == '__main__':
    main()
