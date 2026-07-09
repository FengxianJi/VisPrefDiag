#!/usr/bin/python3
"""
Single-agent-Budgeted (Qwen3-VL-8B-Instruct local).
Two-phase: contact sheet ranking + sequential budgeted pricing.
Dual-GPU: launch two instances (--gpu_id 0 and --gpu_id 1).
"""

import os
import sys
import json
import argparse
import fcntl
import time
from pathlib import Path
from datetime import datetime
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

INTERFACE_DIR = Path(__file__).resolve().parent
METHOD_DIR = INTERFACE_DIR.parent
DATA_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data')
RESULT_DIR = METHOD_DIR / 'result' / 'Single-agent-Budgeted'
EXPERIMENT_DIR = '/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main'

DEFAULT_TEMPERATURE = 0.0
DEFAULT_BUDGET_RATIO = 1.2


def log(msg, gpu_id=None):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]" if gpu_id is None else f"[{ts}][GPU{gpu_id}]"
    print(f"{prefix} {msg}", flush=True)


def get_group_ids(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return sorted(set(item['group_id'] for item in data))


def safe_save(result_file, all_results):
    lock_file = str(result_file) + '.lock'
    with open(lock_file, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        existing = json.load(f)
                    for ds_name, ds_results in existing.get('datasets', {}).items():
                        if ds_name not in all_results['datasets']:
                            all_results['datasets'][ds_name] = ds_results
                        else:
                            my_gids = {r['group_id'] for r in all_results['datasets'][ds_name]}
                            for r in ds_results:
                                if r['group_id'] not in my_gids:
                                    all_results['datasets'][ds_name].append(r)
                except (json.JSONDecodeError, KeyError):
                    pass
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def load_completed_keys(result_file):
    completed = set()
    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                prev = json.load(f)
            for ds_name, ds_results in prev.get('datasets', {}).items():
                for r in ds_results:
                    if r.get('success'):
                        completed.add((ds_name, r['group_id']))
        except Exception:
            pass
    return completed


def evaluate_single_group(agent, json_file, group_id, budget_ratio, seed, gpu_id):
    import calculate_matrix as cm
    from dataset import load_dataset

    try:
        dataset = load_dataset(json_path=str(json_file), budget_ratio=budget_ratio, seed=seed)
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
        log(f"    Phase1 pref={preference_result.get('preference_ranking', [])}", gpu_id)

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
                'rel': cm.calculate_rel(item['price']),
                'reasoning': prediction.get('reasoning', '')
            })

        # Clear cache
        from model_loader import get_qwen_model
        model = get_qwen_model()
        model.clear_cache()

        return {'group_id': group_id, 'items': items_result, 'success': True}

    except Exception as e:
        log(f"    ERROR: {e}", gpu_id)
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush()
        return {'group_id': group_id, 'items': [], 'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-8B Single-agent-Budgeted (Single GPU Worker)')
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--budget_ratio', type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    sys.path.insert(0, str(INTERFACE_DIR))
    sys.path.insert(0, EXPERIMENT_DIR)

    output_dir = RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "Single-agent-Budgeted.json"

    log(f"{'='*60}", gpu_id)
    log(f"Qwen3-VL-8B Single-agent-Budgeted Worker", gpu_id)
    log(f"  GPU={gpu_id}, budget_ratio={args.budget_ratio}", gpu_id)
    log(f"{'='*60}", gpu_id)

    dataset_files = [
        ('Low-price', DATA_DIR / 'Low-price.json'),
        ('Mid-price', DATA_DIR / 'Mid-price.json'),
        ('High-price', DATA_DIR / 'High-price.json'),
        ('Near-Tie-low', DATA_DIR / 'Near-Tie-low.json'),
        ('Near-Tie-high', DATA_DIR / 'Near-Tie-high.json'),
        ('Overall', DATA_DIR / 'Overall.json')
    ]

    all_tasks = []
    for dataset_name, json_file in dataset_files:
        if not json_file.exists():
            continue
        group_ids = get_group_ids(json_file)
        log(f"  {dataset_name}: {len(group_ids)} groups", gpu_id)
        for gid in group_ids:
            all_tasks.append((dataset_name, str(json_file), gid))

    my_tasks = [t for i, t in enumerate(all_tasks) if i % args.num_gpus == gpu_id]
    completed_keys = load_completed_keys(result_file)
    my_tasks = [(ds, jf, gid) for ds, jf, gid in my_tasks if (ds, gid) not in completed_keys]
    log(f"Total: {len(all_tasks)}, this worker: {len(my_tasks)} (after resume)", gpu_id)

    if not my_tasks:
        log("All assigned groups already completed!", gpu_id)
        return

    log("Loading model...", gpu_id)
    from sigelLLM import SigelLLMAuction
    agent = SigelLLMAuction(temperature=args.temperature)
    log("Model loaded", gpu_id)

    all_results = {
        'model_name': 'qwen3-vl-8b-instruct',
        'method': 'Single-agent-Budgeted',
        'temperature': args.temperature,
        'budget_ratio': args.budget_ratio,
        'datasets': {}
    }

    start_time = time.time()
    completed = 0
    success = 0

    for task_idx, (dataset_name, json_file, group_id) in enumerate(my_tasks):
        log(f"  Group {group_id} [{task_idx+1}/{len(my_tasks)}] ({dataset_name})", gpu_id)

        t0 = time.time()
        result = evaluate_single_group(agent, json_file, group_id, args.budget_ratio, args.seed, gpu_id)
        elapsed = time.time() - t0

        completed += 1
        if result['success']:
            success += 1
            prices = [it['pred_score'] for it in result['items']]
            log(f"  -> done {elapsed:.1f}s prices={prices}", gpu_id)
        else:
            log(f"  -> FAILED: {result.get('error','')}", gpu_id)

        if dataset_name not in all_results['datasets']:
            all_results['datasets'][dataset_name] = []
        all_results['datasets'][dataset_name].append(result)
        safe_save(result_file, all_results)

        elapsed_total = time.time() - start_time
        rate = completed / elapsed_total if elapsed_total > 0 else 0
        eta = (len(my_tasks) - completed) / rate if rate > 0 else 0
        log(f"Progress: {completed}/{len(my_tasks)} (success={success}) | "
            f"{elapsed_total/60:.1f}min elapsed | ETA={eta/60:.1f}min", gpu_id)

    total_elapsed = time.time() - start_time
    log(f"\n{'='*60}", gpu_id)
    log(f"DONE: {success}/{completed} groups succeeded in {total_elapsed/60:.1f} min", gpu_id)
    log(f"{'='*60}", gpu_id)


if __name__ == '__main__':
    main()
