#!/usr/bin/python3
"""
Single-agent price prediction (Qwen3-VL-8B-Instruct local).
Single-GPU worker: processes assigned groups on one GPU with batch inference.
Launch two instances (--gpu_id 0 and --gpu_id 1) for dual-GPU parallelism.

Results saved to: /mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-8B/result/Single-agent_price/
"""

import os
import sys
import json
import argparse
import time
import fcntl
from pathlib import Path
from datetime import datetime
import numpy as np

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

# Paths
INTERFACE_DIR = Path(__file__).resolve().parent
METHOD_DIR = INTERFACE_DIR.parent
DATA_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data')
RESULT_DIR = METHOD_DIR / 'result' / 'Single-agent_price'
EXPERIMENT_DIR = '/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main'

DEFAULT_TEMPERATURE = 0.0
BATCH_SIZE = 6  # items per batch (each group has 6 items)


def log(msg, gpu_id=None):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]" if gpu_id is None else f"[{ts}][GPU{gpu_id}]"
    print(f"{prefix} {msg}", flush=True)


def get_group_ids(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return sorted(set(item['group_id'] for item in data))


def calculate_rel(y):
    return float(np.log(1 + y))


def safe_save(result_file, all_results):
    """Thread/process-safe JSON save using file locking."""
    lock_file = str(result_file) + '.lock'
    with open(lock_file, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            # Merge with any existing data from the other GPU worker
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        existing = json.load(f)
                    # Merge datasets: keep all results from both workers
                    for ds_name, ds_results in existing.get('datasets', {}).items():
                        if ds_name not in all_results['datasets']:
                            all_results['datasets'][ds_name] = ds_results
                        else:
                            # Merge: add any group_ids we don't have
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
    """Load completed group keys from existing result file."""
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


def evaluate_single_group(agent, json_file, group_id, seed, gpu_id, batch_size):
    """Evaluate a single group using batched inference."""
    try:
        from dataset import AuctionDataset

        dataset = AuctionDataset(json_path=str(json_file), seed=seed)
        episode = dataset.create_episode(
            group_id=group_id, shuffle_groups=False,
            shuffle_items=False, shuffle_images=False
        )

        items = episode['items']
        if not items:
            return {'group_id': group_id, 'items': [], 'success': False, 'error': 'No items'}

        agent.reset()
        items_result = []

        for batch_start in range(0, len(items), batch_size):
            batch_items = items[batch_start:batch_start + batch_size]
            batch_data = []

            for idx_offset, item in enumerate(batch_items):
                item_idx = batch_start + idx_offset + 1
                img_list = item.get('images', [])
                img_name = os.path.basename(img_list[0]) if img_list else "none"
                log(f"    [batch] Item {item_idx}/{len(items)} [{item['folder_id']}] img={img_name}", gpu_id)
                batch_data.append({
                    'images': img_list,
                    'item_index': item_idx,
                })

            t0 = time.time()
            if len(batch_data) == 1:
                price_results = [agent.predict_price(
                    item_images=batch_data[0]['images'],
                    item_index=batch_data[0]['item_index'],
                )]
            else:
                price_results = agent.predict_price_batch(batch_data)
            elapsed = time.time() - t0

            for idx_offset, (item, price_result) in enumerate(zip(batch_items, price_results)):
                item_idx = batch_start + idx_offset + 1
                bid_price = price_result['bid_price']
                reasoning = price_result.get('reasoning', '')
                log(f"      -> item {item_idx}: price={bid_price:.1f}, reason={reasoning[:80]}", gpu_id)

                items_result.append({
                    'name': item['folder_id'],
                    'folder_id': item['folder_id'],
                    'true_price': item['price'],
                    'pred_score': bid_price,
                    'rel': calculate_rel(item['price']),
                    'reasoning': reasoning,
                    'inference_time': round(elapsed / len(batch_items), 1),
                })

            log(f"    [batch] {len(batch_items)} items done in {elapsed:.1f}s "
                f"({elapsed/len(batch_items):.1f}s/item)", gpu_id)

        return {'group_id': group_id, 'items': items_result, 'success': True}

    except Exception as e:
        log(f"    ERROR: {e}", gpu_id)
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush()
        return {'group_id': group_id, 'items': [], 'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-8B Price Prediction (Single GPU Worker)')
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID (0 or 1)')
    parser.add_argument('--num_gpus', type=int, default=2, help='Total number of GPUs (for task splitting)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    num_gpus = args.num_gpus
    batch_size = args.batch_size

    # Set GPU visibility BEFORE importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    sys.path.insert(0, str(INTERFACE_DIR))

    output_dir = RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "Single-agent_price.json"

    log(f"{'='*60}", gpu_id)
    log(f"Qwen3-VL-8B-Instruct Price Prediction Worker", gpu_id)
    log(f"  GPU={gpu_id}, batch_size={batch_size}", gpu_id)
    log(f"  temperature={args.temperature}, seed={args.seed}", gpu_id)
    log(f"{'='*60}", gpu_id)

    # Build task list
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
            log(f"WARNING: {json_file} not found, skipping", gpu_id)
            continue
        group_ids = get_group_ids(json_file)
        log(f"  {dataset_name}: {len(group_ids)} groups", gpu_id)
        for gid in group_ids:
            all_tasks.append((dataset_name, str(json_file), gid))

    # Split tasks across GPUs: this worker takes every num_gpus-th task
    my_tasks = [t for i, t in enumerate(all_tasks) if i % num_gpus == gpu_id]
    log(f"Total: {len(all_tasks)} groups, this worker: {len(my_tasks)} groups", gpu_id)

    # Check for completed groups (shared result file)
    completed_keys = load_completed_keys(result_file)
    my_tasks = [(ds, jf, gid) for ds, jf, gid in my_tasks
                if (ds, gid) not in completed_keys]
    log(f"Already completed: {len(completed_keys)}, remaining for this worker: {len(my_tasks)}", gpu_id)

    if not my_tasks:
        log("All assigned groups already completed!", gpu_id)
        return

    # Load model
    log("Loading model...", gpu_id)
    from sigelLLM import SigelLLMAuction
    agent = SigelLLMAuction(temperature=args.temperature)
    log("Model loaded, starting evaluation", gpu_id)

    # Process groups
    all_results = {
        'model_name': 'qwen3-vl-8b-instruct',
        'method': 'Single-agent_price',
        'temperature': args.temperature,
        'batch_size': batch_size,
        'datasets': {}
    }

    start_time = time.time()
    completed = 0
    success = 0

    for task_idx, (dataset_name, json_file, group_id) in enumerate(my_tasks):
        log(f"  Group {group_id} [{task_idx+1}/{len(my_tasks)}] ({dataset_name})", gpu_id)

        t_group = time.time()
        result = evaluate_single_group(agent, json_file, group_id, args.seed, gpu_id, batch_size)
        elapsed_group = time.time() - t_group

        if result['success']:
            scores = [it['pred_score'] for it in result['items']]
            log(f"  -> Group {group_id} done: {elapsed_group:.1f}s, prices={scores}", gpu_id)
            success += 1
        else:
            log(f"  -> Group {group_id} FAILED: {result.get('error','')}", gpu_id)

        completed += 1

        # Append to local results
        if dataset_name not in all_results['datasets']:
            all_results['datasets'][dataset_name] = []
        all_results['datasets'][dataset_name].append(result)

        # Save with file locking (merge with other worker's results)
        safe_save(result_file, all_results)

        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(my_tasks) - completed) / rate if rate > 0 else 0
        log(f"Progress: {completed}/{len(my_tasks)} "
            f"(success={success}) | {elapsed/60:.1f}min elapsed | ETA={eta/60:.1f}min", gpu_id)

    total_elapsed = time.time() - start_time
    log(f"\n{'='*60}", gpu_id)
    log(f"DONE: {success}/{completed} groups succeeded", gpu_id)
    log(f"Total time: {total_elapsed/60:.1f} min", gpu_id)
    log(f"{'='*60}", gpu_id)


if __name__ == '__main__':
    main()
