#!/usr/bin/python3
"""
Single-agentLLM_score (Qwen3-VL-4B local) - Per-item scoring 0-100.
Dual-GPU parallel: loads one model per GPU, processes groups concurrently.

Results saved to: /mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-4B/result/
"""

import os
import sys
import json
import glob
import argparse
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"

# Paths
INTERFACE_DIR = Path(__file__).resolve().parent
METHOD_DIR = INTERFACE_DIR.parent
DATA_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data')
RESULT_DIR = METHOD_DIR / 'result'

DEFAULT_TEMPERATURE = 0.0   # greedy decoding
DEFAULT_BUDGET_RATIO = 1.2
NUM_GPUS = 2


def log(msg, gpu_id=None):
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}]" if gpu_id is None else f"[{ts}][GPU{gpu_id}]"
    print(f"{prefix} {msg}", flush=True)


def load_group_items(json_file, group_id):
    """Load items for a specific group_id from JSON, with images."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [d for d in data if d['group_id'] == group_id]
    result = []
    for item in items:
        fig_dir = os.path.join(item['folder_path'], 'figure')
        images = []
        if os.path.exists(fig_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(glob.glob(os.path.join(fig_dir, ext)))
            images.sort()
        result.append({
            'folder_id': item['folder_id'],
            'price': item['price'],
            'folder_path': item['folder_path'],
            'images': images,
        })
    return result


def get_group_ids(json_file):
    """Get sorted unique group_ids from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return sorted(set(item['group_id'] for item in data))


def evaluate_single_group(agent, json_file, group_id, gpu_id):
    """Evaluate a single group (6 items) using local Qwen3-VL."""
    try:
        items = load_group_items(json_file, group_id)
        if not items:
            return {'group_id': group_id, 'items': [], 'success': False, 'error': 'No items'}

        items_result = []
        for idx, item in enumerate(items, 1):
            img_list = item.get('images', [])
            img_name = os.path.basename(img_list[0]) if img_list else "none"
            log(f"    Item {idx}/{len(items)} [{item['folder_id']}] img={img_name}", gpu_id)

            t0 = time.time()
            score_result = agent.predict_score(item_images=img_list, item_index=idx)
            elapsed = time.time() - t0

            score = score_result['score']
            reasoning = score_result.get('reasoning', '')
            log(f"      -> score={score:.1f}, time={elapsed:.1f}s, reason={reasoning[:80]}", gpu_id)

            items_result.append({
                'name': item['folder_id'],
                'folder_id': item['folder_id'],
                'true_price': item['price'],
                'pred_score': score,
                'reasoning': reasoning,
                'inference_time': round(elapsed, 1),
            })

        return {'group_id': group_id, 'items': items_result, 'success': True}

    except Exception as e:
        log(f"    ERROR: {e}", gpu_id)
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush()
        return {'group_id': group_id, 'items': [], 'success': False, 'error': str(e)}


# ── Worker process ──────────────────────────────────────────────────────

def gpu_worker(gpu_id, task_queue, result_queue, temperature):
    """
    Worker process: sets CUDA_VISIBLE_DEVICES, loads model, processes tasks.
    Each worker sees only ONE GPU (mapped to cuda:0 inside the process).
    """
    # Set GPU visibility BEFORE importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['PYTHONUNBUFFERED'] = '1'

    # Add interface dir to path
    sys.path.insert(0, str(INTERFACE_DIR))

    log(f"Worker starting, CUDA_VISIBLE_DEVICES={gpu_id}", gpu_id)

    # Now import torch and model (will only see the assigned GPU)
    from sigelLLM import SigelLLMAuction

    agent = SigelLLMAuction(temperature=temperature)
    log("Model loaded, ready for tasks", gpu_id)

    while True:
        task = task_queue.get()
        if task is None:  # poison pill → exit
            log("Received shutdown signal, exiting", gpu_id)
            break

        dataset_name, json_file, group_id, group_idx, total_groups = task
        log(f"  Group {group_id} [{group_idx}/{total_groups}] ({dataset_name})", gpu_id)

        agent.reset()
        t_group = time.time()
        result = evaluate_single_group(agent, json_file, group_id, gpu_id)
        elapsed_group = time.time() - t_group

        if result['success']:
            scores = [it['pred_score'] for it in result['items']]
            log(f"  -> Group {group_id} done: {elapsed_group:.1f}s, scores={scores}", gpu_id)
        else:
            log(f"  -> Group {group_id} FAILED: {result.get('error','')}", gpu_id)

        result_queue.put((dataset_name, group_id, result))


# ── Main process ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-4B Single-agentLLM_score (Dual GPU)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--budget_ratio', type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_gpus', type=int, default=NUM_GPUS)
    args = parser.parse_args()

    num_gpus = args.num_gpus
    output_dir = RESULT_DIR / 'Single-agentLLM_score'
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "Single-agentLLM_score.json"

    log(f"{'='*60}")
    log(f"Qwen3-VL-4B Single-agentLLM_score (Dual GPU, {num_gpus} workers)")
    log(f"temperature={args.temperature}, budget_ratio={args.budget_ratio}")
    log(f"output_dir={output_dir}")
    log(f"{'='*60}")

    dataset_files = [
        ('Low-price', DATA_DIR / 'Low-price.json'),
        ('Mid-price', DATA_DIR / 'Mid-price.json'),
        ('High-price', DATA_DIR / 'High-price.json'),
        ('Near-Tie-low', DATA_DIR / 'Near-Tie-low.json'),
        ('Near-Tie-high', DATA_DIR / 'Near-Tie-high.json'),
        ('Overall', DATA_DIR / 'Overall.json')
    ]

    # Build complete task list: (dataset_name, json_file, group_id)
    all_tasks = []
    for dataset_name, json_file in dataset_files:
        if not json_file.exists():
            log(f"WARNING: {json_file} not found, skipping")
            continue
        group_ids = get_group_ids(json_file)
        log(f"  {dataset_name}: {len(group_ids)} groups")
        for gid in group_ids:
            all_tasks.append((dataset_name, str(json_file), gid))

    total_tasks = len(all_tasks)
    log(f"Total: {total_tasks} groups to process with {num_gpus} GPUs")

    # Check for existing results to resume from
    completed_keys = set()
    all_results = {
        'model_name': 'qwen3-vl-4b-thinking',
        'method': 'Single-agentLLM_score',
        'temperature': args.temperature,
        'budget_ratio': args.budget_ratio,
        'num_gpus': num_gpus,
        'datasets': {}
    }
    if result_file.exists():
        try:
            with open(result_file, 'r') as f:
                prev = json.load(f)
            for ds_name, ds_results in prev.get('datasets', {}).items():
                all_results['datasets'][ds_name] = ds_results
                for r in ds_results:
                    if r.get('success'):
                        completed_keys.add((ds_name, r['group_id']))
            log(f"Resuming: {len(completed_keys)} groups already completed")
        except Exception as e:
            log(f"Could not load previous results: {e}, starting fresh")

    # Filter out already-completed tasks
    remaining_tasks = [(ds, jf, gid) for ds, jf, gid in all_tasks
                       if (ds, gid) not in completed_keys]
    log(f"Remaining: {len(remaining_tasks)} groups to process")

    if not remaining_tasks:
        log("All groups already completed!")
        return

    # Start worker processes (spawn to avoid CUDA fork issues)
    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    workers = []
    for gpu_id in range(num_gpus):
        p = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, args.temperature),
            name=f"gpu-worker-{gpu_id}",
        )
        p.start()
        workers.append(p)
        log(f"Started worker on GPU {gpu_id} (PID={p.pid})")

    # Enqueue all tasks
    for i, (ds, jf, gid) in enumerate(remaining_tasks):
        task_queue.put((ds, jf, gid, i + 1, len(remaining_tasks)))

    # Send poison pills (one per worker)
    for _ in range(num_gpus):
        task_queue.put(None)

    # Collect results
    start_time = time.time()
    completed = 0
    success = 0

    for _ in range(len(remaining_tasks)):
        dataset_name, group_id, result = result_queue.get()

        # Append to datasets
        if dataset_name not in all_results['datasets']:
            all_results['datasets'][dataset_name] = []
        all_results['datasets'][dataset_name].append(result)

        completed += 1
        if result['success']:
            success += 1

        # Save after every group
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(remaining_tasks) - completed) / rate if rate > 0 else 0
        log(f"Progress: {completed}/{len(remaining_tasks)} "
            f"(success={success}) | {elapsed/60:.1f}min elapsed | ETA={eta/60:.1f}min")

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            log(f"Worker {p.name} did not exit, terminating")
            p.terminate()

    # Sort results within each dataset by group_id
    for ds_name in all_results['datasets']:
        all_results['datasets'][ds_name].sort(key=lambda r: r['group_id'])

    # Final save
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"DONE: {success}/{completed} groups succeeded")
    log(f"Total time: {total_elapsed/60:.1f} min")
    log(f"Results: {result_file}")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()
