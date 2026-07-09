#!/usr/bin/python3
"""
Valuator_judge (Qwen3-VL-4B local).
Multi-agent evaluation with competitive scoring using local Qwen3-VL inference.
Uses Valuator_judge with native local model support (no monkey-patching).
Optimized for memory efficiency with aggressive cache clearing.
"""

import os
import sys
import json
import glob
import argparse
import tempfile
import math
from pathlib import Path
from PIL import Image
import time
from multiprocessing import Pool, set_start_method
import torch

# Set PyTorch CUDA memory allocator to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

INTERFACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTERFACE_DIR))
sys.path.insert(0, '/mnt/data/user/fengxian.ji/AUCTION/method/Valuator_judge')
sys.path.insert(0, '/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main')

from src.item_base import Item
from src.bidder_base import Bidder
from auction_workflow import run_monte_carlo
import calculate_matrix

DATA_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data')
RESULT_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-4B/result/Valuator_judge')

DEFAULT_TEMPERATURE = 0.0
DEFAULT_BUDGET_RATIO = 1.2
DEFAULT_NUM_RUNS = 1


def _create_contact_sheet(image_paths, thumbnail_size=(300, 300), output_path=None):
    """Combine multiple images into a single contact sheet."""
    num_images = len(image_paths)
    if num_images == 0:
        return None
    if num_images == 1:
        # Single image: just copy/save it
        if output_path:
            img = Image.open(image_paths[0])
            img.save(output_path)
        return output_path or image_paths[0]
    grid_cols = math.ceil(math.sqrt(num_images))
    grid_rows = math.ceil(num_images / grid_cols)
    canvas = Image.new("RGB", (grid_cols * thumbnail_size[0], grid_rows * thumbnail_size[1]), "white")
    for idx, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            continue
        row, col = idx // grid_cols, idx % grid_cols
        img = Image.open(img_path)
        img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
        canvas.paste(img, (col * thumbnail_size[0], row * thumbnail_size[1]))
    if output_path:
        canvas.save(output_path)
    return output_path


def load_group_items(json_file, group_id, temp_dir=None):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    group_data = [item for item in data if item['group_id'] == group_id]
    items = []
    for idx, item_data in enumerate(group_data, 1):
        folder_path = item_data['folder_path']
        figure_dir = os.path.join(folder_path, 'figure')
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(figure_dir, ext)))
        image_files.sort()

        # If multiple images, create contact sheet; otherwise use single image
        if len(image_files) > 1 and temp_dir:
            cs_path = os.path.join(temp_dir, f"cs_group{group_id}_item{idx}.jpg")
            _create_contact_sheet(image_files, output_path=cs_path)
            main_image = cs_path
        else:
            main_image = image_files[0] if image_files else None

        item = Item(
            id=idx,
            name=f"Artwork #{item_data['folder_id']}",
            price=1,
            image_path=main_image,
            thumbnail_path=main_image,
            context=None,
            real_final_price=item_data['price']
        )
        items.append(item)

    return items, group_data


def create_bidders_config(temperature, num_bidders=3):
    configs = []
    for i in range(1, num_bidders + 1):
        configs.append({
            'name': f'Bidder {i}',
            'model_name': 'qwen3-vl-4b-local',  # triggers local model in _parse_llm
            'budget': 1000,
            'desire': 'aesthetic_alignment',
            'plan_strategy': 'adaptive',
            'temperature': temperature,
            'overestimate_percent': 0,
            'correct_belief': True,
            'enable_learning': False
        })
    return configs


def evaluate_single_group(json_file, group_id, budget_ratio, num_runs, temperature, seed):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            items, original_data = load_group_items(json_file, group_id, temp_dir=temp_dir)
            if not items:
                return {'group_id': group_id, 'items': [], 'success': False, 'error': 'No items'}

            bidders_config = create_bidders_config(temperature, num_bidders=3)

            total_real_price = sum(item.real_final_price for item in items)
            unified_budget = total_real_price * budget_ratio
            for bc in bidders_config:
                bc['budget'] = int(unified_budget)

            auction_hash = f"group{group_id}_{int(time.time())}"
            auctioneer_config = {'enable_discount': False, 'min_markup_pct': 0.1}

            statistics = run_monte_carlo(
                auction_hash=auction_hash,
                items_list=items,
                bidders_config_list=bidders_config,
                auctioneer_config=auctioneer_config,
                num_runs=num_runs,
                thread_num=1,  # single-thread for local model
                shuffle=False,
                random_seed=seed,
                log_dir=temp_dir
            )

        # Clear KV cache to prevent memory accumulation across groups
        sys.path.insert(0, str(INTERFACE_DIR))
        from model_loader import get_qwen_model
        import torch
        import gc
        model = get_qwen_model()
        model.clear_cache()
        # Additional aggressive cleanup
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

        items_result = []
        for item_name, stats in statistics.items():
            folder_id = item_name.split('#')[-1].strip()
            orig_item = next((d for d in original_data if str(d['folder_id']) == folder_id), None)
            if orig_item:
                items_result.append({
                    'name': folder_id,
                    'folder_id': folder_id,
                    'true_price': orig_item['price'],
                    'pred_score': stats['mean'],
                    'rel': calculate_matrix.calculate_rel(orig_item['price']),
                    'reasoning': f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
                })

        return {'group_id': group_id, 'items': items_result, 'success': True}

    except Exception as e:
        print(f"Error evaluating group {group_id}: {e}")
        import traceback; traceback.print_exc()
        return {'group_id': group_id, 'items': [], 'success': False, 'error': str(e)}


def evaluate_single_group_wrapper(args):
    """Wrapper function for multiprocessing"""
    json_file, gid, budget_ratio, num_runs, temperature, seed = args
    return evaluate_single_group(json_file, gid, budget_ratio, num_runs, temperature, seed)


def evaluate_model(temperature, output_dir, budget_ratio, num_runs, seed, test_mode=False, num_workers=2):
    print(f"\n{'='*80}")
    print(f"Qwen3-VL-4B Valuator_judge (Multi-agent Competitive Scoring 0-100)")
    print(f"Using local Qwen3-VL-4B weights: /mnt/data/user/fengxian.ji/AUCTION/method/Qwen3vl-4B/weight")
    print(f"Memory optimization: Per-item cache clearing + expandable_segments")
    if test_mode:
        print(f"TEST MODE: Only evaluating first 3 groups per dataset")
    print(f"PARALLEL MODE: Using {num_workers} workers")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'model_name': 'qwen3-vl-4b-instruct',
        'method': 'Valuator_judge',
        'temperature': temperature,
        'budget_ratio': budget_ratio,
        'num_runs': num_runs,
        'test_mode': test_mode,
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

        # TEST MODE: only process first 3 groups
        if test_mode:
            group_ids = group_ids[:3]
            print(f"  TEST MODE: Processing {len(group_ids)}/3 groups only")
        else:
            print(f"  {len(group_ids)} groups (parallel with {num_workers} workers)")

        # Prepare arguments for parallel processing
        args_list = [(json_file, gid, budget_ratio, num_runs, temperature, seed) for gid in group_ids]

        # Process in parallel
        results = []
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                for i, r in enumerate(pool.imap(evaluate_single_group_wrapper, args_list)):
                    gid = group_ids[i]
                    status = "OK" if r['success'] else f"FAIL: {r.get('error','')}"
                    print(f"  [{i+1}/{len(group_ids)}] Group {gid}... {status}")
                    results.append(r)

                    # Incremental save after each group
                    all_results['datasets'][dataset_name] = [x for x in results if x['success']]
                    result_file = output_dir / "Valuator_judge.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, indent=2, ensure_ascii=False)
        else:
            # Sequential processing (fallback)
            for i, gid in enumerate(group_ids):
                print(f"  [{i+1}/{len(group_ids)}] Group {gid}...", end=" ", flush=True)
                r = evaluate_single_group(json_file, gid, budget_ratio, num_runs, temperature, seed)
                results.append(r)
                status = "OK" if r['success'] else f"FAIL: {r.get('error','')}"
                print(status)

                # Incremental save after each group
                all_results['datasets'][dataset_name] = [x for x in results if x['success']]
                result_file = output_dir / "Valuator_judge.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

        success = [r for r in results if r['success']]
        print(f"  Done: {len(success)}/{len(group_ids)} groups succeeded")
        all_results['datasets'][dataset_name] = success

    result_file = output_dir / "Valuator_judge.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {result_file}")

    metrics_result = {}
    for ds_name in all_results['datasets']:
        episodes = all_results['datasets'][ds_name]
        if episodes:
            print(f"  {ds_name}:")
            metrics_result[ds_name] = calculate_matrix.calculate_all_metrics(episodes)

    matrix_file = output_dir / "Valuator_judge_matrix.json"
    with open(matrix_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_result, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved: {matrix_file}")

    return all_results, metrics_result


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Qwen3-VL-4B Valuator_judge (Multi-agent Scoring)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--budget_ratio', type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument('--num_runs', type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', action='store_true', help='Test mode: only evaluate first 3 groups per dataset')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of parallel workers (default: 2, optimized for 48GB GPU)')
    args = parser.parse_args()

    start_time = time.time()
    evaluate_model(args.temperature, RESULT_DIR, args.budget_ratio, args.num_runs, args.seed,
                   test_mode=args.test, num_workers=args.num_workers)
    print(f"\nTotal time: {(time.time()-start_time)/60:.2f} min")


if __name__ == '__main__':
    main()
