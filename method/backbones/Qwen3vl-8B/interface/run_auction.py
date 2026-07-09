#!/usr/bin/python3
"""
Valuator / Auction-arena (Qwen3-VL-8B-Instruct local).
Multi-agent auction with Monte Carlo sampling using local Qwen3-VL inference.
Monkey-patches Bidder._parse_llm to use LocalQwenChatModel.
Dual-GPU: launch two instances (--gpu_id 0 and --gpu_id 1).
"""

import os
import sys
import json
import glob
import argparse
import fcntl
import tempfile
import math
import time
from pathlib import Path
from datetime import datetime
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

INTERFACE_DIR = Path(__file__).resolve().parent
METHOD_DIR = INTERFACE_DIR.parent
DATA_DIR = Path('/mnt/data/user/fengxian.ji/AUCTION/benchmark/EmergingArt/data')
RESULT_DIR = METHOD_DIR / 'result' / 'Auction-arena'
EXPERIMENT_DIR = '/mnt/data/user/fengxian.ji/AUCTION/show/experiment/[ICML][][0][0107]Main'

DEFAULT_TEMPERATURE = 0.0
DEFAULT_BUDGET_RATIO = 1.2
DEFAULT_NUM_RUNS = 1


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


def _create_contact_sheet(image_paths, thumbnail_size=(300, 300), output_path=None):
    from PIL import Image
    num_images = len(image_paths)
    if num_images == 0:
        return None
    if num_images == 1:
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
    from src.item_base import Item

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
            'model_name': 'qwen3-vl-8b-local',
            'budget': 1000,
            'desire': 'aesthetic_alignment',
            'plan_strategy': 'adaptive',
            'temperature': temperature,
            'overestimate_percent': 0,
            'correct_belief': True,
            'enable_learning': False
        })
    return configs


def setup_monkey_patch():
    """Monkey-patch Bidder to use local Qwen model."""
    from langchain_local import LocalQwenChatModel
    from src.bidder_base import Bidder

    _original_parse_llm = Bidder._parse_llm
    _original_run_llm_standalone = Bidder._run_llm_standalone

    def _patched_parse_llm(self):
        if 'qwen' in self.model_name.lower() or 'local' in self.model_name.lower():
            self.llm = LocalQwenChatModel(temperature=self.temperature)
        else:
            _original_parse_llm(self)

    def _patched_run_llm_standalone(self, messages: list):
        if 'qwen' in self.model_name.lower() or 'local' in self.model_name.lower():
            try:
                input_token_num = self.llm.get_num_tokens_from_messages(messages)
            except (NotImplementedError, AttributeError):
                total_chars = sum(len(str(getattr(m, 'content', m))) for m in messages)
                input_token_num = total_chars // 4
            max_tokens = max(4000 - input_token_num, 192)
            item_image = None
            if hasattr(self, 'items') and hasattr(self, 'cur_item_id'):
                if self.cur_item_id < len(self.items):
                    cur_item = self.items[self.cur_item_id]
                    if cur_item.has_image():
                        item_image = cur_item.image_path
            result = self.llm(messages, max_tokens=max_tokens, item_image=item_image)
            return result.content if hasattr(result, 'content') else str(result)
        else:
            return _original_run_llm_standalone(self, messages)

    Bidder._parse_llm = _patched_parse_llm
    Bidder._run_llm_standalone = _patched_run_llm_standalone


def evaluate_single_group(json_file, group_id, budget_ratio, num_runs, temperature, seed, gpu_id):
    import calculate_matrix as cm
    from auction_workflow import run_monte_carlo

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
                thread_num=1,
                shuffle=False,
                random_seed=seed,
                log_dir=temp_dir
            )

        # Clear cache
        from model_loader import get_qwen_model
        import torch, gc
        model = get_qwen_model()
        model.clear_cache()
        torch.cuda.empty_cache()
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
                    'rel': cm.calculate_rel(orig_item['price']),
                    'reasoning': f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
                })

        return {'group_id': group_id, 'items': items_result, 'success': True}

    except Exception as e:
        log(f"    ERROR: {e}", gpu_id)
        import traceback; traceback.print_exc(file=sys.stdout); sys.stdout.flush()
        return {'group_id': group_id, 'items': [], 'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL-8B Auction-arena (Single GPU Worker)')
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--budget_ratio', type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument('--num_runs', type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    sys.path.insert(0, str(INTERFACE_DIR))
    sys.path.insert(0, '/mnt/data/user/fengxian.ji/AUCTION/method/Valuator')
    sys.path.insert(0, EXPERIMENT_DIR)

    # Apply monkey-patch after setting up paths
    setup_monkey_patch()

    output_dir = RESULT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "Auction-arena.json"

    log(f"{'='*60}", gpu_id)
    log(f"Qwen3-VL-8B Auction-arena Worker", gpu_id)
    log(f"  GPU={gpu_id}, num_runs={args.num_runs}", gpu_id)
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

    # Pre-load model
    log("Loading model...", gpu_id)
    from model_loader import get_qwen_model
    get_qwen_model()
    log("Model loaded", gpu_id)

    all_results = {
        'model_name': 'qwen3-vl-8b-instruct',
        'method': 'Auction-arena',
        'temperature': args.temperature,
        'budget_ratio': args.budget_ratio,
        'num_runs': args.num_runs,
        'datasets': {}
    }

    start_time = time.time()
    completed = 0
    success = 0

    for task_idx, (dataset_name, json_file, group_id) in enumerate(my_tasks):
        log(f"  Group {group_id} [{task_idx+1}/{len(my_tasks)}] ({dataset_name})", gpu_id)

        t0 = time.time()
        result = evaluate_single_group(
            json_file, group_id, args.budget_ratio, args.num_runs,
            args.temperature, args.seed, gpu_id
        )
        elapsed = time.time() - t0

        completed += 1
        if result['success']:
            success += 1
            scores = [it['pred_score'] for it in result['items']]
            log(f"  -> done {elapsed:.1f}s scores={scores}", gpu_id)
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
