#!/usr/bin/python3
"""Auction-arena on ImageReward using remote qwen3-vl-30b-a3b-instruct."""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path


INTERFACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTERFACE_DIR.parents[2]
VALUATOR_DIR = REPO_ROOT / "method" / "Valuator"
EXPERIMENT_DIR = REPO_ROOT / "show" / "experiment" / "[ICML][][0][0107]Main"

for path in (INTERFACE_DIR, VALUATOR_DIR, EXPERIMENT_DIR):
    sys.path.insert(0, str(path))

try:
    import gradio  # noqa: F401
except ModuleNotFoundError:
    sys.modules["gradio"] = types.SimpleNamespace(update=lambda **kwargs: kwargs)

import auction_workflow as auction_workflow_module
import src.bidder_base as bidder_base_module
from langchain_local import LocalQwenChatModel
from model_loader import get_qwen_model
from src.auctioneer_base import Auctioneer
from src.bidder_base import Bidder
from src.item_base import Item
from auction_workflow import run_monte_carlo


BENCHMARK_CANDIDATES = [
    Path("/mnt/data/user/fengxian.ji/AUCTION/benchmark/ImageReward"),
    REPO_ROOT / "benchmark" / "ImageReward",
]
RESULT_ROOT = (
    REPO_ROOT
    / "show"
    / "experiment"
    / "[ICML][Mechanism-Based Value Elicitation ][2][26015]Human preference sanity "
    / "ImageReward(test)"
    / "ImageReward"
    / "result"
    / "run_qwen3vl-30B"
)
RESULT_DIR = RESULT_ROOT / "Auction-arena"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_BUDGET_RATIO = 1.2
DEFAULT_NUM_RUNS = 1
DEFAULT_PROMPT_TYPES = ["original"]
MEAN_PRICE_OVERALL = 96.8565


_original_parse_llm = Bidder._parse_llm
_original_run_llm_standalone = Bidder._run_llm_standalone
_original_parse_bid = Auctioneer.parse_bid


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_benchmark_dir(cli_path: str | None) -> Path:
    candidates = [Path(cli_path)] if cli_path else []
    candidates.extend(BENCHMARK_CANDIDATES)
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot find ImageReward benchmark directory. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def patch_valuator_runtime() -> None:
    def _fallback_result(func_type: str, bidder_name: str, exc: Exception) -> str:
        log(f"{bidder_name} {func_type} fallback after error: {exc}")
        if func_type == "bid":
            return "I'm out!"
        return ""

    def _patched_bidding_multithread(
        bidder_list,
        instruction_list,
        func_type,
        thread_num=5,
        retry=1,
    ):
        del thread_num, retry
        assert func_type in ["plan", "bid", "summarize", "replan"]

        if isinstance(instruction_list, str):
            instruction_list = [instruction_list] * len(bidder_list)

        results = []
        for bidder, msg in zip(bidder_list, instruction_list):
            try:
                if func_type == "bid":
                    result = bidder.bid(msg)
                elif func_type == "summarize":
                    result = bidder.summarize(msg)
                elif func_type == "plan":
                    result = bidder.init_plan(msg)
                elif func_type == "replan":
                    result = bidder.replan(msg)
                else:
                    raise NotImplementedError(f"func_type {func_type} not implemented")
            except Exception as exc:
                import traceback

                log(f"Sequential {func_type} failed for {bidder.name}: {exc}")
                traceback.print_exc()
                result = _fallback_result(func_type, bidder.name, exc)
            results.append(result)
        return results

    def _patched_parse_llm(self):
        if "qwen" in self.model_name.lower() or "local" in self.model_name.lower():
            self.llm = LocalQwenChatModel(temperature=self.temperature)
        else:
            _original_parse_llm(self)

    def _patched_run_llm_standalone(self, messages: list):
        if "qwen" in self.model_name.lower() or "local" in self.model_name.lower():
            try:
                input_token_num = self.llm.get_num_tokens_from_messages(messages)
            except (NotImplementedError, AttributeError):
                total_chars = sum(len(str(getattr(m, "content", m))) for m in messages)
                input_token_num = total_chars // 4

            max_tokens = min(max(1200 - input_token_num, 192), 512)
            item_image = None
            if hasattr(self, "items") and hasattr(self, "cur_item_id"):
                if self.cur_item_id < len(self.items):
                    cur_item = self.items[self.cur_item_id]
                    if cur_item.has_image():
                        item_image = cur_item.image_path

            result = self.llm(messages, max_tokens=max_tokens, item_image=item_image)
            return result.content if hasattr(result, "content") else str(result)
        return _original_run_llm_standalone(self, messages)

    def _local_parse_bid(self, text: str):
        text_lower = text.lower().strip()
        withdraw_patterns = [
            "i'm out",
            "i am out",
            "withdraw",
            "i pass",
            "i fold",
            "i decline",
            "i quit",
            "not bidding",
            "i choose not to bid",
            "i will not bid",
        ]
        for pattern in withdraw_patterns:
            if pattern in text_lower:
                return -1

        price_matches = re.findall(r"\$\s*([\d,]+)", text)
        if price_matches:
            try:
                return int(price_matches[-1].replace(",", ""))
            except ValueError:
                pass

        bid_matches = re.findall(r"(?:bid|offer|pay)\s+(?:\$?)(\d+)", text_lower)
        if bid_matches:
            return int(bid_matches[-1])

        if re.match(r"^-?\d+$", text.strip()):
            return int(text.strip())

        print(f"* Cannot parse bid locally: {text[:120]}", flush=True)
        return None

    Bidder._parse_llm = _patched_parse_llm
    Bidder._run_llm_standalone = _patched_run_llm_standalone
    Auctioneer.parse_bid = _local_parse_bid
    bidder_base_module.bidding_multithread = _patched_bidding_multithread
    auction_workflow_module.bidding_multithread = _patched_bidding_multithread


def load_split_tasks(benchmark_dir: Path, split: str, prompt_types: list[str]) -> list[dict]:
    json_path = benchmark_dir / f"{split}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    grouped = defaultdict(list)
    prompt_text_by_group = {}
    for row in rows:
        group_id = row["prompt_id"]
        grouped[group_id].append(row)
        prompt_text_by_group[group_id] = row["prompt"].strip()

    tasks = []
    for group_id, images in grouped.items():
        img_paths = [resolve_image_path(benchmark_dir, image["image_path"]) for image in images]
        ranks = [image["rank"] for image in images]
        for prompt_type in prompt_types:
            if prompt_type != "original":
                continue
            tasks.append(
                {
                    "split": split,
                    "group_id": group_id,
                    "prompt_type": prompt_type,
                    "prompt_text": prompt_text_by_group[group_id],
                    "img_paths": img_paths,
                    "ranks": ranks,
                }
            )
    return tasks


def resolve_image_path(benchmark_dir: Path, image_path: str) -> str:
    direct_path = benchmark_dir / image_path
    if direct_path.exists():
        return str(direct_path)
    flat_path = benchmark_dir / Path(image_path).name
    if flat_path.exists():
        return str(flat_path)
    raise FileNotFoundError(f"Cannot resolve image path: {image_path}")


def build_tasks(benchmark_dir: Path, prompt_types: list[str]) -> list[dict]:
    tasks = []
    for split in ("test_1", "test_2"):
        split_tasks = load_split_tasks(benchmark_dir, split, prompt_types)
        log(f"Loaded {split}: {len(split_tasks)} tasks")
        tasks.extend(split_tasks)
    return tasks


def create_bidders_config(temperature: float, budget: float, num_bidders: int = 3) -> list[dict]:
    bidders = []
    for idx in range(1, num_bidders + 1):
        bidders.append(
            {
                "name": f"Bidder {idx}",
                "model_name": "qwen3-vl-30b-a3b-instruct",
                "budget": int(budget),
                "desire": "aesthetic_alignment",
                "plan_strategy": "adaptive",
                "temperature": temperature,
                "overestimate_percent": 0,
                "correct_belief": True,
                "enable_learning": False,
            }
        )
    return bidders


def evaluate_task(task: dict, temperature: float, budget_ratio: float, num_runs: int, seed: int) -> dict:
    split = task["split"]
    group_id = task["group_id"]
    prompt_type = task["prompt_type"]
    prompt_text = task["prompt_text"]
    img_paths = task["img_paths"]
    ranks = task["ranks"]
    num_images = len(img_paths)
    budget = budget_ratio * MEAN_PRICE_OVERALL * num_images

    log(
        f"{split}/{group_id}/{prompt_type}: "
        f"{num_images} images, budget={budget:.2f}"
    )
    t0 = time.time()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            items = []
            for idx, img_path in enumerate(img_paths, 1):
                items.append(
                    Item(
                        id=idx,
                        name=f"Image #{idx}",
                        price=1,
                        image_path=img_path,
                        thumbnail_path=img_path,
                        context=prompt_text,
                        real_final_price=MEAN_PRICE_OVERALL,
                    )
                )

            statistics = run_monte_carlo(
                auction_hash=f"g{group_id}_{prompt_type}_{int(time.time())}",
                items_list=items,
                bidders_config_list=create_bidders_config(temperature, budget),
                auctioneer_config={"enable_discount": False, "min_markup_pct": 0.1},
                num_runs=num_runs,
                thread_num=1,
                shuffle=False,
                random_seed=seed,
                log_dir=temp_dir,
            )

        get_qwen_model().clear_cache()

        scores = []
        auction_stats = []
        for idx in range(1, num_images + 1):
            item_name = f"Image #{idx}"
            item_stats = statistics.get(item_name, {"mean": 0.0, "std": 0.0, "median": 0.0})
            mean_val = round(float(item_stats.get("mean", 0.0)), 2)
            std_val = round(float(item_stats.get("std", 0.0)), 2)
            median_val = round(float(item_stats.get("median", item_stats.get("mean", 0.0))), 2)
            scores.append(mean_val)
            auction_stats.append({"mean": mean_val, "std": std_val, "median": median_val})

        pred_best = max(range(num_images), key=lambda i: scores[i])
        pred_worst = min(range(num_images), key=lambda i: scores[i])
        gt_best = ranks.index(min(ranks))
        gt_worst = ranks.index(max(ranks))
        elapsed = round(time.time() - t0, 1)

        result = {
            "split": split,
            "group_id": group_id,
            "prompt_type": prompt_type,
            "prompt_text": prompt_text,
            "img_paths": [Path(p).name for p in img_paths],
            "budget": round(budget, 2),
            "auction_statistics": auction_stats,
            "scores": scores,
            "ranks": ranks,
            "recall_at_1": 1 if pred_best == gt_best else 0,
            "filter_at_1": 1 if pred_worst == gt_worst else 0,
            "inference_time": elapsed,
            "success": True,
        }
        log(
            f"  done {split}/{group_id}: recall={result['recall_at_1']} "
            f"filter={result['filter_at_1']} time={elapsed:.1f}s"
        )
        return result
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        log(f"  error {split}/{group_id}: {exc}")
        import traceback

        traceback.print_exc()
        return {
            "split": split,
            "group_id": group_id,
            "prompt_type": prompt_type,
            "prompt_text": prompt_text,
            "img_paths": [Path(p).name for p in img_paths],
            "scores": [],
            "ranks": ranks,
            "recall_at_1": 0,
            "filter_at_1": 0,
            "inference_time": elapsed,
            "success": False,
            "error": str(exc),
        }


def result_file_for(split: str, prompt_type: str) -> Path:
    return RESULT_DIR / f"{split}_{prompt_type}.json"


def load_existing_results(prompt_types: list[str]) -> dict:
    results = {}
    for split in ("test_1", "test_2"):
        results[split] = {}
        for prompt_type in prompt_types:
            path = result_file_for(split, prompt_type)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    results[split][prompt_type] = json.load(f)
            else:
                results[split][prompt_type] = []
    return results


def upsert_result(result: dict) -> None:
    path = result_file_for(result["split"], result["prompt_type"])
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    filtered = [
        item
        for item in data
        if not (
            item.get("split") == result["split"]
            and item.get("group_id") == result["group_id"]
            and item.get("prompt_type") == result["prompt_type"]
        )
    ]
    filtered.append(result)
    filtered.sort(key=lambda item: item.get("group_id", ""))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)


def compute_metrics(results: dict, prompt_types: list[str]) -> dict:
    metrics = {}
    total_ok = []

    for split in ("test_1", "test_2"):
        split_metrics = {"prompts": {}, "average": {"recall_at_1": 0.0, "filter_at_1": 0.0, "n": 0}}
        valid_prompt_metrics = []

        for prompt_type in prompt_types:
            items = results.get(split, {}).get(prompt_type, [])
            ok_items = [item for item in items if item.get("success")]
            if ok_items:
                recall = sum(item["recall_at_1"] for item in ok_items) / len(ok_items)
                filt = sum(item["filter_at_1"] for item in ok_items) / len(ok_items)
                prompt_metrics = {
                    "recall_at_1": recall,
                    "filter_at_1": filt,
                    "n": len(ok_items),
                }
                valid_prompt_metrics.append(prompt_metrics)
                total_ok.extend(ok_items)
            else:
                prompt_metrics = {"recall_at_1": 0.0, "filter_at_1": 0.0, "n": 0}
            split_metrics["prompts"][prompt_type] = prompt_metrics

        if valid_prompt_metrics:
            split_metrics["average"] = {
                "recall_at_1": sum(m["recall_at_1"] for m in valid_prompt_metrics) / len(valid_prompt_metrics),
                "filter_at_1": sum(m["filter_at_1"] for m in valid_prompt_metrics) / len(valid_prompt_metrics),
                "n": sum(m["n"] for m in valid_prompt_metrics),
            }
        metrics[split] = split_metrics

    if total_ok:
        metrics["overall"] = {
            "recall_at_1": sum(item["recall_at_1"] for item in total_ok) / len(total_ok),
            "filter_at_1": sum(item["filter_at_1"] for item in total_ok) / len(total_ok),
            "n": len(total_ok),
        }
    else:
        metrics["overall"] = {"recall_at_1": 0.0, "filter_at_1": 0.0, "n": 0}

    return metrics


def save_metrics(metrics: dict) -> None:
    with open(RESULT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def main() -> None:
    global RESULT_DIR

    parser = argparse.ArgumentParser(
        description="Auction-arena on ImageReward using qwen3-vl-30b-a3b-instruct"
    )
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(RESULT_DIR))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--budget_ratio", type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument("--num_runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun even if result already exists")
    args = parser.parse_args()

    benchmark_dir = resolve_benchmark_dir(args.benchmark_dir)
    RESULT_DIR = Path(args.output_dir)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    patch_valuator_runtime()
    get_qwen_model()

    prompt_types = list(DEFAULT_PROMPT_TYPES)
    tasks = build_tasks(benchmark_dir, prompt_types)
    log(f"Total tasks: {len(tasks)}")

    existing_results = load_existing_results(prompt_types)
    completed = {
        (item["split"], item["group_id"], item["prompt_type"])
        for split_results in existing_results.values()
        for prompt_results in split_results.values()
        for item in prompt_results
        if item.get("success")
    }

    if args.force:
        remaining = tasks
    else:
        remaining = [
            task
            for task in tasks
            if (task["split"], task["group_id"], task["prompt_type"]) not in completed
        ]

    if args.max_tasks is not None:
        remaining = remaining[: args.max_tasks]

    log(f"Remaining tasks: {len(remaining)}")

    start_time = time.time()
    for idx, task in enumerate(remaining, 1):
        log(
            f"[{idx}/{len(remaining)}] {task['split']}/{task['group_id']}/{task['prompt_type']}"
        )
        result = evaluate_task(
            task,
            temperature=args.temperature,
            budget_ratio=args.budget_ratio,
            num_runs=args.num_runs,
            seed=args.seed,
        )
        upsert_result(result)

        existing_results = load_existing_results(prompt_types)
        save_metrics(compute_metrics(existing_results, prompt_types))

    metrics = compute_metrics(load_existing_results(prompt_types), prompt_types)
    save_metrics(metrics)
    elapsed_minutes = (time.time() - start_time) / 60
    log(f"Finished. Elapsed: {elapsed_minutes:.2f} min")
    log(f"Metrics: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
