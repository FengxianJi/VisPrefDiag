#!/usr/bin/python3
"""Single-agent-Budgeted on ImageReward using remote qwen3-vl-30b-a3b-instruct."""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


INTERFACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTERFACE_DIR.parents[2]
EXPERIMENT_DIR = REPO_ROOT / "show" / "experiment" / "[ICML][][0][0107]Main"

for path in (INTERFACE_DIR, EXPERIMENT_DIR):
    sys.path.insert(0, str(path))

from sigelLLM import SigelLLMAuction, _parse_json


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
RESULT_DIR = RESULT_ROOT / "Single-agent-Budgeted"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_BUDGET_RATIO = 1.2
DEFAULT_PROMPT_TYPES = ["original"]
MEAN_PRICE_OVERALL = 96.8565


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


def resolve_image_path(benchmark_dir: Path, image_path: str) -> str:
    direct_path = benchmark_dir / image_path
    if direct_path.exists():
        return str(direct_path)
    flat_path = benchmark_dir / Path(image_path).name
    if flat_path.exists():
        return str(flat_path)
    raise FileNotFoundError(f"Cannot resolve image path: {image_path}")


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


def build_tasks(benchmark_dir: Path, prompt_types: list[str]) -> list[dict]:
    tasks = []
    for split in ("test_1", "test_2"):
        split_tasks = load_split_tasks(benchmark_dir, split, prompt_types)
        log(f"Loaded {split}: {len(split_tasks)} tasks")
        tasks.extend(split_tasks)
    return tasks


def evaluate_task(task: dict, agent: SigelLLMAuction, budget_ratio: float) -> dict:
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
        agent.reset()

        contact_sheet = agent.create_contact_sheet(img_paths, thumbnail_size=(300, 300))
        ranking_prompt = (
            "You are an image quality evaluator.\n\n"
            f"The following {num_images} images were generated from this text prompt:\n"
            f"\"{prompt_text}\"\n\n"
            f"You have a budget of {budget:.2f} to allocate across these {num_images} images.\n"
            f"Above is a contact sheet showing all {num_images} candidate images.\n\n"
            "Please analyze these images and provide your preference ranking from most "
            "to least preferred, considering the following aspects:\n"
            "1. Overall quality: How good is the image in terms of composition, aesthetics, and visual appeal?\n"
            "2. Text-image alignment: How well does the image match the text prompt?\n"
            "3. Fidelity: Are the details clear, accurate, and free of artifacts?\n\n"
            "Return your response in the following JSON format:\n"
            "{\n"
            '    "preference_ranking": [best, ..., worst],\n'
            '    "reasoning": "Your reasoning for this ranking"\n'
            "}\n\n"
            f"IMPORTANT: The ranking array must contain all numbers from 1 to {num_images}, each exactly once. "
            "The first number is the BEST image.\n"
            "Return only valid JSON without any additional text or markdown formatting."
        )

        raw_ranking = agent._call_local(ranking_prompt, images=[contact_sheet])
        ranking_result = _parse_json(raw_ranking)
        preference_ranking = ranking_result.get("preference_ranking", list(range(1, num_images + 1)))
        ranking_reasoning = ranking_result.get("reasoning", "")

        if len(preference_ranking) != num_images or set(preference_ranking) != set(range(1, num_images + 1)):
            log(f"  WARNING invalid ranking {preference_ranking}, using default")
            preference_ranking = list(range(1, num_images + 1))

        agent.preference_sequence = preference_ranking

        remaining_budget = budget
        bid_prices = []
        bid_reasonings = []

        for idx, img_path in enumerate(img_paths, 1):
            num_remaining = num_images - idx + 1
            preference_info = ""
            if agent.preference_sequence and idx in agent.preference_sequence:
                rank = agent.preference_sequence.index(idx) + 1
                preference_info = f"\nBased on your initial preference ranking, this image was ranked #{rank}."

            history_text = agent._format_decision_history()
            pricing_prompt = (
                "You are an image quality evaluator.\n\n"
                f"The following image was generated from this text prompt:\n"
                f"\"{prompt_text}\"\n\n"
                f"Total Budget: {budget:.2f}\n"
                f"Remaining Budget: {remaining_budget:.2f}\n"
                f"Current Image: #{idx}\n"
                f"Remaining Images to evaluate: {num_remaining} (including this one)\n"
                f"{preference_info}\n\n"
                f"{history_text}\n\n"
                "Above shows the image you need to evaluate now.\n\n"
                "Please provide your bid price for this image, considering the following aspects:\n"
                "1. Overall quality: How good is the image in terms of composition, aesthetics, and visual appeal?\n"
                "2. Text-image alignment: How well does the image match the text prompt?\n"
                "3. Fidelity: Are the details clear, accurate, and free of artifacts?\n"
                "4. Your initial preference for this image\n"
                "5. Your remaining budget\n"
                "6. Number of remaining images\n"
                "7. Your previous decisions\n"
                "8. Balancing preference with budget constraints\n\n"
                f"You can bid any amount from 0 to your remaining budget ({remaining_budget:.2f}).\n\n"
                "Return your response in the following JSON format:\n"
                '{"bid_price": <number>, "reasoning": "<your reasoning>"}\n\n'
                "IMPORTANT: Return only valid JSON without any additional text or markdown formatting."
            )

            raw_bid = agent._call_local(pricing_prompt, images=[img_path])
            bid_result = _parse_json(raw_bid)
            bid_price = float(bid_result.get("bid_price", 0))
            bid_price = max(0.0, min(bid_price, remaining_budget))

            agent.decision_history.append(
                {
                    "item_index": idx,
                    "bid_price": bid_price,
                    "reasoning": bid_result.get("reasoning", ""),
                    "remaining_budget_before": remaining_budget,
                    "remaining_budget_after": remaining_budget - bid_price,
                }
            )

            bid_prices.append(bid_price)
            bid_reasonings.append(bid_result.get("reasoning", ""))
            remaining_budget -= bid_price

        elapsed = round(time.time() - t0, 1)
        scores = bid_prices
        pred_best = max(range(num_images), key=lambda i: scores[i])
        pred_worst = min(range(num_images), key=lambda i: scores[i])
        gt_best = ranks.index(min(ranks))
        gt_worst = ranks.index(max(ranks))

        result = {
            "split": split,
            "group_id": group_id,
            "prompt_type": prompt_type,
            "prompt_text": prompt_text,
            "img_paths": [Path(p).name for p in img_paths],
            "budget": round(budget, 2),
            "preference_ranking": preference_ranking,
            "ranking_reasoning": ranking_reasoning,
            "scores": scores,
            "reasonings": bid_reasonings,
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
        description="Single-agent-Budgeted on ImageReward using qwen3-vl-30b-a3b-instruct"
    )
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(RESULT_DIR))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--budget_ratio", type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun even if result already exists")
    args = parser.parse_args()

    benchmark_dir = resolve_benchmark_dir(args.benchmark_dir)
    RESULT_DIR = Path(args.output_dir)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    agent = SigelLLMAuction(
        temperature=args.temperature,
        model_name="qwen3-vl-30b-a3b-instruct",
    )

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
        result = evaluate_task(task, agent=agent, budget_ratio=args.budget_ratio)
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
