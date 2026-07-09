#!/usr/bin/python3
"""Valuator_judge on ImageReward using remote qwen3-vl-30b-a3b-instruct."""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


INTERFACE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INTERFACE_DIR.parents[2]

sys.path.insert(0, str(INTERFACE_DIR))

from model_loader import get_qwen_model


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
RESULT_DIR = RESULT_ROOT / "Valuator_judge"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT_TYPES = ["original"]


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


def build_scoring_messages(prompt_text: str, image_path: str) -> list[dict]:
    scoring_prompt = (
        "You are an image quality evaluator.\n\n"
        "The following image was generated from this text prompt:\n"
        f"\"{prompt_text}\"\n\n"
        "Please rate your overall preference for this generated image on a scale of 0-100, "
        "considering the following aspects:\n"
        "1. Overall quality: composition, aesthetics, and visual appeal.\n"
        "2. Text-image alignment: how well the image matches the prompt.\n"
        "3. Fidelity: details, accuracy, and absence of artifacts.\n\n"
        "Scoring guidelines:\n"
        "- 90-100: Exceptional\n"
        "- 70-89: Good\n"
        "- 50-69: Average\n"
        "- 30-49: Below average\n"
        "- 0-29: Poor\n\n"
        "Return ONLY a single number between 0 and 100. "
        "Do not include any explanation or extra text.\n\n"
        "Your score:"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": scoring_prompt},
            ],
        }
    ]


def parse_score(text: str) -> float:
    numbers = re.findall(r"\d+\.?\d*", text.strip())
    if not numbers:
        return 50.0
    return max(0.0, min(100.0, float(numbers[0])))


def evaluate_task(task: dict, model, temperature: float) -> dict:
    split = task["split"]
    group_id = task["group_id"]
    prompt_type = task["prompt_type"]
    prompt_text = task["prompt_text"]
    img_paths = task["img_paths"]
    ranks = task["ranks"]
    num_images = len(img_paths)

    log(f"{split}/{group_id}/{prompt_type}: {num_images} images")
    t0 = time.time()

    try:
        scores = []
        reasonings = []
        for img_path in img_paths:
            raw = model.generate(
                build_scoring_messages(prompt_text=prompt_text, image_path=img_path),
                max_new_tokens=64,
                temperature=temperature,
            )
            reasonings.append(raw.strip())
            scores.append(parse_score(raw))

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
            "scores": scores,
            "reasonings": reasonings,
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
            "reasonings": [],
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
        description="Valuator_judge on ImageReward using qwen3-vl-30b-a3b-instruct"
    )
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(RESULT_DIR))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Rerun even if result already exists")
    args = parser.parse_args()

    benchmark_dir = resolve_benchmark_dir(args.benchmark_dir)
    RESULT_DIR = Path(args.output_dir)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model = get_qwen_model()
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
        result = evaluate_task(task, model=model, temperature=args.temperature)
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
