#!/usr/bin/python3
"""
Single-agentLLM_score ([f]gemini-3-flash-preview) - ImageReward benchmark.

Default behavior:
- only runs `test_1` and `test_2`
- only runs the original prompt
- single worker
- retries on temporary API failures
- saves results under `run_gemini3flash/Single-agentLLM_score`
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from collections import defaultdict
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import requests

INTERFACE_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "benchmark" / "ImageReward").exists() and (candidate / "method" / "Gemini-3-flash").exists():
            return candidate
    raise RuntimeError("Could not locate Valuator project root")


PROJECT_ROOT = find_project_root(INTERFACE_DIR)
BENCHMARK_CANDIDATES = [
    Path("/mnt/data/user/fengxian.ji/AUCTION/benchmark/ImageReward"),
    PROJECT_ROOT / "benchmark" / "ImageReward",
]
BENCHMARK_PATH = PROJECT_ROOT / "benchmark" / "ImageReward"
RESULT_DIR = (
    PROJECT_ROOT
    / "show"
    / "experiment"
    / "[ICML][Mechanism-Based Value Elicitation ][2][26015]Human preference sanity "
    / "ImageReward(test)"
    / "ImageReward"
    / "result"
    / "run_gemini3flash"
    / "Single-agentLLM_score"
)

API_BASE_URL = "https://new.lemonapi.site/v1"
API_KEY = os.getenv("GEMINI_FLASH_API_KEY", os.getenv("OPENAI_API_KEY", ""))
MODEL_NAME = "[L][V]gemini-3-flash-preview"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_WORKERS = 1
DEFAULT_MAX_RETRIES = 12
DEFAULT_RETRY_DELAY = 20.0
DEFAULT_SPLITS = ("test_1", "test_2")

_SESSION = None
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
FALLBACK_SCORE = 50.0


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def resolve_benchmark_path() -> Path:
    for candidate in BENCHMARK_CANDIDATES:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Cannot locate ImageReward benchmark directory. Checked: "
        + ", ".join(str(path) for path in BENCHMARK_CANDIDATES)
    )


def get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(
            {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
        )
    return _SESSION


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"
    with open(image_path, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = _JSON_OBJ_RE.search(text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from model output: {text[:300]}")


def is_prohibited_content_error(exc: Exception | str) -> bool:
    return "PROHIBITED_CONTENT" in str(exc)


def call_api(content, *, group_id, stage, temperature, max_tokens=768, max_retries=DEFAULT_MAX_RETRIES, initial_delay=DEFAULT_RETRY_DELAY):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    delay = initial_delay
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = get_session().post(
                f"{API_BASE_URL}/chat/completions",
                data=json.dumps(payload),
                timeout=240,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            message_content = data["choices"][0]["message"]["content"]
            if isinstance(message_content, list):
                return "\n".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message_content
                ).strip()
            return str(message_content).strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            log(f"{group_id} [{stage}] retry {attempt}/{max_retries} after error: {exc}. Sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 1.6, 180.0)
    raise RuntimeError(f"{stage} failed after {max_retries} attempts: {last_error}") from last_error


def load_test_data(splits, max_groups_per_split=None):
    log("Loading ImageReward benchmark data...")
    data = {}
    for split in splits:
        with open(BENCHMARK_PATH / f"{split}.json", "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["prompt_id"]].append(row)
        prompt_ids = sorted(grouped)
        if max_groups_per_split is not None:
            prompt_ids = prompt_ids[:max_groups_per_split]
        split_data = []
        for prompt_id in prompt_ids:
            split_data.append(
                {
                    "id": prompt_id,
                    "prompt": grouped[prompt_id][0]["prompt"],
                    "images": grouped[prompt_id],
                }
            )
        data[split] = split_data
        log(f"  {split}: {len(split_data)} prompt groups")
    return data


def get_image_path(path_str):
    direct_path = BENCHMARK_PATH / path_str
    if direct_path.exists():
        return str(direct_path)
    flat_path = BENCHMARK_PATH / os.path.basename(path_str)
    if flat_path.exists():
        return str(flat_path)
    raise FileNotFoundError(f"Cannot resolve image path: {path_str}")


def build_task_list(data, splits):
    tasks = []
    for split in splits:
        for item in data[split]:
            tasks.append(
                {
                    "split": split,
                    "group_id": item["id"],
                    "prompt_type": "original",
                    "prompt_text": item["prompt"],
                    "img_paths": [get_image_path(img["image_path"]) for img in item["images"]],
                    "ranks": [img["rank"] for img in item["images"]],
                }
            )
    return tasks


def build_score_content(prompt_text, image_path):
    prompt = (
        "You are evaluating a generated image.\n\n"
        f'Text prompt: "{prompt_text}"\n\n'
        "Rate the image on a 0-100 scale based on:\n"
        "1. overall visual quality\n"
        "2. alignment with the text prompt\n"
        "3. fidelity and artifact level\n\n"
        "Return only JSON in this format:\n"
        '{"score": <number>, "reasoning": "<short explanation>"}'
    )
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
    ]


def evaluate_task(args):
    split, group_id, prompt_text, img_paths, ranks, temperature = args
    try:
        scores = []
        reasonings = []
        for idx, image_path in enumerate(img_paths, start=1):
            try:
                response = call_api(
                    build_score_content(prompt_text, image_path),
                    group_id=group_id,
                    stage=f"score_item_{idx}",
                    temperature=temperature,
                )
                result_json = parse_json(response)
                score = float(result_json.get("score", FALLBACK_SCORE))
                score = max(0.0, min(100.0, score))
                scores.append(score)
                reasonings.append(result_json.get("reasoning", ""))
            except Exception as exc:  # noqa: BLE001
                if not is_prohibited_content_error(exc):
                    raise
                scores.append(FALLBACK_SCORE)
                reasonings.append("fallback_due_prohibited_content")

        pred_best = int(np.argmax(scores))
        pred_worst = int(np.argmin(scores))
        gt_best = ranks.index(min(ranks))
        gt_worst = ranks.index(max(ranks))

        return {
            "split": split,
            "group_id": group_id,
            "prompt_type": "original",
            "prompt_text": prompt_text,
            "img_paths": [os.path.basename(path) for path in img_paths],
            "scores": scores,
            "reasonings": reasonings,
            "ranks": ranks,
            "recall_at_1": 1 if pred_best == gt_best else 0,
            "filter_at_1": 1 if pred_worst == gt_worst else 0,
            "success": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "split": split,
            "group_id": group_id,
            "prompt_type": "original",
            "prompt_text": prompt_text,
            "img_paths": [os.path.basename(path) for path in img_paths],
            "scores": [],
            "ranks": ranks,
            "recall_at_1": 0,
            "filter_at_1": 0,
            "success": False,
            "error": str(exc),
        }


def load_existing_results(splits):
    results = {}
    for split in splits:
        results[split] = {"original": []}
        result_file = RESULT_DIR / f"{split}_original.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as handle:
                results[split]["original"] = json.load(handle)
    return results


def save_result(result):
    result_file = RESULT_DIR / f"{result['split']}_original.json"
    if result_file.exists():
        with open(result_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = []
    data = [
        item
        for item in data
        if not (
            item.get("split") == result["split"]
            and item.get("group_id") == result["group_id"]
            and item.get("prompt_type") == "original"
        )
    ]
    data.append(result)
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def compute_metrics(results, splits):
    metrics = {}
    for split in splits:
        ok_entries = [entry for entry in results[split]["original"] if entry.get("success")]
        if ok_entries:
            recall = float(np.mean([entry["recall_at_1"] for entry in ok_entries]))
            filt = float(np.mean([entry["filter_at_1"] for entry in ok_entries]))
            metrics[split] = {
                "prompts": {"original": {"recall_at_1": recall, "filter_at_1": filt, "n": len(ok_entries)}},
                "average": {"recall_at_1": recall, "filter_at_1": filt},
            }
        else:
            metrics[split] = {
                "prompts": {"original": {"recall_at_1": 0.0, "filter_at_1": 0.0, "n": 0}},
                "average": {"recall_at_1": 0.0, "filter_at_1": 0.0},
            }
    metrics["overall_average"] = {
        "recall_at_1": float(np.mean([metrics[split]["average"]["recall_at_1"] for split in splits])) if splits else 0.0,
        "filter_at_1": float(np.mean([metrics[split]["average"]["filter_at_1"] for split in splits])) if splits else 0.0,
    }
    return metrics


def main():
    global BENCHMARK_PATH
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Single-agentLLM_score on ImageReward benchmark")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--max_groups_per_split", type=int, default=None)
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = RESULT_DIR / "metrics.json"
    BENCHMARK_PATH = resolve_benchmark_path()

    log("=" * 60)
    log("Single-agentLLM_score — ImageReward benchmark")
    log(f"splits={args.splits}")
    log(f"num_workers={args.num_workers} temperature={args.temperature}")
    log(f"benchmark_path={BENCHMARK_PATH}")
    log(f"result_dir={RESULT_DIR}")
    log("=" * 60)

    data = load_test_data(args.splits, max_groups_per_split=args.max_groups_per_split)
    all_tasks = build_task_list(data, args.splits)
    log(f"Total tasks: {len(all_tasks)}")

    existing_results = load_existing_results(args.splits)
    completed_keys = {
        (entry["split"], entry["group_id"], entry["prompt_type"])
        for split in args.splits
        for entry in existing_results[split]["original"]
        if entry.get("success")
    }
    remaining_tasks = [
        (
            task["split"],
            task["group_id"],
            task["prompt_text"],
            task["img_paths"],
            task["ranks"],
            args.temperature,
        )
        for task in all_tasks
        if (task["split"], task["group_id"], task["prompt_type"]) not in completed_keys
    ]
    log(f"Remaining tasks: {len(remaining_tasks)}")

    start_time = time.time()
    completed = 0
    success = 0

    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            iterator = pool.imap_unordered(evaluate_task, remaining_tasks)
            for result in iterator:
                save_result(result)
                existing_results[result["split"]]["original"] = [
                    entry
                    for entry in existing_results[result["split"]]["original"]
                    if not (
                        entry.get("split") == result["split"]
                        and entry.get("group_id") == result["group_id"]
                        and entry.get("prompt_type") == "original"
                    )
                ]
                existing_results[result["split"]]["original"].append(result)
                completed += 1
                if result.get("success"):
                    success += 1
                if completed % 10 == 0 or completed == len(remaining_tasks):
                    with open(metrics_file, "w", encoding="utf-8") as handle:
                        json.dump(compute_metrics(existing_results, args.splits), handle, indent=2, ensure_ascii=False)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(remaining_tasks) - completed) / rate if rate > 0 else 0
                log(f"Progress: {completed}/{len(remaining_tasks)} (success={success}) | {elapsed/60:.1f}min | ETA={eta/60:.1f}min")
    else:
        for task in remaining_tasks:
            result = evaluate_task(task)
            save_result(result)
            existing_results[result["split"]]["original"] = [
                entry
                for entry in existing_results[result["split"]]["original"]
                if not (
                    entry.get("split") == result["split"]
                    and entry.get("group_id") == result["group_id"]
                    and entry.get("prompt_type") == "original"
                )
            ]
            existing_results[result["split"]]["original"].append(result)
            completed += 1
            if result.get("success"):
                success += 1
            if completed % 10 == 0 or completed == len(remaining_tasks):
                with open(metrics_file, "w", encoding="utf-8") as handle:
                    json.dump(compute_metrics(existing_results, args.splits), handle, indent=2, ensure_ascii=False)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(remaining_tasks) - completed) / rate if rate > 0 else 0
            log(f"Progress: {completed}/{len(remaining_tasks)} (success={success}) | {elapsed/60:.1f}min | ETA={eta/60:.1f}min")

    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(compute_metrics(existing_results, args.splits), handle, indent=2, ensure_ascii=False)

    final_metrics = compute_metrics(existing_results, args.splits)
    log("=" * 60)
    log("FINAL RESULTS")
    for split in args.splits:
        avg = final_metrics[split]["average"]
        log(f"{split:>8s} Recall@1={avg['recall_at_1']:.4f} Filter@1={avg['filter_at_1']:.4f}")
    log(f"Metrics: {metrics_file}")
    log("=" * 60)


if __name__ == "__main__":
    main()
