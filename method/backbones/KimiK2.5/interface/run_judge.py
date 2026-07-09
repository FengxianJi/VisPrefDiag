#!/usr/bin/python3
"""
Valuator_judge (kimi-k2.5 API) on the ImageReward benchmark.

Default behavior:
- only runs test_1 and test_2
- only runs the original prompt
- uses a single worker by default
- saves results under run_KimiK2.5/Valuator_judge
- flushes per-result outputs immediately after each completed task
"""

import argparse
import base64
import io
import json
import os
import re
import time
from collections import defaultdict
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import requests
from PIL import Image


INTERFACE_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "benchmark" / "ImageReward").exists():
            return candidate
    raise RuntimeError("Could not locate Valuator project root")


PROJECT_ROOT = find_project_root(INTERFACE_DIR)
SERVER_BENCHMARK_ROOT = Path("/mnt/data/user/fengxian.ji/AUCTION/benchmark/ImageReward")
LOCAL_BENCHMARK_ROOT = PROJECT_ROOT / "benchmark" / "ImageReward"
RESULT_DIR = (
    PROJECT_ROOT
    / "show"
    / "experiment"
    / "[ICML][Mechanism-Based Value Elicitation ][2][26015]Human preference sanity "
    / "ImageReward(test)"
    / "ImageReward"
    / "result"
    / "run_KimiK2.5"
    / "Valuator_judge"
)

DEFAULT_API_KEY = ""
DEFAULT_API_BASE_URL = "https://mydamoxing.cn/v1"
DEFAULT_MODEL_NAME = "kimi-k2.5"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_WORKERS = 1
DEFAULT_MAX_RETRIES = 12
DEFAULT_RETRY_DELAY = 20.0
DEFAULT_SPLITS = ("test_1", "test_2")
PROMPT_TYPE = "original"

_SESSION = None
_SESSION_CONFIG = None
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def env_first(names, default=""):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def get_api_key() -> str:
    return env_first(("KIMI_K25_API_KEY", "OPENAI_API_KEY", "MYDAMOXING_API_KEY"), DEFAULT_API_KEY)


def get_api_base_url() -> str:
    return env_first(("KIMI_K25_BASE_URL", "OPENAI_API_BASE"), DEFAULT_API_BASE_URL).rstrip("/")


def get_model_name() -> str:
    return env_first(("KIMI_K25_MODEL", "OPENAI_MODEL_NAME"), DEFAULT_MODEL_NAME)


def resolve_benchmark_root() -> Path:
    for candidate in (SERVER_BENCHMARK_ROOT, LOCAL_BENCHMARK_ROOT):
        if (candidate / "test_1.json").exists() and (candidate / "test_2.json").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find ImageReward benchmark files in either "
        f"{SERVER_BENCHMARK_ROOT} or {LOCAL_BENCHMARK_ROOT}"
    )


BENCHMARK_ROOT = resolve_benchmark_root()


def get_session() -> requests.Session:
    global _SESSION, _SESSION_CONFIG

    config = (get_api_key(), get_api_base_url())
    if not config[0]:
        raise RuntimeError("Missing API key. Set KIMI_K25_API_KEY or pass --api_key.")

    if _SESSION is None or _SESSION_CONFIG != config:
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {config[0]}",
                "Content-Type": "application/json",
            }
        )
        _SESSION = session
        _SESSION_CONFIG = config

    return _SESSION


def image_to_data_url(image_path: str) -> str:
    max_side = 1400
    quality_steps = [80, 65, 50]

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

        payload = None
        for quality in quality_steps:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            candidate = buffer.getvalue()
            if len(candidate) <= 550_000:
                payload = candidate
                break
            payload = candidate

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


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


def extract_first_number(text: str):
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def call_api(
    content,
    *,
    group_id,
    stage,
    temperature,
    max_tokens=768,
    max_retries=DEFAULT_MAX_RETRIES,
    initial_delay=DEFAULT_RETRY_DELAY,
):
    payload = {
        "model": get_model_name(),
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    delay = initial_delay
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = get_session().post(
                f"{get_api_base_url()}/chat/completions",
                data=json.dumps(payload),
                timeout=240,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            response_content = data["choices"][0]["message"]["content"]
            if isinstance(response_content, list):
                return "\n".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in response_content
                ).strip()
            return str(response_content).strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            log(
                f"{group_id} [{stage}] retry {attempt}/{max_retries} after error: {exc}. "
                f"Sleeping {delay:.1f}s"
            )
            time.sleep(delay)
            delay = min(delay * 1.6, 180.0)

    raise RuntimeError(f"{stage} failed after {max_retries} attempts: {last_error}") from last_error


def load_test_data(splits, max_groups_per_split=None):
    log(f"Loading ImageReward benchmark data from {BENCHMARK_ROOT} ...")
    data = {}
    for split in splits:
        split_file = BENCHMARK_ROOT / f"{split}.json"
        with open(split_file, "r", encoding="utf-8") as handle:
            split_items = json.load(handle)

        grouped = defaultdict(list)
        for item in split_items:
            grouped[item["prompt_id"]].append(item)

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


def get_image_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)

    primary = BENCHMARK_ROOT / path
    if primary.exists():
        return str(primary)

    fallback = BENCHMARK_ROOT / path.name
    if fallback.exists():
        return str(fallback)

    return str(primary)


def build_task_list(data, splits):
    tasks = []
    for split in splits:
        for item in data[split]:
            tasks.append(
                {
                    "split": split,
                    "group_id": item["id"],
                    "prompt_type": PROMPT_TYPE,
                    "prompt_text": item["prompt"],
                    "img_paths": [get_image_path(img["image_path"]) for img in item["images"]],
                    "ranks": [img["rank"] for img in item["images"]],
                }
            )
    return tasks


def build_judge_content(prompt_text, image_path):
    prompt = (
        "You are an expert image judge evaluating a generated image.\n\n"
        f'Text prompt: "{prompt_text}"\n\n'
        "Judge the image on a 0-100 scale based on:\n"
        "1. overall visual quality\n"
        "2. alignment with the text prompt\n"
        "3. fidelity and artifact level\n"
        "4. overall preference versus other likely generations for the same prompt\n\n"
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
        for index, image_path in enumerate(img_paths, start=1):
            response = call_api(
                build_judge_content(prompt_text, image_path),
                group_id=group_id,
                stage=f"judge_item_{index}",
                temperature=temperature,
            )
            try:
                result_json = parse_json(response)
                score = float(result_json.get("score", 50))
                reasoning = result_json.get("reasoning", "")
            except Exception:
                recovered = extract_first_number(response)
                score = recovered if recovered is not None else 50.0
                reasoning = response
            score = max(0.0, min(100.0, score))
            scores.append(score)
            reasonings.append(reasoning)

        pred_best = int(np.argmax(scores))
        pred_worst = int(np.argmin(scores))
        gt_best = ranks.index(min(ranks))
        gt_worst = ranks.index(max(ranks))

        return {
            "split": split,
            "group_id": group_id,
            "prompt_type": PROMPT_TYPE,
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
            "prompt_type": PROMPT_TYPE,
            "prompt_text": prompt_text,
            "img_paths": [os.path.basename(path) for path in img_paths],
            "scores": [],
            "ranks": ranks,
            "recall_at_1": 0,
            "filter_at_1": 0,
            "success": False,
            "error": str(exc),
        }


def load_existing_results(result_dir: Path, splits):
    results = {}
    for split in splits:
        results[split] = {PROMPT_TYPE: []}
        result_file = result_dir / f"{split}_{PROMPT_TYPE}.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as handle:
                results[split][PROMPT_TYPE] = json.load(handle)
    return results


def save_result(result_dir: Path, result) -> None:
    result_file = result_dir / f"{result['split']}_{PROMPT_TYPE}.json"
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
            and item.get("prompt_type") == PROMPT_TYPE
        )
    ]
    data.append(result)

    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def compute_metrics(results, splits):
    metrics = {}
    for split in splits:
        ok_entries = [entry for entry in results[split][PROMPT_TYPE] if entry.get("success")]
        if ok_entries:
            recall = float(np.mean([entry["recall_at_1"] for entry in ok_entries]))
            filt = float(np.mean([entry["filter_at_1"] for entry in ok_entries]))
            metrics[split] = {
                "prompts": {PROMPT_TYPE: {"recall_at_1": recall, "filter_at_1": filt, "n": len(ok_entries)}},
                "average": {"recall_at_1": recall, "filter_at_1": filt},
            }
        else:
            metrics[split] = {
                "prompts": {PROMPT_TYPE: {"recall_at_1": 0.0, "filter_at_1": 0.0, "n": 0}},
                "average": {"recall_at_1": 0.0, "filter_at_1": 0.0},
            }

    metrics["overall_average"] = {
        "recall_at_1": float(np.mean([metrics[split]["average"]["recall_at_1"] for split in splits])) if splits else 0.0,
        "filter_at_1": float(np.mean([metrics[split]["average"]["filter_at_1"] for split in splits])) if splits else 0.0,
    }
    return metrics


def flush_metrics(metrics_file: Path, existing_results, splits) -> None:
    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(compute_metrics(existing_results, splits), handle, indent=2, ensure_ascii=False)


def merge_result(existing_results, result) -> None:
    existing_results[result["split"]][PROMPT_TYPE] = [
        entry
        for entry in existing_results[result["split"]][PROMPT_TYPE]
        if not (
            entry.get("split") == result["split"]
            and entry.get("group_id") == result["group_id"]
            and entry.get("prompt_type") == PROMPT_TYPE
        )
    ]
    existing_results[result["split"]][PROMPT_TYPE].append(result)


def main():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="kimi-k2.5 Valuator_judge on ImageReward benchmark")
    parser.add_argument("--api_key", type=str, default=get_api_key())
    parser.add_argument("--api_base_url", type=str, default=get_api_base_url())
    parser.add_argument("--model_name", type=str, default=get_model_name())
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--max_groups_per_split", type=int, default=None)
    parser.add_argument("--result_dir", type=str, default=str(RESULT_DIR))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api_key or set KIMI_K25_API_KEY.")

    os.environ["KIMI_K25_API_KEY"] = args.api_key
    os.environ["KIMI_K25_BASE_URL"] = args.api_base_url.rstrip("/")
    os.environ["KIMI_K25_MODEL"] = args.model_name

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = result_dir / "metrics.json"

    log("=" * 60)
    log("kimi-k2.5 Valuator_judge - ImageReward benchmark")
    log(f"benchmark_root={BENCHMARK_ROOT}")
    log(f"splits={args.splits}")
    log(f"prompt_types={[PROMPT_TYPE]}")
    log(f"num_workers={args.num_workers} temperature={args.temperature}")
    log(f"api_base_url={args.api_base_url.rstrip('/')}")
    log(f"model_name={args.model_name}")
    log(f"result_dir={result_dir}")
    log("=" * 60)

    data = load_test_data(args.splits, max_groups_per_split=args.max_groups_per_split)
    all_tasks = build_task_list(data, args.splits)
    log(f"Total tasks: {len(all_tasks)}")

    existing_results = load_existing_results(result_dir, args.splits)
    completed_keys = {
        (entry["split"], entry["group_id"], entry["prompt_type"])
        for split in args.splits
        for entry in existing_results[split][PROMPT_TYPE]
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
                save_result(result_dir, result)
                merge_result(existing_results, result)
                flush_metrics(metrics_file, existing_results, args.splits)
                completed += 1
                if result.get("success"):
                    success += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(remaining_tasks) - completed) / rate if rate > 0 else 0
                log(f"Progress: {completed}/{len(remaining_tasks)} (success={success}) | {elapsed/60:.1f}min | ETA={eta/60:.1f}min")
    else:
        for task in remaining_tasks:
            result = evaluate_task(task)
            save_result(result_dir, result)
            merge_result(existing_results, result)
            flush_metrics(metrics_file, existing_results, args.splits)
            completed += 1
            if result.get("success"):
                success += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (len(remaining_tasks) - completed) / rate if rate > 0 else 0
            log(f"Progress: {completed}/{len(remaining_tasks)} (success={success}) | {elapsed/60:.1f}min | ETA={eta/60:.1f}min")

    final_metrics = compute_metrics(existing_results, args.splits)
    flush_metrics(metrics_file, existing_results, args.splits)

    log("=" * 60)
    log("FINAL RESULTS")
    for split in args.splits:
        average = final_metrics[split]["average"]
        log(f"{split:>8s} Recall@1={average['recall_at_1']:.4f} Filter@1={average['filter_at_1']:.4f}")
    log(f"Metrics: {metrics_file}")
    log("=" * 60)


if __name__ == "__main__":
    main()
