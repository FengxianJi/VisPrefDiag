#!/usr/bin/python3
"""
Valuator / Auction-arena (Gemini 3 Flash API).

This runner is self-contained and does not depend on the legacy Valuator
runtime. It simulates a lightweight 3-bidder auction by asking the same model
to produce bidder-specific bid prices for each item image.
"""

import argparse
import base64
import glob
import io
import json
import mimetypes
import os
import random
import re
import time
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import requests

INTERFACE_DIR = Path(__file__).resolve().parent
METHOD_DIR = INTERFACE_DIR.parent
PROJECT_ROOT = INTERFACE_DIR.parents[2]

DATA_DIR = PROJECT_ROOT / "benchmark" / "EmergingArt" / "data"
RESULT_DIR = METHOD_DIR / "result" / "Auction-arena"

API_BASE_URL = "https://new.lemonapi.site/v1"
API_KEY = os.getenv("GEMINI_FLASH_API_KEY", os.getenv("OPENAI_API_KEY", ""))
MODEL_NAME = "[L][V]gemini-3-flash-preview"

DEFAULT_TEMPERATURE = 0.2
DEFAULT_BUDGET_RATIO = 1.2
DEFAULT_NUM_RUNS = 1
DEFAULT_NUM_WORKERS = 6
DEFAULT_MAX_RETRIES = 12
DEFAULT_RETRY_DELAY = 20.0

BIDDER_PERSONAS = [
    "You favor balanced composition, color harmony, and polished execution.",
    "You favor originality, bold style choices, and memorable visual impact.",
    "You favor marketability, broad appeal, and clear craftsmanship.",
]

_SESSION = None
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def log(message: str):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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

    raise ValueError(f"Cannot parse JSON from model output: {text[:200]}")


def extract_numeric_field(text: str, field_name: str):
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))
    return None


def extract_text_field(text: str, field_name: str):
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_bid_response(text: str, fallback_bid_price: float):
    try:
        parsed = parse_json(text)
    except ValueError:
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    bid_price = parsed.get("bid_price")
    if bid_price is None:
        bid_price = extract_numeric_field(text, "bid_price")
    if bid_price is None:
        bid_price = fallback_bid_price

    reasoning = parsed.get("reasoning", "")
    if not reasoning:
        reasoning = extract_text_field(text, "reasoning") or "fallback_due_parse_error"

    return {"bid_price": bid_price, "reasoning": reasoning}


def image_to_data_url(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/jpeg"
    with open(image_path, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def call_chat(
    prompt: str,
    image_path: str | None,
    temperature: float,
    max_tokens: int = 512,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    content = [{"type": "text", "text": prompt}]
    if image_path:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(image_path)},
            }
        )

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
                timeout=180,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                return "\n".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ).strip()
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_retries:
                break
            log(f"call_chat retry {attempt}/{max_retries} after error: {exc}. Sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 1.6, 180.0)
    raise RuntimeError(f"Gemini API call failed: {last_error}")


def bidder_prompt(context: str | None, budget: float, bidder_idx: int, run_idx: int):
    context_line = f"Evaluation context: {context}\n\n" if context else ""
    return (
        "You are participating in an art auction.\n"
        f"Bidder profile: {BIDDER_PERSONAS[bidder_idx]}\n"
        f"{context_line}"
        f"Your total budget for this item group is ${budget:.2f}.\n"
        f"This is independent simulation run #{run_idx + 1}.\n\n"
        "Estimate how much you would bid for the artwork shown.\n"
        "Return only JSON in this format:\n"
        '{"bid_price": <number>, "reasoning": "<short explanation>"}'
    )


def run_item_auction(image_path: str, context: str | None, budget: float, temperature: float, num_runs: int, seed: int):
    bids_per_run = []
    reasoning_per_run = []

    for run_idx in range(num_runs):
        run_bids = []
        run_reasoning = []
        for bidder_idx in range(len(BIDDER_PERSONAS)):
            random.seed(seed + run_idx * 10 + bidder_idx)
            response = call_chat(
                bidder_prompt(context, budget, bidder_idx, run_idx),
                image_path=image_path,
                temperature=temperature,
            )
            parsed = parse_bid_response(response, fallback_bid_price=budget / max(len(BIDDER_PERSONAS), 1))
            bid_price = float(parsed.get("bid_price", 0))
            bid_price = max(0.0, min(bid_price, budget))
            run_bids.append(bid_price)
            run_reasoning.append(parsed.get("reasoning", ""))

        bids_per_run.append(float(np.mean(run_bids)))
        reasoning_per_run.append(run_reasoning)

    stats = {
        "mean": float(np.mean(bids_per_run)),
        "std": float(np.std(bids_per_run)),
        "median": float(np.median(bids_per_run)),
        "per_run_scores": bids_per_run,
        "per_run_reasoning": reasoning_per_run,
    }
    return stats


def calculate_rel(value):
    return float(np.log(1 + value))


def load_group_items(json_file: Path, group_id: int):
    with open(json_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    group_data = [item for item in data if item["group_id"] == group_id]
    items = []
    for item_data in group_data:
        folder_path = item_data["folder_path"]
        figure_dir = os.path.join(folder_path, "figure")
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP"):
            image_files.extend(glob.glob(os.path.join(figure_dir, ext)))
        image_files.sort()
        items.append(
            {
                "folder_id": item_data["folder_id"],
                "price": item_data["price"],
                "image_path": image_files[0] if image_files else None,
            }
        )
    return items


def evaluate_single_group(args):
    json_file, group_id, budget_ratio, num_runs, temperature, seed = args
    try:
        items = load_group_items(Path(json_file), group_id)
        if not items:
            return {"group_id": group_id, "items": [], "success": False, "error": "No items"}

        total_real_price = sum(item["price"] for item in items)
        budget = total_real_price * budget_ratio

        results = []
        for item in items:
            if not item["image_path"]:
                continue
            auction_stats = run_item_auction(
                image_path=item["image_path"],
                context=None,
                budget=budget,
                temperature=temperature,
                num_runs=num_runs,
                seed=seed,
            )
            results.append(
                {
                    "name": str(item["folder_id"]),
                    "folder_id": str(item["folder_id"]),
                    "true_price": item["price"],
                    "pred_score": auction_stats["mean"],
                    "rel": calculate_rel(item["price"]),
                    "reasoning": "; ".join(auction_stats["per_run_reasoning"][0]) if auction_stats["per_run_reasoning"] else "",
                }
            )

        return {"group_id": group_id, "items": results, "success": True}
    except Exception as exc:  # noqa: BLE001
        return {"group_id": group_id, "items": [], "success": False, "error": str(exc)}


def evaluate_model(temperature, output_dir, budget_ratio, num_runs, seed, test_mode=False, num_workers=DEFAULT_NUM_WORKERS):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {
        "model_name": MODEL_NAME,
        "method": "Auction-arena",
        "temperature": temperature,
        "budget_ratio": budget_ratio,
        "num_runs": num_runs,
        "datasets": {},
    }

    dataset_files = [
        ("Low-price", DATA_DIR / "Low-price.json"),
        ("Mid-price", DATA_DIR / "Mid-price.json"),
        ("High-price", DATA_DIR / "High-price.json"),
        ("Near-Tie-low", DATA_DIR / "Near-Tie-low.json"),
        ("Near-Tie-high", DATA_DIR / "Near-Tie-high.json"),
        ("Overall", DATA_DIR / "Overall.json"),
    ]

    for dataset_name, json_file in dataset_files:
        if not json_file.exists():
            continue
        with open(json_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        group_ids = sorted({item["group_id"] for item in data})
        if test_mode:
            group_ids = group_ids[:3]

        args_list = [
            (str(json_file), group_id, budget_ratio, num_runs, temperature, seed)
            for group_id in group_ids
        ]

        dataset_results = []
        with Pool(processes=num_workers) as pool:
            for idx, result in enumerate(pool.imap(evaluate_single_group, args_list), start=1):
                status = "OK" if result["success"] else f"FAIL: {result.get('error', '')}"
                log(f"{dataset_name} [{idx}/{len(group_ids)}] Group {result['group_id']} -> {status}")
                if result["success"]:
                    dataset_results.append(result)

        all_results["datasets"][dataset_name] = dataset_results

    result_file = output_dir / "Auction-arena.json"
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)
    log(f"Results saved: {result_file}")
    return all_results


def main():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Gemini-3-Flash Auction-arena")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--budget_ratio", type=float, default=DEFAULT_BUDGET_RATIO)
    parser.add_argument("--num_runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()

    evaluate_model(
        args.temperature,
        RESULT_DIR,
        args.budget_ratio,
        args.num_runs,
        args.seed,
        test_mode=args.test,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
