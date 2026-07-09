#!/usr/bin/python3
"""
Single-agent_pair ([f]gemini-3-flash-preview).

Runs the EmergingArt benchmark with a single ranking call per group.
"""

import argparse
import base64
import glob
import json
import mimetypes
import os
import re
import sys
import time
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np
import requests

INTERFACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INTERFACE_DIR.parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "show" / "experiment" / "[ICML][][0][0107]Main"
sys.path.insert(0, str(EXPERIMENT_DIR))
import calculate_matrix

DATA_DIR = PROJECT_ROOT / "benchmark" / "EmergingArt" / "data"
RESULT_DIR = PROJECT_ROOT / "method" / "Gemini-3-flash" / "result" / "Single-agent_pair"

API_BASE_URL = "https://new.lemonapi.site/v1"
API_KEY = os.getenv("GEMINI_FLASH_API_KEY", os.getenv("OPENAI_API_KEY", ""))
MODEL_NAME = "[L][V]gemini-3-flash-preview"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_WORKERS = 1
DEFAULT_MAX_RETRIES = 12
DEFAULT_RETRY_DELAY = 20.0

_SESSION = None
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def extract_ranking_list(text: str):
    match = re.search(r'"ranking"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if not match:
        return None
    numbers = [int(num) for num in re.findall(r"\d+", match.group(1))]
    return numbers if numbers else None


def extract_text_field(text: str, field_name: str):
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"([^"]*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_ranking_response(text: str, num_items: int):
    try:
        parsed = parse_json(text)
    except ValueError:
        parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}

    ranking = parsed.get("ranking")
    if not isinstance(ranking, list):
        ranking = extract_ranking_list(text)
    if not isinstance(ranking, list):
        ranking = list(range(1, num_items + 1))

    reasoning = parsed.get("reasoning", "")
    if not reasoning:
        reasoning = extract_text_field(text, "reasoning") or "fallback_due_parse_error"

    return {"ranking": ranking, "reasoning": reasoning}


def call_api(content, *, group_id, stage, temperature, max_tokens=1024, max_retries=DEFAULT_MAX_RETRIES, initial_delay=DEFAULT_RETRY_DELAY):
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
            log(f"{group_id} [{stage}] retry {attempt}/{max_retries} after error: {exc}. Sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 1.6, 180.0)
    raise RuntimeError(f"{stage} failed after {max_retries} attempts: {last_error}") from last_error


def load_group_items(json_file: Path, group_id: int):
    with open(json_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    group_data = [item for item in data if item["group_id"] == group_id]
    items = []
    for item in group_data:
        figure_dir = Path(item["folder_path"]) / "figure"
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP"):
            image_files.extend(glob.glob(str(figure_dir / ext)))
        image_files.sort()
        items.append({"folder_id": item["folder_id"], "price": item["price"], "images": image_files})
    return items


def build_ranking_content(items):
    content = [
        {
            "type": "text",
            "text": (
                "You are ranking several artworks from BEST to WORST.\n"
                "Consider artistic quality, composition, originality, and likely market appeal.\n"
                "Return only JSON in this format:\n"
                '{"ranking": [1, 2, ...], "reasoning": "..."}'
            ),
        }
    ]
    for idx, item in enumerate(items, start=1):
        content.append({"type": "text", "text": f"Artwork #{idx} (folder_id={item['folder_id']})"})
        for image_path in item["images"][:1]:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    return content


def evaluate_single_group(args):
    json_file, group_id, temperature = args
    try:
        items = load_group_items(Path(json_file), group_id)
        if not items:
            return {"group_id": group_id, "items": [], "success": False, "error": "No items"}

        response = call_api(
            build_ranking_content(items),
            group_id=group_id,
            stage="rank_items",
            temperature=temperature,
        )
        result_json = parse_ranking_response(response, len(items))
        ranking = result_json.get("ranking", [])
        if len(ranking) != len(items) or set(ranking) != set(range(1, len(items) + 1)):
            ranking = list(range(1, len(items) + 1))
        reasoning = result_json.get("reasoning", "")

        items_result = []
        for idx, item in enumerate(items, start=1):
            item_rank = ranking.index(idx) + 1 if idx in ranking else len(items)
            items_result.append(
                {
                    "name": str(item["folder_id"]),
                    "folder_id": str(item["folder_id"]),
                    "true_price": item["price"],
                    "pred_score": float(len(items) - item_rank + 1),
                    "rank": item_rank,
                    "rel": calculate_matrix.calculate_rel(item["price"]),
                    "reasoning": f"Rank {item_rank}/{len(items)} | {reasoning[:120]}",
                }
            )
        return {"group_id": group_id, "items": items_result, "ranking": ranking, "success": True}
    except Exception as exc:  # noqa: BLE001
        return {"group_id": group_id, "items": [], "success": False, "error": str(exc)}


def evaluate_model(temperature, output_dir, num_workers=DEFAULT_NUM_WORKERS):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {"model_name": MODEL_NAME, "method": "Single-agent_pair", "temperature": temperature, "datasets": {}}
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
        args_list = [(str(json_file), group_id, temperature) for group_id in group_ids]
        dataset_results = []
        with Pool(processes=num_workers) as pool:
            for idx, result in enumerate(pool.imap(evaluate_single_group, args_list), start=1):
                status = "OK" if result["success"] else f"FAIL: {result.get('error', '')}"
                log(f"{dataset_name} [{idx}/{len(group_ids)}] Group {result['group_id']} -> {status}")
                if result["success"]:
                    dataset_results.append(result)
        all_results["datasets"][dataset_name] = dataset_results
    result_file = output_dir / "Single-agent_pair.json"
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2, ensure_ascii=False)
    log(f"Results saved: {result_file}")
    return all_results


def main():
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser(description="Single-agent_pair")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()
    evaluate_model(args.temperature, RESULT_DIR, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
