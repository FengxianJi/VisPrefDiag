#!/usr/bin/env python3
"""Aggregate repeated PrefHardBench web annotations by question ID."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
            else:
                raise ValueError(f"{path}:{line_number}: non-ok record")
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["question_id"])].append(row)
    output = []
    for question_id, items in sorted(groups.items()):
        difficulties = [int(row["difficulty"]) for row in items]
        confidences = [int(row["confidence"]) for row in items]
        preferences = Counter(row["preference"] for row in items)
        reasons = Counter(code for row in items for code in row["reason"].split("|") if code)
        majority = sorted(preferences.items(), key=lambda item: (-item[1], item[0]))[0]
        output.append(
            {
                "question_id": question_id,
                "prompt_en": items[0]["prompt_en"],
                "prompt_zh": items[0]["prompt_zh"],
                "image_left": items[0]["original_image_left"],
                "image_right": items[0]["original_image_right"],
                "n_annotations": len(items),
                "difficulty_mean": round(mean(difficulties), 6),
                "difficulty_median": median(difficulties),
                "difficulty_std": round(pstdev(difficulties), 6) if len(difficulties) > 1 else 0.0,
                "confidence_mean": round(mean(confidences), 6),
                "count_A": preferences.get("A", 0),
                "count_B": preferences.get("B", 0),
                "count_U": preferences.get("U", 0),
                "majority_preference": majority[0],
                "majority_fraction": round(majority[1] / len(items), 6),
                "reason_counts": json.dumps(dict(sorted(reasons.items())), ensure_ascii=False),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(
            f"Input annotation file does not exist: {args.input}\n"
            "Start PrefHardBench/web_app.py with the same --output path and "
            "submit at least one annotation before aggregating."
        )
    if args.input.stat().st_size == 0:
        raise SystemExit(
            f"Input annotation file is empty: {args.input}\n"
            "Submit at least one annotation before aggregating."
        )
    rows = read_jsonl(args.input)
    result = aggregate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(result[0]) if result else ["question_id"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps({"annotations": len(rows), "questions": len(result), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
