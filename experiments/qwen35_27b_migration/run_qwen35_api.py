#!/usr/bin/env python3
"""Qwen3.5-27B API evaluation for the three table-2 protocols.

Uses the two DashScope-compatible keys from qwenkey.txt in round-robin order,
with retry and append-only JSONL checkpoints. Keys are never printed.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(os.environ.get("AUCTION_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "PrefHardBench"
IMAGE_ROOT = DATA / "images"
KEY_FILE = ROOT / "qwenkey.txt"
MODEL = "qwen3.5-27b"


def rows() -> list[dict]:
    grouped = {}
    with (DATA / "data_annotations.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row["question_id"], row)
    return list(grouped.values())


def image_data_url(path: Path) -> str:
    mime = "image/webp" if path.suffix.lower() == ".webp" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def prompt_for(protocol: str, prompt: str) -> str:
    if protocol == "single_score":
        return ("Rate only the image shown for this text-to-image prompt. Score from 0 to 100, "
                "where 100 means excellent prompt alignment and visual quality.\n"
                f"Prompt: {prompt}\nReply with exactly JSON and nothing else: "
                '{"score": 0}.')
    if protocol == "pairwise":
        return ("Compare image A and image B for this text-to-image prompt. Judge prompt alignment, "
                "visual quality, composition, aesthetics, and visible defects.\n"
                f"Prompt: {prompt}\nReply with exactly JSON and nothing else: "
                '{"winner":"A"} or {"winner":"B"} or {"winner":"U"}.')
    return ("Compare image A and image B for this text-to-image prompt using evidence first. "
            "Return exactly five labeled lines and nothing else:\n"
            "1. Prompt alignment evidence: <brief evidence>\n"
            "2. Visual quality evidence: <brief evidence>\n"
            "3. Composition/style evidence: <brief evidence>\n"
            "4. Conflicting evidence: <brief evidence or None>\n"
            "5. Final preference: <A, B, or Tie>\n"
            f"Prompt: {prompt}")


def parse_json(text: str) -> dict:
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict): return obj
        except json.JSONDecodeError: pass
    raise ValueError(f"invalid JSON response: {text[:300]!r}")


def parse_winner(text: str) -> str:
    obj = parse_json(text)
    winner = str(obj.get("winner", obj.get("choice", ""))).upper().strip()
    if winner in {"TIE", "U", "UNCERTAIN", "NONE"}: return "U"
    if winner not in {"A", "B"}: raise ValueError(f"invalid winner: {obj!r}")
    return winner


def parse_score(text: str) -> float:
    obj = parse_json(text)
    value = float(obj["score"])
    if not 0 <= value <= 100: raise ValueError(f"invalid score: {value}")
    return value


def parse_evidence(text: str) -> str:
    m = re.search(r"(?:final preference|final answer|winner)\s*[:\-]?\s*(?:image\s*)?([AB]|Tie)\b", text, re.I)
    if not m:
        m = re.search(r"(?:prefer|choose|select|pick|favor|favour)\s+(?:image\s+)?([AB])\b", text, re.I)
    if not m: raise ValueError(f"invalid evidence response: {text[:300]!r}")
    value = m.group(1).upper()
    return "U" if value == "TIE" else value


class ApiPool:
    def __init__(self, key_file: Path, base_url: str):
        keys = [line.strip() for line in key_file.read_text().splitlines() if line.strip()]
        if not keys: raise RuntimeError(f"no API keys found in {key_file}")
        self.clients = [OpenAI(api_key=k, base_url=base_url, timeout=360) for k in keys]
        self.index = 0

    def call(self, content: list[dict]) -> str:
        last = None
        for attempt in range(6):
            client = self.clients[self.index % len(self.clients)]
            self.index += 1
            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": content}],
                    extra_body={"enable_thinking": True},
                    stream=True,
                    temperature=0,
                    max_tokens=768,
                )
                answer = []
                for chunk in stream:
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None): answer.append(delta.content)
                text = "".join(answer).strip()
                if text: return text
                raise ValueError("empty content response")
            except Exception as exc:
                last = exc
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"API failed after retries: {last!r}")


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["single_score", "pairwise", "evidence"], required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-items", type=int)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--key-file", type=Path, default=KEY_FILE)
    ap.add_argument("--base-url", default="https://maas.qianwenaiapi.com/compatible-mode/v1")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / ("single_score_predictions.jsonl" if args.protocol == "single_score" else "pairwise_predictions.jsonl" if args.protocol == "pairwise" else "evidence_reasoning_predictions.jsonl")
    data = rows()[:args.max_items] if args.max_items else rows()
    done = set()
    if output.exists():
        for line in output.open(encoding="utf-8"):
            try:
                rec=json.loads(line)
                if rec.get("success"): done.add(rec["question_id"])
            except json.JSONDecodeError: pass
    MODEL = args.model
    pool = ApiPool(args.key_file, args.base_url)
    print(json.dumps({"model": MODEL, "protocol": args.protocol, "items": len(data), "completed": len(done), "output": str(output)}), flush=True)
    with output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(data, 1):
            if row["question_id"] in done: continue
            started=time.time(); left=IMAGE_ROOT/row["image_left"]; right=IMAGE_ROOT/row["image_right"]
            try:
                def content(order):
                    items=[]
                    for label, path in order:
                        items.append({"type":"text", "text":f"Image {label}:"})
                        items.append({"type":"image_url", "image_url":{"url":image_data_url(path)}})
                    items.append({"type":"text", "text":prompt_for(args.protocol,row["prompt_en"])})
                    return items
                if args.protocol == "single_score":
                    score_a=parse_score(pool.call(content([("A",left)])))
                    score_b=parse_score(pool.call(content([("B",right)])))
                    winner="A" if score_a>score_b else "B" if score_b>score_a else "U"
                    rec={"success":True,"protocol":"Single-Score","model":MODEL,"question_id":row["question_id"],"difficulty":row["difficulty"],"score_a":score_a,"score_b":score_b,"winner_identity":winner}
                elif args.protocol == "pairwise":
                    w1=parse_winner(pool.call(content([("A",left),("B",right)])))
                    w2=parse_winner(pool.call(content([("B",right),("A",left)])))
                    map2={"A":"B","B":"A","U":"U"}
                    rec={"success":True,"model":MODEL,"question_id":row["question_id"],"difficulty":row["difficulty"],"original":{"winner_identity":w1},"swapped":{"winner_identity":map2[w2]},"winner_identity":w1,"order_consistent":w1==map2[w2]}
                else:
                    w1=parse_evidence(pool.call(content([("A",left),("B",right)])))
                    w2=parse_evidence(pool.call(content([("B",right),("A",left)])))
                    map2={"A":"B","B":"A","U":"U"}
                    rec={"success":True,"protocol":"Evidence-First Reasoning","model":MODEL,"question_id":row["question_id"],"difficulty":row["difficulty"],"original":{"winner_identity":w1},"swapped":{"winner_identity":map2[w2]},"winner_identity":w1,"order_consistent":w1==map2[w2]}
                rec["seconds"]=round(time.time()-started,3)
            except Exception as exc:
                rec={"success":False,"model":MODEL,"protocol":args.protocol,"question_id":row["question_id"],"difficulty":row["difficulty"],"error":repr(exc),"seconds":round(time.time()-started,3)}
            handle.write(json.dumps(rec,ensure_ascii=False)+"\n"); handle.flush()
            print(f"[{index}/{len(data)}] {row['question_id']} {'OK' if rec['success'] else rec['error']}",flush=True)


if __name__ == "__main__": main()
