#!/usr/bin/env python3
"""Dependency-free local web annotator for PrefHardBench.

The app reads the existing benchmark CSV but never exposes its labels to
annotators. New responses are appended as JSONL and can be merged later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import random
import re
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data_annotations.csv"
DEFAULT_IMAGES = ROOT / "images"
DEFAULT_OUTPUT = ROOT / "annotations" / "prefhard_annotations.jsonl"
WEB_DIR = ROOT / "web"
WRITE_LOCK = threading.Lock()
ID_RE = re.compile(r"^[A-Za-z0-9_.:@-]{1,80}$")

REASONS_U = {
    "a1": "Almost no preference-relevant visible difference",
    "a2": "Visible differences, but no clear winner",
    "a3": "Each image is better in a different aspect",
    "a4": "The prompt has multiple reasonable interpretations",
    "a5": "The comparison mainly depends on personal taste",
    "a6": "Both images have comparable defects",
    "a7": "Other difficulty that cannot be specified",
}
REASONS_PREF = {
    "b1": "Prompt alignment clearly determines the choice",
    "b2": "Visual quality clearly determines the choice",
    "b3": "Composition, aesthetics, or style clearly determines the choice",
    "b4": "One side wins after a cross-dimension trade-off",
    "b5": "A visible error clearly disqualifies the other side",
    "b6": "The images are close, but one has a slight overall advantage",
    "b7": "The choice comes mainly from an overall first impression",
}


def read_question_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "question_id" not in (reader.fieldnames or []):
                raise ValueError("question list CSV must contain a question_id column")
            values = [str(row["question_id"]).strip() for row in reader]
    else:
        values = [line.strip().split(",", 1)[0] for line in path.read_text(encoding="utf-8").splitlines()]
    return list(dict.fromkeys(value for value in values if value))


def read_questions(path: Path, image_dir: Path) -> tuple[dict[str, dict], set[str]]:
    questions: dict[str, dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = row["question_id"]
            if question_id in questions:
                continue
            questions[question_id] = {
                "question_id": question_id,
                "prompt_en": row["prompt_en"],
                "prompt_zh": row["prompt_zh"],
                "image_left": row["image_left"],
                "image_right": row["image_right"],
            }
    images = {name.name for name in image_dir.iterdir() if name.is_file()}
    missing = {
        image
        for question in questions.values()
        for image in (question["image_left"], question["image_right"])
        if image not in images
    }
    if missing:
        raise ValueError(f"CSV references missing images, e.g. {sorted(missing)[:3]}")
    return questions, images


def read_annotations(path: Path, image_names: set[str]) -> list[dict]:
    """Read every existing annotation row for the review-only browser."""
    annotations = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 1):
            left = row.get("image_left", "")
            right = row.get("image_right", "")
            if left not in image_names or right not in image_names:
                raise ValueError(f"annotation row {row_number} references a missing image")
            row["record_number"] = row_number
            row["image_left_url"] = f"/images/{left}"
            row["image_right_url"] = f"/images/{right}"
            annotations.append(row)
    return annotations


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_annotator_id(value: object) -> str:
    candidate = str(value or "").strip()
    if not ID_RE.fullmatch(candidate):
        raise ValueError("annotator_id must contain only letters, numbers, _, -, ., :, or @")
    return candidate


def load_completed(output: Path, annotator_id: str) -> set[str]:
    if not output.exists():
        return set()
    completed = set()
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("annotator_id") == annotator_id and row.get("status") == "ok":
                completed.add(str(row.get("question_id")))
    return completed


def choose_question(questions: dict[str, dict], completed: set[str], annotator_id: str, seed: int) -> dict | None:
    available = [question for key, question in questions.items() if key not in completed]
    if not available:
        return None
    stable_seed = f"{seed}:{annotator_id}"
    digest = hashlib.sha256(stable_seed.encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    rng.shuffle(available)
    return available[0]


class AppState:
    def __init__(self, csv_path: Path, image_dir: Path, output: Path, seed: int, limit: int | None, question_list: Path | None):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.output = output
        self.seed = seed
        questions, image_names = read_questions(csv_path, image_dir)
        self.questions = questions
        self.image_names = image_names
        self.annotations = read_annotations(csv_path, image_names)
        if question_list is not None:
            requested = read_question_list(question_list)
            missing = [question_id for question_id in requested if question_id not in questions]
            if missing:
                raise ValueError(f"question list contains unknown IDs, e.g. {missing[:3]}")
            self.questions = {question_id: questions[question_id] for question_id in requested}
        elif limit is not None:
            keys = sorted(questions)[:limit]
            self.questions = {key: questions[key] for key in keys}


class Handler(BaseHTTPRequestHandler):
    server_version = "PrefHardBenchAnnotator/1.0"

    @property
    def state(self) -> AppState:
        return self.server.app_state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_bytes(self, payload: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value: object, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_bytes(payload, "application/json; charset=utf-8", status)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/images/"):
            name = unquote(parsed.path.removeprefix("/images/"))
            if name not in self.state.image_names or Path(name).name != name:
                self.send_response(HTTPStatus.NOT_FOUND)
                self.end_headers()
                return
            image = self.state.image_dir / name
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "image/webp")
            self.send_header("Content-Length", str(image.stat().st_size))
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes((WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path == "/static/app.js":
            self.send_bytes((WEB_DIR / "app.js").read_bytes(), "text/javascript; charset=utf-8")
            return
        if parsed.path == "/static/style.css":
            self.send_bytes((WEB_DIR / "style.css").read_bytes(), "text/css; charset=utf-8")
            return
        if parsed.path.startswith("/images/"):
            name = unquote(parsed.path.removeprefix("/images/"))
            if name not in self.state.image_names or Path(name).name != name:
                self.send_json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
                return
            image = self.state.image_dir / name
            content_type = mimetypes.guess_type(name)[0] or "image/webp"
            self.send_bytes(image.read_bytes(), content_type)
            return
        if parsed.path == "/api/next":
            try:
                annotator_id = safe_annotator_id(parse_qs(parsed.query).get("annotator_id", [""])[0])
                completed = load_completed(self.state.output, annotator_id)
                question = choose_question(self.state.questions, completed, annotator_id, self.state.seed)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if question is None:
                self.send_json({"done": True, "completed": len(completed), "total": len(self.state.questions)})
                return
            swap = bool(int(hashlib.sha256(f"{self.state.seed}:{annotator_id}:{question['question_id']}".encode()).hexdigest()[-1], 16) % 2)
            shown = dict(question)
            shown["shown_left"] = question["image_right"] if swap else question["image_left"]
            shown["shown_right"] = question["image_left"] if swap else question["image_right"]
            shown["shown_order"] = "swapped" if swap else "original"
            shown["image_left_url"] = f"/images/{shown['shown_left']}"
            shown["image_right_url"] = f"/images/{shown['shown_right']}"
            self.send_json({"done": False, "completed": len(completed), "total": len(self.state.questions), "question": shown})
            return
        if parsed.path == "/api/annotations":
            try:
                index = int(parse_qs(parsed.query).get("index", ["0"])[0])
            except ValueError:
                self.send_json({"error": "index must be an integer"}, HTTPStatus.BAD_REQUEST)
                return
            if not 0 <= index < len(self.state.annotations):
                self.send_json({"error": "annotation index out of range"}, HTTPStatus.NOT_FOUND)
                return
            self.send_json({
                "index": index,
                "total": len(self.state.annotations),
                "annotation": self.state.annotations[index],
            })
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/submit":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 20000:
                raise ValueError("request is too large")
            data = json.loads(self.rfile.read(length))
            record = self.validate_record(data)
            self.state.output.parent.mkdir(parents=True, exist_ok=True)
            with WRITE_LOCK, self.state.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            self.send_json({"ok": True, "question_id": record["question_id"]})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def validate_record(self, data: object) -> dict:
        if not isinstance(data, dict):
            raise ValueError("JSON object expected")
        annotator_id = safe_annotator_id(data.get("annotator_id"))
        question_id = str(data.get("question_id", ""))
        if question_id not in self.state.questions:
            raise ValueError("unknown question_id")
        completed = load_completed(self.state.output, annotator_id)
        if question_id in completed:
            raise ValueError("this annotator already submitted this question")
        preference = str(data.get("preference", ""))
        if preference not in {"A", "B", "U"}:
            raise ValueError("preference must be A, B, or U")
        difficulty = int(data.get("difficulty"))
        confidence = int(data.get("confidence"))
        if difficulty not in range(1, 6) or confidence not in range(1, 6):
            raise ValueError("difficulty and confidence must be integers from 1 to 5")
        reason = data.get("reason", [])
        if not isinstance(reason, list):
            raise ValueError("reason must be a list")
        allowed = set(REASONS_U if preference == "U" else REASONS_PREF)
        if not reason or not set(reason).issubset(allowed):
            raise ValueError("select at least one valid reason for this preference")
        if preference != "U" and len(reason) != 1:
            raise ValueError("A/B requires exactly one primary reason")
        question = self.state.questions[question_id]
        shown_order = str(data.get("shown_order", ""))
        if shown_order not in {"original", "swapped"}:
            raise ValueError("invalid shown_order")
        shown_left = question["image_left"] if shown_order == "original" else question["image_right"]
        shown_right = question["image_right"] if shown_order == "original" else question["image_left"]
        note = str(data.get("note", "")).strip()
        if len(note) > 1000:
            raise ValueError("note is limited to 1000 characters")
        chosen = "" if preference == "U" else (shown_left if preference == "A" else shown_right)
        rejected = "" if preference == "U" else (shown_right if preference == "A" else shown_left)
        return {
            "status": "ok",
            "annotation_id": uuid.uuid4().hex,
            "annotator_id": annotator_id,
            "submitted_at": utc_now(),
            "question_id": question_id,
            "prompt_en": question["prompt_en"],
            "prompt_zh": question["prompt_zh"],
            "original_image_left": question["image_left"],
            "original_image_right": question["image_right"],
            "shown_order": shown_order,
            "shown_left": shown_left,
            "shown_right": shown_right,
            "preference": preference,
            "chosen": chosen,
            "rejected": rejected,
            "difficulty": difficulty,
            "confidence": confidence,
            "reason": "|".join(reason),
            "note": note,
            "elapsed_seconds": max(0, float(data.get("elapsed_seconds", 0))),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local PrefHardBench annotation web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--limit", type=int, help="Use only the first N question IDs for a pilot")
    parser.add_argument("--question-list", type=Path, help="Text file or CSV containing selected question_id values")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = AppState(args.csv.resolve(), args.images.resolve(), args.output.resolve(), args.seed, args.limit, args.question_list.resolve() if args.question_list else None)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app_state = state  # type: ignore[attr-defined]
    print(f"PrefHardBench annotator: http://{args.host}:{args.port}")
    print(f"questions={len(state.questions)} output={state.output}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
