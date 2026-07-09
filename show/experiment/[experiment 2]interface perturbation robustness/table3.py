#!/usr/bin/env python3
"""
Generate table3.tex: Combined 3-panel table showing:
  Panel 1: Normal (default prompt & order)
  Panel 2: Prompt perturbations (averaged over rephrasings)
  Panel 3: Order perturbations (averaged over shuffles)

Metrics:
  PairAcc under the human pair difficulty split from table13.py:
    Hard: level gap d = 1
    Mid : level gap d = 2
    Easy: level gap d >= 3

Data sources:
  Panel 1: table3/origin/ (_detail files for baselines, test_original.json for agents)
  Panel 2: table3/different_instruction/ (_detail per prompt, test_*.json for agents)
  Panel 3: table3/different_order/ (disorder_1..4, averaged; baselines from origin)
"""

import json
from pathlib import Path


# ──────────────────── paths ────────────────────
BASE_DIR = Path(__file__).resolve().parent / "table3"
ORIGIN_DIR = BASE_DIR / "origin"
INSTR_DIR = BASE_DIR / "different_instruction"
ORDER_DIR = BASE_DIR / "different_order"
OUTPUT_PATH = Path(__file__).resolve().parent / "table3.tex"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_GROUP_JSONS = [
    PROJECT_ROOT / "benchmark" / "ImageReward" / "test_1.json",
    PROJECT_ROOT / "benchmark" / "ImageReward" / "test_2.json",
]

# ──────────────────── method definitions ────────────────────
BASELINE_DETAIL = {
    "CLIPScore": "CLIPScore_detail.json",
    "NIMA": "NIMA_detail.json",
    "HPSv2": "HPSv2_detail.json",
    "PickScore": "PickScore_detail.json",
    "ImageReward": "ImageReward_detail.json",
    "VQAscore": "VQAscore_detail.json",
}

QALIGN_MODES = [("IQA", "Q-Align (IQA)"), ("IAA", "Q-Align (IAA)")]

AGENT_DIRS = {
    "Single-agent (Pair)": "Single-agent_pair",
    "Budgeted Single-agent": "Single-agent-Budgeted",
    "Single-agent (LLM)": "Single-agentLLM_score",
    "Single-agent (Price)": "Single-agent_price",
    "Valuator Judge": "Valuator_judge",
    "Valuator": "Auction-arena",
}

BASELINES = [
    "CLIPScore",
    "NIMA",
    "HPSv2",
    "PickScore",
    "ImageReward",
    "Q-Align (IQA)",
    "Q-Align (IAA)",
    "VQAscore",
]
AGENTS = [
    "Single-agent (Pair)",
    "Budgeted Single-agent",
    "Single-agent (LLM)",
    "Single-agent (Price)",
    "Valuator Judge",
    "Valuator",
]
ALL_METHODS = BASELINES + AGENTS

PROMPT_VARIANTS = ["original", "alternative_1", "alternative_2", "alternative_3", "alternative_4"]
DISORDERS = ["disorder_1", "disorder_2", "disorder_3", "disorder_4"]
SPLITS = ["Hard", "Mid", "Easy"]


# ──────────────────── difficulty split helpers ────────────────────
def _compress_ranks_to_levels(gt_ranks):
    uniq = sorted(set(gt_ranks))
    rank_to_level = {rank: idx + 1 for idx, rank in enumerate(uniq)}
    return [rank_to_level[rank] for rank in gt_ranks]


def _difficulty_from_gap(gap):
    if gap == 1:
        return "Hard"
    if gap == 2:
        return "Mid"
    return "Easy"


def build_benchmark_group_map():
    grouped_rows = {}
    for json_path in BENCHMARK_GROUP_JSONS:
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            gid = row["prompt_id"]
            grouped_rows.setdefault(gid, []).append(row)

    benchmark_group_map = {}
    for gid, rows in grouped_rows.items():
        sorted_rows = sorted(rows, key=lambda row: Path(row["image_path"]).name)
        order = [Path(row["image_path"]).name for row in sorted_rows]
        ranks = [row["rank"] for row in sorted_rows]
        benchmark_group_map[gid] = {
            "order": order,
            "rank_by_name": {name: rank for name, rank in zip(order, ranks)},
        }
    return benchmark_group_map


def _align_group_to_benchmark(group, gt_key, benchmark_group_map):
    scores = group.get("scores", [])
    gt_ranks = group.get(gt_key, [])
    gid = group.get("group_id")
    img_paths = group.get("img_paths", [])

    if gid and img_paths and len(scores) == len(img_paths):
        group_info = benchmark_group_map.get(gid)
        if group_info:
            score_by_name = {Path(path).name: score for path, score in zip(img_paths, scores)}
            order = group_info["order"]
            rank_by_name = group_info["rank_by_name"]
            if all(name in score_by_name and name in rank_by_name for name in order):
                return [score_by_name[name] for name in order], [rank_by_name[name] for name in order]

    return scores, gt_ranks


# ──────────────────── metric computation ────────────────────
def _empty_metrics():
    return {split: None for split in SPLITS}


def compute_pairacc_by_split(groups, gt_key, benchmark_group_map):
    pair_acc_sum = {split: 0.0 for split in SPLITS}
    pair_count = {split: 0 for split in SPLITS}

    for group in groups:
        scores, gt_ranks = _align_group_to_benchmark(group, gt_key, benchmark_group_map)
        if not scores or not gt_ranks or len(scores) != len(gt_ranks):
            continue

        levels = _compress_ranks_to_levels(gt_ranks)
        n = len(scores)
        for i in range(n):
            for j in range(i + 1, n):
                if gt_ranks[i] == gt_ranks[j]:
                    continue

                split = _difficulty_from_gap(abs(levels[i] - levels[j]))
                pair_count[split] += 1

                if gt_ranks[i] < gt_ranks[j]:
                    correct = scores[i] > scores[j]
                else:
                    correct = scores[j] > scores[i]
                pair_acc_sum[split] += 1.0 if correct else 0.0

    metrics = _empty_metrics()
    for split in SPLITS:
        if pair_count[split]:
            metrics[split] = pair_acc_sum[split] / pair_count[split]
    return metrics


def _avg_split_metrics(metric_list):
    avg = _empty_metrics()
    for split in SPLITS:
        vals = [m[split] for m in metric_list if m.get(split) is not None]
        if vals:
            avg[split] = sum(vals) / len(vals)
    return avg


def _safe_load(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_split_agent(path: Path):
    data = _safe_load(path)
    if data:
        return data

    name = path.stem
    parent = path.parent
    suffix = path.suffix
    p1 = parent / (name.replace("test_", "test_1_", 1) + suffix)
    p2 = parent / (name.replace("test_", "test_2_", 1) + suffix)
    if p1.exists() and p2.exists():
        d1 = _safe_load(p1)
        d2 = _safe_load(p2)
        if d1 and d2:
            return d1 + d2
    return None


# ──────────────────── panel loaders ────────────────────
def load_panel1(benchmark_group_map):
    results = {}

    for name, fname in BASELINE_DETAIL.items():
        data = _safe_load(ORIGIN_DIR / fname)
        if data:
            groups = data.get("test", {}).get("details", [])
            if groups:
                results[name] = compute_pairacc_by_split(groups, "gt_ranks", benchmark_group_map)

    for suffix, disp in QALIGN_MODES:
        data = _safe_load(ORIGIN_DIR / f"{suffix}_detail.json")
        if data:
            groups = data.get("test", {}).get("details", [])
            if groups:
                results[disp] = compute_pairacc_by_split(groups, "gt_ranks", benchmark_group_map)

    for name, dirname in AGENT_DIRS.items():
        data = _load_split_agent(ORIGIN_DIR / dirname / "test_original.json")
        if data:
            results[name] = compute_pairacc_by_split(data, "ranks", benchmark_group_map)

    return results


def load_panel2(benchmark_group_map):
    results = {}

    for name, fname in BASELINE_DETAIL.items():
        data = _safe_load(INSTR_DIR / fname)
        if data:
            test_prompts = data.get("test", {}).get("prompts", {})
            ms = []
            for pname in PROMPT_VARIANTS:
                details = test_prompts.get(pname, {}).get("details", [])
                if details:
                    ms.append(compute_pairacc_by_split(details, "gt_ranks", benchmark_group_map))
            if ms:
                results[name] = _avg_split_metrics(ms)

    for suffix, disp in QALIGN_MODES:
        data = _safe_load(INSTR_DIR / f"{suffix}_detail.json")
        if data:
            test_prompts = data.get("test", {}).get("prompts", {})
            ms = []
            for pname in PROMPT_VARIANTS:
                details = test_prompts.get(pname, {}).get("details", [])
                if details:
                    ms.append(compute_pairacc_by_split(details, "gt_ranks", benchmark_group_map))
            if ms:
                results[disp] = _avg_split_metrics(ms)

    for name, dirname in AGENT_DIRS.items():
        ms = []
        for pname in PROMPT_VARIANTS:
            data = _load_split_agent(INSTR_DIR / dirname / f"test_{pname}.json")
            if data:
                ms.append(compute_pairacc_by_split(data, "ranks", benchmark_group_map))
        if ms:
            results[name] = _avg_split_metrics(ms)

    return results


def load_panel3(benchmark_group_map):
    results = {}

    for name, dirname in AGENT_DIRS.items():
        ms = []
        for dname in DISORDERS:
            data = _safe_load(ORDER_DIR / dname / dirname / "test_original.json")
            if data:
                ms.append(compute_pairacc_by_split(data, "ranks", benchmark_group_map))
        if ms:
            results[name] = _avg_split_metrics(ms)

    for name, fname in BASELINE_DETAIL.items():
        if name not in results:
            data = _safe_load(ORIGIN_DIR / fname)
            if data:
                groups = data.get("test", {}).get("details", [])
                if groups:
                    results[name] = compute_pairacc_by_split(groups, "gt_ranks", benchmark_group_map)

    for suffix, disp in QALIGN_MODES:
        if disp not in results:
            data = _safe_load(ORIGIN_DIR / f"{suffix}_detail.json")
            if data:
                groups = data.get("test", {}).get("details", [])
                if groups:
                    results[disp] = compute_pairacc_by_split(groups, "gt_ranks", benchmark_group_map)

    return results


# ──────────────────── formatting helpers ────────────────────
def fmt(value):
    if value is None:
        return "--"
    return f"{value * 100:.2f}"


# ──────────────────── LaTeX generation ────────────────────
def build_panel_latex(panel_data, title):
    bests = {}
    for split in SPLITS:
        vals = [panel_data[m][split] for m in ALL_METHODS if m in panel_data and panel_data[m][split] is not None]
        bests[split] = max(vals) if vals else None

    def fmt_bold(method, split):
        value = panel_data.get(method, {}).get(split)
        text = fmt(value)
        if value is not None and bests[split] is not None and abs(value - bests[split]) < 1e-12:
            return f"\\textbf{{{text}}}"
        return text

    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{title}}}}}\\\\")
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{Method} & \textbf{Hard}$\uparrow$ & \textbf{Mid}$\uparrow$ & \textbf{Easy}$\uparrow$ \\"
    )
    lines.append(r"\midrule")

    for method in BASELINES:
        if method not in panel_data:
            continue
        cols = " & ".join(fmt_bold(method, split) for split in SPLITS)
        lines.append(f"{method:<28} & {cols} \\\\")

    lines.append(r"\midrule")

    for method in AGENTS:
        if method not in panel_data:
            continue
        cols = " & ".join(fmt_bold(method, split) for split in SPLITS)
        lines.append(f"{method:<28} & {cols} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def generate_table(p1, p2, p3):
    panel1 = build_panel_latex(p1, "Normal (default prompt \\& order)")
    panel2 = build_panel_latex(p2, "Prompt perturbations (rephrasings)")
    panel3 = build_panel_latex(p3, "Order perturbations (shuffle)")

    latex = r"""\begin{table*}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.7pt}
\renewcommand{\arraystretch}{0.9}

\begin{minipage}[t]{0.32\textwidth}
\centering
""" + panel1 + r"""
\end{minipage}\hfill
\begin{minipage}[t]{0.32\textwidth}
\centering
""" + panel2 + r"""
\end{minipage}\hfill
\begin{minipage}[t]{0.32\textwidth}
\centering
""" + panel3 + r"""
\end{minipage}

\caption{PairAcc by difficulty under interface perturbations on ImageReward (test).
Hard, Mid, and Easy follow Table~\ref{tab:table13}. Results are shown for normal, prompt perturbations, and order perturbations. Best per column in \textbf{bold}.}
\label{tab:imagereward_threepanel}
\end{table*}
"""
    return latex


# ──────────────────── main ────────────────────
def main():
    print("=" * 70)
    print("Table 3 — Combined 3-panel table (Hard / Mid / Easy PairAcc)")
    print("=" * 70)

    benchmark_group_map = build_benchmark_group_map()

    print("\nPanel 1 (Normal)...")
    p1 = load_panel1(benchmark_group_map)
    for method in ALL_METHODS:
        if method in p1:
            d = p1[method]
            print(
                f"  {method:<28} "
                f"Hard={fmt(d['Hard']):>6}  Mid={fmt(d['Mid']):>6}  Easy={fmt(d['Easy']):>6}"
            )

    print("\nPanel 2 (Prompt perturbation avg)...")
    p2 = load_panel2(benchmark_group_map)
    for method in ALL_METHODS:
        if method in p2:
            d = p2[method]
            print(
                f"  {method:<28} "
                f"Hard={fmt(d['Hard']):>6}  Mid={fmt(d['Mid']):>6}  Easy={fmt(d['Easy']):>6}"
            )

    print("\nPanel 3 (Order perturbation avg)...")
    p3 = load_panel3(benchmark_group_map)
    for method in ALL_METHODS:
        if method in p3:
            d = p3[method]
            print(
                f"  {method:<28} "
                f"Hard={fmt(d['Hard']):>6}  Mid={fmt(d['Mid']):>6}  Easy={fmt(d['Easy']):>6}"
            )

    print("\nGenerating LaTeX...")
    latex = generate_table(p1, p2, p3)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"\n✓ Written to {OUTPUT_PATH}")
    print("\n" + "=" * 70)
    print(latex)
    print("=" * 70)


if __name__ == "__main__":
    main()
