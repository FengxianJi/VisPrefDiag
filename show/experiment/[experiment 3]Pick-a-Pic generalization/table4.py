#!/usr/bin/env python3
"""
Generate table4.tex: Pick-a-Pic pairwise PairAcc by approximate difficulty split.

Unlike table13.py, Pick-a-Pic only provides binary image pairs rather than
group-wise rankings over 4 images, so we cannot reproduce the exact
level-gap-based partition d(i, j)=|level_i-level_j|.

We therefore use the closest analogous split driven by Pick-a-Pic scores:
1. Compute PickScore score gap for every pair:
       Delta s = |pickscore_0 - pickscore_1|
2. Sort all unique ranking_ids by Delta s ascending.
3. Split the global list into three contiguous bins:
       Hard = bottom tertile
       Mid  = middle tertile
       Easy = top tertile

Reported metric per split:
    - PairAcc

The table only shows Overall / Hard / Mid / Easy PairAcc.
"""

import csv
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SHOW_DIR = THIS_DIR.parent
BASE_DIR = THIS_DIR / "table5"
SAMPLE_DIR = THIS_DIR / "sample"
OUTPUT_PATH = THIS_DIR / "table4.tex"

SEEDS = [42, 2024, 7777]
PICKSCORE_FILE = BASE_DIR / "PickScore_v1.json"

BASELINE_MODELS = [
    ("CLIPScore", "CLIPScore.json", "clipscore_0", "clipscore_1"),
    ("NIMA", "NIMA.json", "nima_mean_0", "nima_mean_1"),
    ("HPSv2", "HPSv2.json", "hpsv2_0", "hpsv2_1"),
    ("PickScore", "PickScore_v1.json", "pickscore_0", "pickscore_1"),
    ("ImageReward", "ImageReward.json", "imagereward_0", "imagereward_1"),
    ("RichHF", "RichHF.json", "overall_0", "overall_1"),
    ("Q-Align", "Q-Align.json", "iqa_score_0", "iqa_score_1"),
    ("LLaVA-Reward", "LLaVA-Reward.json", "reward_score_0", "reward_score_1"),
    ("VQAScore", "VQAscore.json", "vqascore_0", "vqascore_1"),
]

AGENT_MODELS = [
    ("Single-agent (Pair)", "Single-agent_pair", None, None),
    ("Single-agent", "Single-agent_price", "price_0", "price_1"),
    ("Single-agent (LLM)", "Single-agentLLM_score", "score_0", "score_1"),
    ("Budgeted Single-agent", "Single-agent-Budgeted", "price_0", "price_1"),
    ("Valuator Judge", "Valuator_judge", "score_0", "score_1"),
    ("Valuator", "Auction-arena", "price_0", "price_1"),
]


def load_global_difficulty_bins():
    """Return global Overall/Hard/Mid/Easy ranking_id bins from PickScore deltas."""
    with open(PICKSCORE_FILE, "r", encoding="utf-8") as f:
        pickscore_data = json.load(f)

    deltas_by_rid = {}
    for row in pickscore_data:
        ranking_id = row.get("ranking_id")
        score_0 = row.get("pickscore_0")
        score_1 = row.get("pickscore_1")
        if ranking_id is None or score_0 is None or score_1 is None:
            continue
        ranking_id = int(ranking_id)
        if ranking_id in deltas_by_rid:
            continue
        deltas_by_rid[ranking_id] = abs(float(score_0) - float(score_1))

    items = sorted(deltas_by_rid.items(), key=lambda x: (x[1], x[0]))
    total = len(items)
    bin_size = total // 3
    hard_ids = {ranking_id for ranking_id, _ in items[:bin_size]}
    mid_ids = {ranking_id for ranking_id, _ in items[bin_size: 2 * bin_size]}
    easy_ids = {ranking_id for ranking_id, _ in items[2 * bin_size:]}
    overall_ids = {ranking_id for ranking_id, _ in items}
    return {
        "Overall": overall_ids,
        "Hard": hard_ids,
        "Mid": mid_ids,
        "Easy": easy_ids,
        "total": total,
        "sizes": {
            "Overall": len(overall_ids),
            "Hard": len(hard_ids),
            "Mid": len(mid_ids),
            "Easy": len(easy_ids),
        },
    }


def load_seed_meta(seed):
    csv_path = SAMPLE_DIR / f"pickapic_seed{seed}_n500_meta.csv"
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_ranking_id_to_index():
    with open(BASE_DIR / "CLIPScore.json", "r", encoding="utf-8") as f:
        clipscore_data = json.load(f)

    rid_to_idx = {}
    for idx, row in enumerate(clipscore_data):
        ranking_id = row.get("ranking_id")
        if ranking_id is None:
            continue
        ranking_id = int(ranking_id)
        if ranking_id not in rid_to_idx:
            rid_to_idx[ranking_id] = idx
    return rid_to_idx


def count_sampled_split_sizes(seed_meta, split_ids):
    counts = {split: 0 for split in ("Overall", "Hard", "Mid", "Easy")}
    for rows in seed_meta.values():
        for meta in rows:
            ranking_id = int(meta["ranking_id"])
            if ranking_id in split_ids["Overall"]:
                counts["Overall"] += 1
            for split in ("Hard", "Mid", "Easy"):
                if ranking_id in split_ids[split]:
                    counts[split] += 1
                    break
    return counts


def load_method_entries(method_dir, seed):
    json_path = BASE_DIR / method_dir / f"seed{seed}.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(entry["row_idx"]): entry for entry in data}


def gt_pair_ranks(label_0, label_1):
    if label_0 == label_1:
        return None
    return [1, 2] if label_0 > label_1 else [2, 1]


def _pair_acc_from_scores(gt_ranks, score_0, score_1):
    if gt_ranks[0] == 1:
        return 1.0 if score_0 > score_1 else 0.0
    return 1.0 if score_1 > score_0 else 0.0


def _pair_acc_from_pred_preferred(gt_ranks, pred_preferred):
    better_idx = 0 if gt_ranks[0] == 1 else 1
    return 1.0 if pred_preferred == better_idx else 0.0


def compute_method_split_metrics(method_dir, split_ids, score_key_0, score_key_1):
    correct = 0.0
    total = 0

    for seed in SEEDS:
        meta_rows = load_seed_meta(seed)
        entries_by_idx = load_method_entries(method_dir, seed)

        for row_idx, meta in enumerate(meta_rows):
            ranking_id = int(meta["ranking_id"])
            if ranking_id not in split_ids:
                continue

            entry = entries_by_idx.get(row_idx)
            if entry is None:
                continue

            label_0 = float(entry.get("label_0", 0))
            label_1 = float(entry.get("label_1", 0))
            gt_ranks = gt_pair_ranks(label_0, label_1)
            if gt_ranks is None:
                continue

            if score_key_0 is not None and score_key_1 is not None:
                score_0 = entry.get(score_key_0)
                score_1 = entry.get(score_key_1)
                if score_0 is None or score_1 is None:
                    continue
                score_0 = float(score_0)
                score_1 = float(score_1)
                pair_acc = _pair_acc_from_scores(gt_ranks, score_0, score_1)
            else:
                pred_preferred = entry.get("pred_preferred")
                if pred_preferred not in (0, 1):
                    continue
                pair_acc = _pair_acc_from_pred_preferred(gt_ranks, pred_preferred)

            total += 1
            correct += pair_acc

    return {
        "total": total,
        "pair_acc": (correct / total) if total else None,
    }


def compute_baseline_split_metrics(data_file, split_ids, score_key_0, score_key_1, rid_to_idx, seed_meta):
    with open(BASE_DIR / data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct = 0.0
    total = 0

    for seed in SEEDS:
        for meta in seed_meta[seed]:
            ranking_id = int(meta["ranking_id"])
            if ranking_id not in split_ids:
                continue

            data_idx = rid_to_idx.get(ranking_id)
            if data_idx is None or data_idx >= len(data):
                continue
            entry = data[data_idx]

            label_0 = float(entry.get("label_0", 0))
            label_1 = float(entry.get("label_1", 0))
            gt_ranks = gt_pair_ranks(label_0, label_1)
            if gt_ranks is None:
                continue

            score_0 = entry.get(score_key_0)
            score_1 = entry.get(score_key_1)
            if score_0 is None or score_1 is None:
                continue
            score_0 = float(score_0)
            score_1 = float(score_1)

            total += 1
            correct += _pair_acc_from_scores(gt_ranks, score_0, score_1)

    return {
        "total": total,
        "pair_acc": (correct / total) if total else None,
    }


def fmt_pct(value):
    if value is None:
        return "--"
    return f"{value * 100:.2f}"


def generate_latex(baseline_results, agent_results, split_sizes):
    all_results = baseline_results + agent_results
    best_by_split = {}
    for split in ("Overall", "Hard", "Mid", "Easy"):
        best_by_split[split] = max(
            stats[split]["pair_acc"]
            for _, stats in all_results
            if stats[split]["pair_acc"] is not None
        )

    def maybe_bold(value, best):
        s = fmt_pct(value)
        if value is not None and abs(value - best) < 1e-12:
            return rf"\textbf{{{s}}}"
        return s

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{0.8}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        (
            r"\textbf{Method}"
            f" & \\shortstack{{\\textbf{{Overall}}\\\\($n={split_sizes['Overall']}$)}}"
            f" & \\shortstack{{\\textbf{{Hard}}\\\\($n={split_sizes['Hard']}$)}}"
            f" & \\shortstack{{\\textbf{{Mid}}\\\\($n={split_sizes['Mid']}$)}}"
            f" & \\shortstack{{\\textbf{{Easy}}\\\\($n={split_sizes['Easy']}$)}} \\\\"
        ),
        r"\midrule",
    ]

    for name, stats in baseline_results:
        overall = maybe_bold(stats["Overall"]["pair_acc"], best_by_split["Overall"])
        hard = maybe_bold(stats["Hard"]["pair_acc"], best_by_split["Hard"])
        mid = maybe_bold(stats["Mid"]["pair_acc"], best_by_split["Mid"])
        easy = maybe_bold(stats["Easy"]["pair_acc"], best_by_split["Easy"])
        lines.append(f"{name:<24} & {overall} & {hard} & {mid} & {easy} \\\\")

    lines.append(r"\midrule")

    for name, stats in agent_results:
        overall = maybe_bold(stats["Overall"]["pair_acc"], best_by_split["Overall"])
        hard = maybe_bold(stats["Hard"]["pair_acc"], best_by_split["Hard"])
        mid = maybe_bold(stats["Mid"]["pair_acc"], best_by_split["Mid"])
        easy = maybe_bold(stats["Easy"]["pair_acc"], best_by_split["Easy"])
        lines.append(f"{name:<24} & {overall} & {hard} & {mid} & {easy} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\caption{Pairwise performance on Pick-a-Pic by approximate difficulty. "
        r"We report PairAcc for Overall, Hard, Mid, and Easy, where difficulty is "
        r"defined by tertiles of the PickScore gap $\Delta s = |s_0-s_1|$. Best per "
        r"column in \textbf{bold}.}",
        r"\label{tab:pickapic_hard}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    print("=" * 90)
    print("Table 4 — Pick-a-Pic Overall / Hard / Mid / Easy PairAcc")
    print(f"PickScore   : {PICKSCORE_FILE}")
    print(f"Sample dir  : {SAMPLE_DIR}")
    print(f"Output path : {OUTPUT_PATH}")
    print("=" * 90)

    difficulty_bins = load_global_difficulty_bins()
    split_ids = {
        split: difficulty_bins[split]
        for split in ("Overall", "Hard", "Mid", "Easy")
    }
    split_sizes = difficulty_bins["sizes"]
    print(f"\n[1] Global PickScore pairs: {difficulty_bins['total']}")
    print(f"[2] Overall bin size      : {split_sizes['Overall']}")
    print(f"[3] Hard bin size         : {split_sizes['Hard']}")
    print(f"[4] Mid bin size          : {split_sizes['Mid']}")
    print(f"[5] Easy bin size         : {split_sizes['Easy']}")
    rid_to_idx = build_ranking_id_to_index()
    seed_meta = {seed: load_seed_meta(seed) for seed in SEEDS}
    sampled_split_sizes = count_sampled_split_sizes(seed_meta, split_ids)
    print(
        "[6] Sampled Overall/Hard/Mid/Easy : "
        f"{sampled_split_sizes['Overall']}/{sampled_split_sizes['Hard']}/"
        f"{sampled_split_sizes['Mid']}/{sampled_split_sizes['Easy']}"
    )

    baseline_results = []
    agent_results = []

    print("\n[7] Computing PairAcc for baselines ...")
    for method_name, data_file, score_key_0, score_key_1 in BASELINE_MODELS:
        stats = {
            split: compute_baseline_split_metrics(
                data_file, split_ids[split], score_key_0, score_key_1, rid_to_idx, seed_meta
            )
            for split in ("Overall", "Hard", "Mid", "Easy")
        }
        print(
            f"    {method_name:22s}: "
            f"Overall={fmt_pct(stats['Overall']['pair_acc'])} "
            f"Hard={fmt_pct(stats['Hard']['pair_acc'])} "
            f"Mid={fmt_pct(stats['Mid']['pair_acc'])} "
            f"Easy={fmt_pct(stats['Easy']['pair_acc'])} "
            f"(n={stats['Overall']['total']}/{stats['Hard']['total']}/"
            f"{stats['Mid']['total']}/{stats['Easy']['total']})"
        )
        baseline_results.append((method_name, stats))

    print("\n[8] Computing PairAcc for agent models ...")
    for method_name, method_dir, score_key_0, score_key_1 in AGENT_MODELS:
        stats = {
            split: compute_method_split_metrics(
                method_dir, split_ids[split], score_key_0, score_key_1
            )
            for split in ("Overall", "Hard", "Mid", "Easy")
        }
        print(
            f"    {method_name:22s}: "
            f"Overall={fmt_pct(stats['Overall']['pair_acc'])} "
            f"Hard={fmt_pct(stats['Hard']['pair_acc'])} "
            f"Mid={fmt_pct(stats['Mid']['pair_acc'])} "
            f"Easy={fmt_pct(stats['Easy']['pair_acc'])} "
            f"(n={stats['Overall']['total']}/{stats['Hard']['total']}/"
            f"{stats['Mid']['total']}/{stats['Easy']['total']})"
        )
        agent_results.append((method_name, stats))

    tex = generate_latex(baseline_results, agent_results, sampled_split_sizes)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(tex)

    print("\n[9] LaTeX preview:\n")
    print(tex)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
