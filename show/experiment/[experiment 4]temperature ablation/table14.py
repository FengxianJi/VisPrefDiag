#!/usr/bin/env python3
"""
table14.py - Temperature ablation on ImageReward with the same pairwise
difficulty split and table header style as table13.py.

For each prompt group, we enumerate all valid image pairs with different human
ranks. Since the original ranks can be non-consecutive, we compress unique
ranks in each group into consecutive levels while preserving order, and define
the human pair gap as:

    d(i, j) = |level_i - level_j|

Difficulty bins:
  Hard: d = 1
  Mid : d = 2
  Easy: d >= 3

Metrics per split:
  PairAcc : pairwise accuracy over all valid image pairs in the split.
  NormGap : mean normalized score gap over all valid image pairs in the split,
            reported in absolute value.
  AUROC   : global ROC-AUC over all pairs that belong to the split.

Data source:
  show/table/table14/T=*/test_original.json
  with fallback to test_1_original.json + test_2_original.json

Output:
  show/table/table14.tex
"""

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = Path(__file__).resolve().parent
TABLE14_DIR = TABLE_DIR / "table14"
BENCHMARK_GROUP_JSONS = [
    ROOT / "benchmark" / "ImageReward" / "test_1.json",
    ROOT / "benchmark" / "ImageReward" / "test_2.json",
]
OUTPUT_PATH = TABLE_DIR / "table14.tex"

SPLITS = ["Overall", "Hard", "Mid", "Easy"]
BIN_SPLITS = ["Hard", "Mid", "Easy"]
METRIC_KEYS = ["pair_acc", "norm_gap", "auc"]
GT_KEY = "ranks"


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


def build_benchmark_maps():
    grouped_rows = {}
    for json_path in BENCHMARK_GROUP_JSONS:
        with open(json_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            gid = row["prompt_id"]
            grouped_rows.setdefault(gid, []).append(row)

    pair_counts = Counter()
    benchmark_group_map = {}
    for gid, rows in grouped_rows.items():
        sorted_rows = sorted(rows, key=lambda row: Path(row["image_path"]).name)
        order = [Path(row["image_path"]).name for row in sorted_rows]
        ranks = [row["rank"] for row in sorted_rows]
        levels = _compress_ranks_to_levels(ranks)

        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                if ranks[i] == ranks[j]:
                    continue
                split = _difficulty_from_gap(abs(levels[i] - levels[j]))
                pair_counts["Overall"] += 1
                pair_counts[split] += 1

        benchmark_group_map[gid] = {
            "order": order,
            "rank_by_name": {name: rank for name, rank in zip(order, ranks)},
        }

    return pair_counts, benchmark_group_map


def _compute_auroc(labels, score_diffs):
    pos = [score for score, label in zip(score_diffs, labels) if label == 1]
    neg = [score for score, label in zip(score_diffs, labels) if label == 0]
    if not pos or not neg:
        return None

    concordant = 0
    tied = 0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                concordant += 1
            elif pos_score == neg_score:
                tied += 1
    return (concordant + 0.5 * tied) / (len(pos) * len(neg))


def _minmax_normalize(scores):
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score
    if span == 0:
        return [0.0 for _ in scores]
    return [(score - min_score) / span for score in scores]


def _align_group_to_benchmark(group, gt_key, benchmark_group_map=None):
    scores = group.get("scores", [])
    gt_ranks = group.get(gt_key, [])
    gid = group.get("group_id")
    img_paths = group.get("img_paths", [])
    if gid and img_paths and benchmark_group_map and len(scores) == len(img_paths):
        group_info = benchmark_group_map.get(gid)
        if group_info:
            score_by_name = {
                Path(path).name: score for path, score in zip(img_paths, scores)
            }
            order = group_info["order"]
            rank_by_name = group_info["rank_by_name"]
            if all(name in score_by_name and name in rank_by_name for name in order):
                return (
                    [score_by_name[name] for name in order],
                    [rank_by_name[name] for name in order],
                )
    return scores, gt_ranks


def _build_pair_records(scores, gt_ranks):
    levels = _compress_ranks_to_levels(gt_ranks)
    norm_scores = _minmax_normalize(scores)
    pair_records = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if gt_ranks[i] == gt_ranks[j]:
                continue

            label = 1 if gt_ranks[i] < gt_ranks[j] else 0
            if label == 1:
                better_idx, worse_idx = i, j
            else:
                better_idx, worse_idx = j, i

            better_level = levels[better_idx]
            worse_level = levels[worse_idx]
            split = _difficulty_from_gap(abs(better_level - worse_level))
            pair_records.append(
                {
                    "split": split,
                    "label": label,
                    "score_diff": scores[i] - scores[j],
                    "norm_score_diff": norm_scores[better_idx] - norm_scores[worse_idx],
                }
            )
    return pair_records


def _empty_metric():
    return {
        "pair_acc": None,
        "norm_gap": None,
        "auc": None,
        "n_groups": 0,
        "n_pairs": 0,
    }


def _summarize_split(pair_entries, n_groups):
    metric = _empty_metric()
    metric["n_groups"] = n_groups
    metric["n_pairs"] = len(pair_entries)

    if pair_entries:
        pair_acc_vals = [
            1.0
            if (pair["label"] == 1 and pair["score_diff"] > 0)
            or (pair["label"] == 0 and pair["score_diff"] < 0)
            else 0.0
            for pair in pair_entries
        ]
        norm_gap_vals = [pair["norm_score_diff"] for pair in pair_entries]
        metric["pair_acc"] = sum(pair_acc_vals) / len(pair_acc_vals)
        metric["norm_gap"] = abs(sum(norm_gap_vals) / len(norm_gap_vals))
        metric["auc"] = _compute_auroc(
            [pair["label"] for pair in pair_entries],
            [pair["score_diff"] for pair in pair_entries],
        )

    return metric


def _summarize_records(records):
    all_pairs = {split: [] for split in SPLITS}
    group_ids = {split: set() for split in SPLITS}

    for record in records:
        scores = record["scores"]
        gt_ranks = record["gt_ranks"]
        gid = record["group_id"]
        if not scores or not gt_ranks or len(scores) != len(gt_ranks):
            continue

        pair_records = _build_pair_records(scores, gt_ranks)
        if not pair_records:
            continue

        all_pairs["Overall"].extend(pair_records)
        group_ids["Overall"].add(gid)

        for split in BIN_SPLITS:
            split_pairs = [pair for pair in pair_records if pair["split"] == split]
            all_pairs[split].extend(split_pairs)
            if split_pairs:
                group_ids[split].add(gid)

    return {
        split: _summarize_split(all_pairs[split], len(group_ids[split]))
        for split in SPLITS
    }


def build_human_pair_metrics(groups, gt_key, benchmark_group_map):
    records = []
    for group in groups:
        gid = group.get("group_id")
        if not gid:
            continue
        scores, gt_ranks = _align_group_to_benchmark(group, gt_key, benchmark_group_map)
        if not scores or not gt_ranks or len(scores) != len(gt_ranks):
            continue
        records.append({"group_id": gid, "scores": scores, "gt_ranks": gt_ranks})
    return _summarize_records(records)


def _safe_load(path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: failed to parse JSON {path}: {e}")
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


def discover_temperatures():
    entries = []
    for path in sorted(TABLE14_DIR.glob("T=*"), key=lambda p: float(p.name.split("=", 1)[1])):
        if path.is_dir():
            entries.append((path.name, path))
    return entries


def load_all_results(benchmark_group_map):
    results = {}
    for temp_label, temp_dir in discover_temperatures():
        data = _load_split_agent(temp_dir / "test_original.json")
        if not data:
            continue
        results[temp_label] = build_human_pair_metrics(data, GT_KEY, benchmark_group_map)
    return results


def fmt(metric_key, value):
    if value is None:
        return "--"
    if metric_key == "norm_gap":
        return f"{value:.3f}"
    return f"{value * 100:.2f}"


def compute_best(results):
    best = {}
    for split in SPLITS:
        for metric in METRIC_KEYS:
            vals = [
                temp_metrics[split][metric]
                for temp_metrics in results.values()
                if temp_metrics[split][metric] is not None
            ]
            best[(split, metric)] = max(vals) if vals else None
    return best


def make_cell(results, best, temp_label, split, metric):
    value = results[temp_label][split][metric]
    text = fmt(metric, value)
    if value is not None and best[(split, metric)] is not None:
        if abs(value - best[(split, metric)]) < 1e-12:
            return rf"\textbf{{{text}}}"
    return text


def generate_table(results, pair_counts):
    best = compute_best(results)

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.38}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "ccc" * 4 + "}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Temperature}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Overall}} ($n={pair_counts['Overall']}$ pairs)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Hard}} ($d{{=}}1$, $n={pair_counts['Hard']}$)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Mid}} ($d{{=}}2$, $n={pair_counts['Mid']}$)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Easy}} ($d{{\\geq}}3$, $n={pair_counts['Easy']}$)}} \\\\"
    )
    lines.append(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}")
    sub = r"\textbf{PairAcc}$\uparrow$ & \textbf{NormGap}$\uparrow$ & \textbf{AUROC}$\uparrow$"
    lines.append(f" & {sub} & {sub} & {sub} & {sub} \\\\")
    lines.append(r"\midrule")
    lines.append(
        r"\textit{Random baseline}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00} \\"
    )
    lines.append(r"\midrule")

    for temp_label, _ in discover_temperatures():
        if temp_label not in results:
            continue
        cells = []
        for split in SPLITS:
            for metric in METRIC_KEYS:
                cells.append(make_cell(results, best, temp_label, split, metric))
        lines.append(f"{temp_label:<12} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{Temperature ablation on ImageReward (test). "
        r"Valid image pairs are grouped by human level gap "
        r"$d(i,j)=|level_i-level_j|$ into Hard ($d=1$), Mid ($d=2$), and Easy ($d\geq3$). "
        r"We report PairAcc, NormGap, and AUROC for each split. "
        r"Best per column in bold.}"
    )
    lines.append(r"\label{tab:table14}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Table 14 - Temperature ablation with pairwise Hard/Mid/Easy split")
    print("=" * 70)

    pair_counts, benchmark_group_map = build_benchmark_maps()
    print(
        "Benchmark pair split: "
        f"Overall={pair_counts['Overall']}, "
        f"Hard(d=1)={pair_counts['Hard']}, "
        f"Mid(d=2)={pair_counts['Mid']}, "
        f"Easy(d>=3)={pair_counts['Easy']}"
    )

    results = load_all_results(benchmark_group_map)
    if not results:
        raise RuntimeError(f"No valid results found under {TABLE14_DIR}")

    print("\nResults (PairAcc / NormGap / AUROC):")
    for temp_label, _ in discover_temperatures():
        if temp_label not in results:
            print(f"  {temp_label:<8} [no data]")
            continue
        row = results[temp_label]
        print(
            f"  {temp_label:<8} "
            f"Overall: {fmt('pair_acc', row['Overall']['pair_acc'])} / "
            f"{fmt('norm_gap', row['Overall']['norm_gap'])} / "
            f"{fmt('auc', row['Overall']['auc'])}  "
            f"Hard: {fmt('pair_acc', row['Hard']['pair_acc'])} / "
            f"{fmt('norm_gap', row['Hard']['norm_gap'])} / "
            f"{fmt('auc', row['Hard']['auc'])}  "
            f"Mid: {fmt('pair_acc', row['Mid']['pair_acc'])} / "
            f"{fmt('norm_gap', row['Mid']['norm_gap'])} / "
            f"{fmt('auc', row['Mid']['auc'])}  "
            f"Easy: {fmt('pair_acc', row['Easy']['pair_acc'])} / "
            f"{fmt('norm_gap', row['Easy']['norm_gap'])} / "
            f"{fmt('auc', row['Easy']['auc'])}"
        )

    latex = generate_table(results, pair_counts)
    OUTPUT_PATH.write_text(latex + "\n", encoding="utf-8")
    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
