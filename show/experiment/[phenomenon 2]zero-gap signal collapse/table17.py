#!/usr/bin/env python3
"""
table17.py - Table 13 with an extra Zero-gap column to the right of Hard.

Unlike the first draft, the split is unified across all methods using the
benchmark's human overall scores from test_1.json and test_2.json:

  Zero-gap: |overall_rating_i - overall_rating_j| = 0
  Hard    : |overall_rating_i - overall_rating_j| = 1
  Mid     : |overall_rating_i - overall_rating_j| = 2
  Easy    : |overall_rating_i - overall_rating_j| >= 3

Only pairs with different human ranks are evaluated, matching the pairwise
metric setup of Table 13.
"""

from collections import Counter
from pathlib import Path

from table13 import (
    AGENT_DIRS,
    AGENTS,
    BASELINES,
    BASELINE_DETAIL,
    BENCHMARK_GROUP_JSONS,
    EXTRA_MODELS,
    METRIC_KEYS,
    QALIGN_MODES,
    TABLE13_DIR,
    TABLE3_ORIGIN,
    _empty_metric,
    _load_split_agent,
    _safe_load,
    _summarize_split,
    _minmax_normalize,
)


OUTPUT_PATH = Path(__file__).resolve().parent / "table17.tex"
SPLITS = ["Overall", "Hard", "Zero-gap", "Mid", "Easy"]


def _score_gap_to_split(gap):
    if gap == 0:
        return "Zero-gap"
    if gap == 1:
        return "Hard"
    if gap == 2:
        return "Mid"
    return "Easy"


def build_benchmark_score_maps():
    grouped_rows = {}
    for json_path in BENCHMARK_GROUP_JSONS:
        data = _safe_load(json_path) or []
        for row in data:
            gid = row["prompt_id"]
            grouped_rows.setdefault(gid, []).append(row)

    pair_counts = Counter()
    benchmark_group_map = {}

    for gid, rows in grouped_rows.items():
        sorted_rows = sorted(rows, key=lambda row: Path(row["image_path"]).name)
        order = [Path(row["image_path"]).name for row in sorted_rows]
        ranks = [row["rank"] for row in sorted_rows]
        overall_ratings = [row["overall_rating"] for row in sorted_rows]

        for i in range(len(sorted_rows)):
            for j in range(i + 1, len(sorted_rows)):
                if ranks[i] == ranks[j]:
                    continue
                split = _score_gap_to_split(abs(overall_ratings[i] - overall_ratings[j]))
                pair_counts["Overall"] += 1
                pair_counts[split] += 1

        benchmark_group_map[gid] = {
            "order": order,
            "rank_by_name": {name: rank for name, rank in zip(order, ranks)},
            "overall_by_name": {
                name: overall for name, overall in zip(order, overall_ratings)
            },
        }

    return pair_counts, benchmark_group_map


def _align_group_to_benchmark_with_scores(group, gt_key, benchmark_group_map):
    scores = group.get("scores", [])
    gt_ranks = group.get(gt_key, [])
    gid = group.get("group_id")
    img_paths = group.get("img_paths", [])
    if gid and img_paths and benchmark_group_map and len(scores) == len(img_paths):
        group_info = benchmark_group_map.get(gid)
        if group_info:
            score_by_name = {Path(path).name: score for path, score in zip(img_paths, scores)}
            order = group_info["order"]
            rank_by_name = group_info["rank_by_name"]
            overall_by_name = group_info["overall_by_name"]
            if all(
                name in score_by_name and name in rank_by_name and name in overall_by_name
                for name in order
            ):
                return (
                    [score_by_name[name] for name in order],
                    [rank_by_name[name] for name in order],
                    [overall_by_name[name] for name in order],
                )
    return scores, gt_ranks, []


def _build_pair_records(scores, gt_ranks, overall_ratings):
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

            split = _score_gap_to_split(abs(overall_ratings[i] - overall_ratings[j]))
            pair_records.append(
                {
                    "split": split,
                    "label": label,
                    "score_diff": scores[i] - scores[j],
                    "norm_score_diff": norm_scores[better_idx] - norm_scores[worse_idx],
                }
            )
    return pair_records


def _summarize_records_with_zero_gap(records):
    all_pairs = {split: [] for split in SPLITS}
    group_ids = {split: set() for split in SPLITS}

    for record in records:
        scores = record["scores"]
        gt_ranks = record["gt_ranks"]
        overall_ratings = record["overall_ratings"]
        gid = record["group_id"]
        if (
            not scores
            or not gt_ranks
            or not overall_ratings
            or len(scores) != len(gt_ranks)
            or len(scores) != len(overall_ratings)
        ):
            continue

        pair_records = _build_pair_records(scores, gt_ranks, overall_ratings)
        if not pair_records:
            continue

        all_pairs["Overall"].extend(pair_records)
        group_ids["Overall"].add(gid)

        for split in ("Hard", "Zero-gap", "Mid", "Easy"):
            split_pairs = [pair for pair in pair_records if pair["split"] == split]
            if split_pairs:
                all_pairs[split].extend(split_pairs)
                group_ids[split].add(gid)

    return {
        split: _summarize_split(all_pairs[split], len(group_ids[split]))
        for split in SPLITS
    }


def build_human_pair_metrics_with_zero_gap(groups, gt_key, benchmark_group_map):
    records = []
    for group in groups:
        gid = group.get("group_id")
        if not gid:
            continue
        scores, gt_ranks, overall_ratings = _align_group_to_benchmark_with_scores(
            group, gt_key, benchmark_group_map
        )
        if (
            not scores
            or not gt_ranks
            or not overall_ratings
            or len(scores) != len(gt_ranks)
            or len(scores) != len(overall_ratings)
        ):
            continue
        records.append(
            {
                "group_id": gid,
                "scores": scores,
                "gt_ranks": gt_ranks,
                "overall_ratings": overall_ratings,
            }
        )
    return _summarize_records_with_zero_gap(records)


def empty_split_metrics():
    return {split: _empty_metric() for split in SPLITS}


def load_all_results(benchmark_group_map):
    results = {}
    extra_model_results = {}

    for name, fname in BASELINE_DETAIL.items():
        data = _safe_load(TABLE3_ORIGIN / fname)
        if not data:
            continue
        groups = data.get("test", {}).get("details", [])
        if not groups:
            continue
        results[name] = build_human_pair_metrics_with_zero_gap(
            groups, "gt_ranks", benchmark_group_map
        )

    for suffix, display_name in QALIGN_MODES:
        data = _safe_load(TABLE3_ORIGIN / f"{suffix}_detail.json")
        if not data:
            continue
        groups = data.get("test", {}).get("details", [])
        if not groups:
            continue
        results[display_name] = build_human_pair_metrics_with_zero_gap(
            groups, "gt_ranks", benchmark_group_map
        )

    for display_name, model_dir in EXTRA_MODELS:
        model_path = TABLE13_DIR / model_dir
        model_results = {agent_name: empty_split_metrics() for agent_name in AGENTS}
        if model_path.exists():
            for agent_name, dirname in AGENT_DIRS.items():
                data = _load_split_agent(model_path / dirname / "test_original.json")
                if data:
                    model_results[agent_name] = build_human_pair_metrics_with_zero_gap(
                        data, "ranks", benchmark_group_map
                    )
        extra_model_results[display_name] = model_results

    return results, extra_model_results


def iter_display_metrics(results, extra_model_results):
    for method in BASELINES:
        if method in results:
            yield results[method]
    for display_name, _ in EXTRA_MODELS:
        for agent_name in AGENTS:
            yield extra_model_results.get(display_name, {}).get(
                agent_name, empty_split_metrics()
            )


def fmt(metric_key, value):
    if value is None:
        return "--"
    if metric_key == "norm_gap":
        return f"{value:.3f}"
    return f"{value * 100:.2f}"


def generate_table(results, extra_model_results, pair_counts):
    best = {}
    for split in SPLITS:
        for metric_key in METRIC_KEYS:
            values = [
                method_metrics[split][metric_key]
                for method_metrics in iter_display_metrics(results, extra_model_results)
                if method_metrics[split][metric_key] is not None
            ]
            if not values:
                best[(split, metric_key)] = None
            elif any(abs(value - values[0]) >= 1e-12 for value in values[1:]):
                best[(split, metric_key)] = max(values)
            else:
                best[(split, metric_key)] = None

    def cell(method_metrics, split, metric_key):
        value = method_metrics.get(split, {}).get(metric_key)
        text = fmt(metric_key, value)
        best_value = best.get((split, metric_key))
        if value is not None and best_value is not None and abs(value - best_value) < 1e-12:
            return f"\\textbf{{{text}}}"
        return text

    def row(label, method_metrics):
        cells = []
        for split in SPLITS:
            for metric_key in METRIC_KEYS:
                cells.append(cell(method_metrics, split, metric_key))
        return f"{label:<28} & " + " & ".join(cells) + r" \\"

    n_overall = pair_counts["Overall"]
    n_hard = pair_counts["Hard"]
    n_zero_gap = pair_counts["Zero-gap"]
    n_mid = pair_counts["Mid"]
    n_easy = pair_counts["Easy"]

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.38}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "ccc" * 5 + "}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{2}{*}{\textbf{Method}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Overall}} ($n={n_overall}$ pairs)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Hard}} ($\\Delta_{{\\mathrm{{overall}}}}=1$, $n={n_hard}$)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Zero-gap}} ($\\Delta_{{\\mathrm{{overall}}}}=0$, $n={n_zero_gap}$)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Mid}} ($\\Delta_{{\\mathrm{{overall}}}}=2$, $n={n_mid}$)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Easy}} ($\\Delta_{{\\mathrm{{overall}}}}\\geq3$, $n={n_easy}$)}} \\\\"
    )
    lines.append(
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}"
        r"\cmidrule(lr){11-13}\cmidrule(lr){14-16}"
    )
    sub = r"\textbf{PairAcc}$\uparrow$ & \textbf{NormGap}$\uparrow$ & \textbf{AUROC}$\uparrow$"
    lines.append(f" & {sub} & {sub} & {sub} & {sub} & {sub} \\\\")
    lines.append(r"\midrule")
    lines.append(
        r"\textit{Random baseline}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00}"
        r" & \textit{50.00} & \textit{0.000} & \textit{50.00} \\"
    )
    lines.append(r"\midrule")

    for method in BASELINES:
        if method in results:
            lines.append(row(method, results[method]))

    for display_name, _ in EXTRA_MODELS:
        if display_name not in extra_model_results:
            continue
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{16}}{{c}}{{\textbf{{{display_name}}}}} \\")
        lines.append(r"\midrule")
        for agent_name in AGENTS:
            lines.append(row(agent_name, extra_model_results[display_name][agent_name]))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{Pairwise performance by difficulty on ImageReward (test), "
        r"augmented from Table~13 with a \textbf{Zero-gap} column. "
        r"All methods share the same benchmark-defined split computed from human "
        r"\texttt{overall\_rating} in \texttt{test\_1.json} and \texttt{test\_2.json}: "
        r"Zero-gap ($\Delta_{\mathrm{overall}}=0$), Hard ($\Delta_{\mathrm{overall}}=1$), "
        r"Mid ($\Delta_{\mathrm{overall}}=2$), and Easy ($\Delta_{\mathrm{overall}}\geq3$). "
        r"Only pairs with different human ranks are evaluated. "
        r"We report PairAcc, NormGap, and AUROC for each split. "
        r"Best per column in bold.}"
    )
    lines.append(r"\label{tab:table17}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Table 17 - Table 13 with benchmark-score Zero-gap / Hard / Mid / Easy")
    print("=" * 70)

    pair_counts, benchmark_group_map = build_benchmark_score_maps()
    print(
        "Benchmark pair split: "
        f"Overall={pair_counts['Overall']}, "
        f"Zero-gap(Delta_overall=0)={pair_counts['Zero-gap']}, "
        f"Hard(Delta_overall=1)={pair_counts['Hard']}, "
        f"Mid(Delta_overall=2)={pair_counts['Mid']}, "
        f"Easy(Delta_overall>=3)={pair_counts['Easy']}"
    )

    results, extra_model_results = load_all_results(benchmark_group_map)

    print("\nResults (PairAcc / NormGap / AUROC / Zero-gap n_pairs):")
    for method in BASELINES:
        if method not in results:
            print(f"  {method:<28} [no data]")
            continue
        metrics = results[method]
        print(
            f"  {method:<28} "
            f"Overall: {fmt('pair_acc', metrics['Overall']['pair_acc'])} / "
            f"{fmt('norm_gap', metrics['Overall']['norm_gap'])} / "
            f"{fmt('auc', metrics['Overall']['auc'])}  "
            f"Hard: {fmt('pair_acc', metrics['Hard']['pair_acc'])} / "
            f"{fmt('norm_gap', metrics['Hard']['norm_gap'])} / "
            f"{fmt('auc', metrics['Hard']['auc'])}  "
            f"Zero-gap: {fmt('pair_acc', metrics['Zero-gap']['pair_acc'])} / "
            f"{fmt('norm_gap', metrics['Zero-gap']['norm_gap'])} / "
            f"{fmt('auc', metrics['Zero-gap']['auc'])} "
            f"(n={metrics['Zero-gap']['n_pairs']})"
        )

    print("\nExtra models:")
    for display_name, _ in EXTRA_MODELS:
        print(f"  {display_name}:")
        for agent_name in AGENTS:
            metrics = extra_model_results.get(display_name, {}).get(agent_name, empty_split_metrics())
            print(
                f"    {agent_name:<24} "
                f"Hard: {fmt('pair_acc', metrics['Hard']['pair_acc'])} / "
                f"{fmt('norm_gap', metrics['Hard']['norm_gap'])} / "
                f"{fmt('auc', metrics['Hard']['auc'])}  "
                f"Zero-gap: {fmt('pair_acc', metrics['Zero-gap']['pair_acc'])} / "
                f"{fmt('norm_gap', metrics['Zero-gap']['norm_gap'])} / "
                f"{fmt('auc', metrics['Zero-gap']['auc'])} "
                f"(n={metrics['Zero-gap']['n_pairs']})"
            )

    latex = generate_table(results, extra_model_results, pair_counts)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"\nWritten to {OUTPUT_PATH}")
    print("\n" + "=" * 70)
    print(latex)
    print("=" * 70)


if __name__ == "__main__":
    main()
