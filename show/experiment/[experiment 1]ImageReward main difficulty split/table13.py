#!/usr/bin/env python3
"""
table13.py - ImageReward test set: pairwise difficulty split by human level gap.

For each prompt group, we enumerate all valid image pairs with different human
ranks. Since the original ranks can be non-consecutive (for example, 1/2/5/3),
we first compress unique ranks in each group into consecutive levels while
preserving order, and define the human pair gap as:

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

Data loading priority: test_original.json (unbiased, random order) first,
then test_1+test_2 as fallback (for models that only have split files).
"""

import json
from collections import Counter
from pathlib import Path


# Paths
TABLE3_ORIGIN = Path(__file__).resolve().parent / "table3" / "origin"
TABLE13_DIR = Path(__file__).resolve().parent / "table13"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_GROUP_JSONS = [
    PROJECT_ROOT / "benchmark" / "ImageReward" / "test_1.json",
    PROJECT_ROOT / "benchmark" / "ImageReward" / "test_2.json",
]
OUTPUT_PATH = Path(__file__).resolve().parent / "table13.tex"


# Method definitions
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
    "Valuator": "Auction-arena",
    "Single-agent (Pair)": "Single-agent_pair",
    "Single-agent (Price)": "Single-agent_price",
    "Budgeted Single-agent": "Single-agent-Budgeted",
    "Single-agent (LLM)": "Single-agentLLM_score",
    "Valuator Judge": "Valuator_judge",
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
    "Valuator",
    "Single-agent (Pair)",
    "Single-agent (Price)",
    "Budgeted Single-agent",
    "Single-agent (LLM)",
    "Valuator Judge",
]
ALL_METHODS = BASELINES + AGENTS

EXTRA_MODELS = [
    ("Qwen3VL-4B", "qwen3vl4b"),
    ("Qwen3VL-8B", "qwen3vl8b"),
    ("Qwen3VL-30B", "qwen3vl30b"),
    ("Kimi-K2.5", "kimi-k2.5"),
    ("Gemini-3-Flash", "gemini-3-flash"),
    ("Llama3.2V-11B", "llama3.2V-11b"),
]
SPLITS = ["Overall", "Hard", "Mid", "Easy"]
BIN_SPLITS = ["Hard", "Mid", "Easy"]
METRIC_KEYS = ["pair_acc", "norm_gap", "auc"]
ORDER_GUARD_MIN_GROUPS = 100
ORDER_GUARD_MIN_MONOTONIC_RATE = 0.98
ORDER_GUARD_MAX_TOP1_RATE = 0.35
ORDER_GUARD_MIN_PAIR_ACC = 0.75
ORDER_GUARD_MAX_UNIQUE_SCORE_RATIO = 0.20
ORDER_GUARD_MIN_DOMINANT_TOP_RATE = 0.60
ORDER_GUARD_MIN_SWAP_BASELINE = 0.90


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
    """Return pair counts and benchmark group alignment metadata."""
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
            score_by_name = {Path(path).name: score for path, score in zip(img_paths, scores)}
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
                    "better_idx": better_idx,
                    "worse_idx": worse_idx,
                    "selection_key": (
                        better_level,
                        worse_level,
                        better_idx,
                        worse_idx,
                    ),
                }
            )
    return pair_records


def _select_top_pair(pair_records, split=None):
    candidates = pair_records
    if split is not None:
        candidates = [pair for pair in pair_records if pair["split"] == split]
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair["selection_key"])


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


def _rate(count, total):
    if not total:
        return None
    return count / total


def _position_template_metrics(groups, gt_key, benchmark_group_map, score_maker):
    records = []
    for group in groups:
        gid = group.get("group_id")
        img_paths = group.get("img_paths", [])
        gt_ranks = group.get(gt_key, [])
        if not gid or not img_paths or not gt_ranks or len(img_paths) != len(gt_ranks):
            continue

        group_info = benchmark_group_map.get(gid)
        if not group_info:
            continue

        scores = score_maker(len(gt_ranks))
        score_by_name = {
            Path(path).name: score
            for path, score in zip(img_paths, scores)
        }
        order = group_info["order"]
        rank_by_name = group_info["rank_by_name"]
        if all(name in score_by_name and name in rank_by_name for name in order):
            records.append(
                {
                    "group_id": gid,
                    "scores": [score_by_name[name] for name in order],
                    "gt_ranks": [rank_by_name[name] for name in order],
                }
            )

    return _summarize_records(records)


def diagnose_order_template_contamination(groups, gt_key, benchmark_group_map):
    n_groups = len(groups)
    n_ranked = 0
    n_monotonic = 0
    n_top = 0
    n_top_correct = 0
    n_default = 0
    unique_scores = Counter()
    top_positions = Counter()

    for group in groups:
        gt_ranks = group.get(gt_key, [])
        scores = group.get("scores", [])
        if group.get("ranking_source") == "default":
            n_default += 1

        if gt_ranks:
            n_ranked += 1
            if all(gt_ranks[idx] <= gt_ranks[idx + 1] for idx in range(len(gt_ranks) - 1)):
                n_monotonic += 1

        if scores:
            unique_scores[tuple(scores)] += 1

        if scores and gt_ranks and len(scores) == len(gt_ranks):
            top_idx = max(range(len(scores)), key=lambda idx: scores[idx])
            best_idx = min(range(len(gt_ranks)), key=lambda idx: gt_ranks[idx])
            n_top += 1
            n_top_correct += int(top_idx == best_idx)
            top_positions[top_idx + 1] += 1

    metrics = build_human_pair_metrics(groups, gt_key, benchmark_group_map)
    pair_acc = metrics["Overall"]["pair_acc"]
    auc = metrics["Overall"]["auc"]
    top1_rate = _rate(n_top_correct, n_top)
    monotonic_rate = _rate(n_monotonic, n_ranked)
    unique_score_ratio = _rate(len(unique_scores), n_groups)
    dominant_top_rate = _rate(top_positions.most_common(1)[0][1], n_top) if top_positions else None
    default_rate = _rate(n_default, n_groups)

    swap_metrics = _position_template_metrics(
        groups,
        gt_key,
        benchmark_group_map,
        lambda n: [n - 1, n] + list(range(n - 2, 0, -1)) if n >= 2 else [1],
    )
    swap_pair_acc = swap_metrics["Overall"]["pair_acc"]

    contaminated = (
        n_groups >= ORDER_GUARD_MIN_GROUPS
        and monotonic_rate is not None
        and monotonic_rate >= ORDER_GUARD_MIN_MONOTONIC_RATE
        and top1_rate is not None
        and top1_rate <= ORDER_GUARD_MAX_TOP1_RATE
        and pair_acc is not None
        and pair_acc >= ORDER_GUARD_MIN_PAIR_ACC
        and swap_pair_acc is not None
        and swap_pair_acc >= ORDER_GUARD_MIN_SWAP_BASELINE
        and (
            (
                unique_score_ratio is not None
                and unique_score_ratio <= ORDER_GUARD_MAX_UNIQUE_SCORE_RATIO
            )
            or (
                dominant_top_rate is not None
                and dominant_top_rate >= ORDER_GUARD_MIN_DOMINANT_TOP_RATE
            )
        )
    )

    return {
        "contaminated": contaminated,
        "n_groups": n_groups,
        "monotonic_rate": monotonic_rate,
        "top1_rate": top1_rate,
        "pair_acc": pair_acc,
        "auc": auc,
        "unique_score_ratio": unique_score_ratio,
        "dominant_top_rate": dominant_top_rate,
        "default_rate": default_rate,
        "swap_pair_acc": swap_pair_acc,
        "top_positions": dict(top_positions.most_common(5)),
    }


def contamination_reason(diagnostics):
    return (
        "order-template contamination: "
        f"monotonic={diagnostics['monotonic_rate']:.3f}, "
        f"top1={diagnostics['top1_rate']:.3f}, "
        f"pair_acc={diagnostics['pair_acc']:.3f}, "
        f"unique_scores={diagnostics['unique_score_ratio']:.3f}, "
        f"swap_baseline={diagnostics['swap_pair_acc']:.3f}"
    )


def attach_warning(metrics, reason, diagnostics):
    metrics["__warning__"] = True
    metrics["__warning_reason"] = reason
    metrics["__diagnostics"] = diagnostics
    return metrics


def _safe_load(path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: failed to parse JSON {path}: {e}")
    return None


def _load_split_agent(path: Path):
    """Priority: test_original.json first, then test_1+test_2 as fallback."""
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
        results[name] = build_human_pair_metrics(groups, "gt_ranks", benchmark_group_map)

    for suffix, display_name in QALIGN_MODES:
        data = _safe_load(TABLE3_ORIGIN / f"{suffix}_detail.json")
        if not data:
            continue
        groups = data.get("test", {}).get("details", [])
        if not groups:
            continue
        results[display_name] = build_human_pair_metrics(groups, "gt_ranks", benchmark_group_map)

    for display_name, model_dir in EXTRA_MODELS:
        model_path = TABLE13_DIR / model_dir
        model_agent_names = AGENTS
        model_results = {agent_name: empty_split_metrics() for agent_name in model_agent_names}
        if model_path.exists():
            for agent_name, dirname in AGENT_DIRS.items():
                data = _load_split_agent(model_path / dirname / "test_original.json")
                if data:
                    diagnostics = diagnose_order_template_contamination(
                        data, "ranks", benchmark_group_map
                    )
                    metrics = build_human_pair_metrics(
                        data, "ranks", benchmark_group_map
                    )
                    if diagnostics["contaminated"]:
                        reason = contamination_reason(diagnostics)
                        print(
                            f"Warning: flagged {display_name} / {agent_name}: {reason}"
                        )
                        metrics = attach_warning(metrics, reason, diagnostics)
                    model_results[agent_name] = metrics
        extra_model_results[display_name] = model_results

    return results, extra_model_results


def iter_display_metrics(results, extra_model_results):
    for method in BASELINES:
        if method in results:
            yield results[method]
    for display_name, _ in EXTRA_MODELS:
        agent_names = AGENTS
        for agent_name in agent_names:
            yield extra_model_results.get(display_name, {}).get(
                agent_name,
                empty_split_metrics(),
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
            best[(split, metric_key)] = max(values) if values else None

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
    n_mid = pair_counts["Mid"]
    n_easy = pair_counts["Easy"]

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
        r"\multirow{2}{*}{\textbf{Method}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Overall}} ($n={n_overall}$ pairs)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Hard}} ($d{{=}}1$, $n={n_hard}$)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Mid}} ($d{{=}}2$, $n={n_mid}$)}}"
        f" & \\multicolumn{{3}}{{c}}{{\\textbf{{Easy}} ($d{{\\geq}}3$, $n={n_easy}$)}} \\\\"
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

    for method in BASELINES:
        if method in results:
            lines.append(row(method, results[method]))

    for display_name, _ in EXTRA_MODELS:
        if display_name not in extra_model_results:
            continue
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{13}}{{c}}{{\textbf{{{display_name}}}}} \\")
        lines.append(r"\midrule")
        agent_names = AGENTS
        for agent_name in agent_names:
            lines.append(row(agent_name, extra_model_results[display_name][agent_name]))

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{Pairwise performance by difficulty on ImageReward (test). "
        r"Valid image pairs are grouped by human level gap $d(i,j)=|level_i-level_j|$ into "
        r"Hard ($d=1$), Mid ($d=2$), and Easy ($d\geq3$). "
        r"We report PairAcc, NormGap, and AUROC for each split. "
        r"Best per column in bold. "
        r"\textit{Caution:} the Llama3.2V-11B Single-agent (Pair) row is flagged by an order-template contamination diagnostic "
        r"(monotonic saved ranks, low top-1, repeated position-like rankings), so its unusually high pairwise scores should be interpreted as potentially inflated rather than clean model gains.}"
    )
    lines.append(r"\label{tab:table13}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Table 13 - ImageReward pairwise Hard/Mid/Easy split (level-gap based)")
    print("=" * 70)

    pair_counts, benchmark_group_map = build_benchmark_maps()
    print(
        "Benchmark pair split: "
        f"Overall={pair_counts['Overall']}, "
        f"Hard(d=1)={pair_counts['Hard']}, "
        f"Mid(d=2)={pair_counts['Mid']}, "
        f"Easy(d>=3)={pair_counts['Easy']}"
    )

    results, extra_model_results = load_all_results(benchmark_group_map)

    print("\nResults (PairAcc / NormGap / AUROC):")
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
            f"Mid: {fmt('pair_acc', metrics['Mid']['pair_acc'])} / "
            f"{fmt('norm_gap', metrics['Mid']['norm_gap'])} / "
            f"{fmt('auc', metrics['Mid']['auc'])}  "
            f"Easy: {fmt('pair_acc', metrics['Easy']['pair_acc'])} / "
            f"{fmt('norm_gap', metrics['Easy']['norm_gap'])} / "
            f"{fmt('auc', metrics['Easy']['auc'])}"
        )

    print("\nExtra models:")
    for display_name, _ in EXTRA_MODELS:
        print(f"  {display_name}:")
        agent_names = AGENTS
        for agent_name in agent_names:
            metrics = extra_model_results.get(display_name, {}).get(agent_name, empty_split_metrics())
            print(
                f"    {agent_name:<24} "
                f"Overall: {fmt('pair_acc', metrics['Overall']['pair_acc'])} / "
                f"{fmt('norm_gap', metrics['Overall']['norm_gap'])} / "
                f"{fmt('auc', metrics['Overall']['auc'])}  "
                f"Hard: {fmt('pair_acc', metrics['Hard']['pair_acc'])} / "
                f"{fmt('norm_gap', metrics['Hard']['norm_gap'])} / "
                f"{fmt('auc', metrics['Hard']['auc'])}"
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
