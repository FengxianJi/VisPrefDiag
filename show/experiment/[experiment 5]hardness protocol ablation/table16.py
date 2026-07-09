#!/usr/bin/env python3
"""
table16.py - Unified appendix table for four hardness protocols on ImageReward.

This table is a pair-level behavioral audit. It compares four protocol
families under the same downstream metrics:

1. Pair-gap (ours): pair-level gap bins from table13
2. Human-only: group-level human rank-gap bins
3. Model-only: group-level oracle-gap tertiles
4. Dual-filter (PrefHard): intersection-style group protocol

For the three group-level protocols, groups are first assigned to Hard/Mid/Easy,
and then all valid pairs inside those selected groups inherit the group's label.
We then evaluate every protocol using the same pair-level metrics:

- PairAcc / AUROC

This makes the table suitable for checking whether each protocol induces a clear
behavioral difficulty gradient (Hard < Mid < Easy).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from table13 import (
    AGENT_DIRS,
    AGENTS,
    BASELINES,
    BASELINE_DETAIL,
    QALIGN_MODES,
    _build_pair_records,
    contamination_reason,
    diagnose_order_template_contamination,
    _load_split_agent,
    _safe_load,
    _summarize_split,
    build_benchmark_maps,
    build_human_pair_metrics,
)


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = Path(__file__).resolve().parent
TABLE13_DIR = TABLE_DIR / "table13"
BENCHMARK_SPLITS = [
    ROOT / "benchmark" / "ImageReward" / "test_1.json",
    ROOT / "benchmark" / "ImageReward" / "test_2.json",
]
OUTPUT_PATH = TABLE_DIR / "table16.tex"

SPLITS = ["Overall", "Hard", "Mid", "Easy"]
BIN_SPLITS = ["Hard", "Mid", "Easy"]
PROTOCOL_ORDER = [
    ("Pair-gap", "Pair-gap (ours)"),
    ("Human-only", "Human-only (group gap)"),
    ("Model-only", "Model-only (feature gap)"),
    ("Dual-filter", "Dual-filter (PrefHard)"),
]
METRIC_KEYS = ["pair_acc", "auc"]

MODEL_SECTIONS = [
    ("Qwen3VL-4B", "qwen3vl4b"),
    ("Qwen3VL-8B", "qwen3vl8b"),
    ("Qwen3VL-30B", "qwen3vl30b"),
    ("Kimi-K2.5", "kimi-k2.5"),
    ("Gemini-3-Flash", "gemini-3-flash"),
    ("Llama3.2V-11B", "llama3.2V-11b"),
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def human_bin_from_delta(delta: int) -> str:
    mapping = {2: "Hard", 3: "Mid", 4: "Easy"}
    if delta in mapping:
        return mapping[delta]
    if delta < 3:
        return "Hard"
    if delta == 3:
        return "Mid"
    return "Easy"


def build_benchmark_meta() -> dict[str, dict]:
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for split_path in BENCHMARK_SPLITS:
        for row in load_json(split_path):
            grouped_rows[row["prompt_id"]].append(row)

    benchmark_meta = {}
    for gid, rows in grouped_rows.items():
        sorted_rows = sorted(rows, key=lambda row: Path(row["image_path"]).name)
        order = [Path(row["image_path"]).name for row in sorted_rows]
        ranking = [row["rank"] for row in sorted_rows]
        human_delta = max(ranking) - min(ranking)
        benchmark_meta[gid] = {
            "group_id": gid,
            "prompt_text": sorted_rows[0]["prompt"],
            "order": order,
            "rank_by_name": {
                name: rank
                for name, rank in zip(order, ranking)
            },
            "ranking": ranking,
            "human_delta": human_delta,
            "human_bin": human_bin_from_delta(human_delta),
        }

    return benchmark_meta


def empty_pair_metric() -> dict:
    return {
        "pair_acc": None,
        "auc": None,
        "n_groups": 0,
        "n_pairs": 0,
    }


def empty_protocol_metrics() -> dict[str, dict[str, dict]]:
    return {
        protocol_id: {
            split: empty_pair_metric()
            for split in SPLITS
        }
        for protocol_id, _ in PROTOCOL_ORDER
    }


def attach_protocol_warning(metrics: dict, reason: str, diagnostics: dict) -> dict:
    metrics["__warning__"] = True
    metrics["__warning_reason"] = reason
    metrics["__diagnostics"] = diagnostics
    return metrics


def summarize_pair_entries(pair_entries: list[dict], n_groups: int) -> dict:
    raw = _summarize_split(pair_entries, n_groups)
    return {
        "pair_acc": raw["pair_acc"],
        "auc": raw["auc"],
        "n_groups": raw["n_groups"],
        "n_pairs": raw["n_pairs"],
    }


def assign_equal_frequency_bins(records: list[dict], value_key: str) -> dict[str, list[dict]]:
    ordered = sorted(records, key=lambda row: (row[value_key], row["group_id"]))
    total = len(ordered)
    buckets = {label: [] for label in BIN_SPLITS}
    if total == 0:
        return buckets
    for idx, row in enumerate(ordered):
        bucket_id = min(len(BIN_SPLITS) - 1, idx * len(BIN_SPLITS) // total)
        buckets[BIN_SPLITS[bucket_id]].append(row)
    return buckets


def _align_group_to_benchmark(group, gt_key: str, benchmark_meta: dict[str, dict]):
    scores = group.get("scores", [])
    gt_ranks = group.get(gt_key, [])
    gid = group.get("group_id")
    img_paths = group.get("img_paths", [])

    if gid and gid in benchmark_meta and img_paths and len(scores) == len(img_paths):
        meta = benchmark_meta[gid]
        score_by_name = {
            Path(img_path).name: score
            for img_path, score in zip(img_paths, scores)
        }
        order = meta["order"]
        rank_by_name = meta["rank_by_name"]
        if all(name in score_by_name and name in rank_by_name for name in order):
            return (
                [score_by_name[name] for name in order],
                [rank_by_name[name] for name in order],
            )
    return scores, gt_ranks


def compute_oracle_gap(scores: list[float], gt_ranks: list[int]) -> float:
    paired = sorted(zip(gt_ranks, scores), key=lambda x: (x[0],))
    if len(paired) < 2:
        return 0.0
    return float(paired[0][1] - paired[1][1])


def load_group_records(groups: list[dict], gt_key: str, benchmark_meta: dict[str, dict]) -> list[dict]:
    records = []
    for group in groups:
        gid = group.get("group_id")
        if gid not in benchmark_meta:
            continue
        scores, gt_ranks = _align_group_to_benchmark(group, gt_key, benchmark_meta)
        if not scores or not gt_ranks or len(scores) != len(gt_ranks):
            continue
        records.append(
            {
                "group_id": gid,
                "scores": scores,
                "gt_ranks": gt_ranks,
                "human_bin": benchmark_meta[gid]["human_bin"],
                "oracle_gap": compute_oracle_gap(scores, gt_ranks),
            }
        )
    return records


def summarize_group_bucket(records: list[dict], group_pair_records: dict[str, list[dict]]) -> dict:
    pair_entries = []
    group_ids = set()
    for record in records:
        gid = record["group_id"]
        pairs = group_pair_records.get(gid, [])
        if not pairs:
            continue
        pair_entries.extend(pairs)
        group_ids.add(gid)
    return summarize_pair_entries(pair_entries, len(group_ids))


def build_group_protocol_metrics(records: list[dict]) -> dict[str, dict[str, dict]]:
    results = {
        protocol_id: {
            split: empty_pair_metric()
            for split in SPLITS
        }
        for protocol_id, _ in PROTOCOL_ORDER
        if protocol_id != "Pair-gap"
    }
    if not records:
        return results

    overall_pairs = []
    overall_group_ids = set()
    group_pair_records = {}

    for record in records:
        gid = record["group_id"]
        pair_records = _build_pair_records(record["scores"], record["gt_ranks"])
        if not pair_records:
            continue
        group_pair_records[gid] = pair_records
        overall_pairs.extend(pair_records)
        overall_group_ids.add(gid)

    overall_metric = summarize_pair_entries(overall_pairs, len(overall_group_ids))

    human_only = {label: [] for label in BIN_SPLITS}
    for record in records:
        human_only[record["human_bin"]].append(record)

    model_only = assign_equal_frequency_bins(records, "oracle_gap")
    model_label_by_gid = {
        row["group_id"]: label
        for label, rows in model_only.items()
        for row in rows
    }

    dual_filter = {label: [] for label in BIN_SPLITS}
    for record in records:
        human_bin = record["human_bin"]
        model_bin = model_label_by_gid.get(record["group_id"])
        if model_bin is None:
            continue
        if human_bin == "Hard" and model_bin == "Hard":
            dual_filter["Hard"].append(record)
        elif human_bin == "Easy" and model_bin == "Easy":
            dual_filter["Easy"].append(record)
        else:
            dual_filter["Mid"].append(record)

    protocol_buckets = {
        "Human-only": human_only,
        "Model-only": model_only,
        "Dual-filter": dual_filter,
    }
    for protocol_id, buckets in protocol_buckets.items():
        results[protocol_id]["Overall"] = dict(overall_metric)
        for split in BIN_SPLITS:
            results[protocol_id][split] = summarize_group_bucket(
                buckets[split],
                group_pair_records,
            )
    return results


def build_method_protocol_metrics(
    groups: list[dict],
    gt_key: str,
    benchmark_meta: dict[str, dict],
    pair_benchmark_map: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    if not groups:
        return empty_protocol_metrics()

    protocol_metrics = empty_protocol_metrics()

    pair_gap_metrics = build_human_pair_metrics(groups, gt_key, pair_benchmark_map)
    for split in SPLITS:
        protocol_metrics["Pair-gap"][split] = {
            "pair_acc": pair_gap_metrics[split]["pair_acc"],
            "auc": pair_gap_metrics[split]["auc"],
            "n_groups": pair_gap_metrics[split]["n_groups"],
            "n_pairs": pair_gap_metrics[split]["n_pairs"],
        }

    group_records = load_group_records(groups, gt_key, benchmark_meta)
    group_protocol_metrics = build_group_protocol_metrics(group_records)
    for protocol_id in ("Human-only", "Model-only", "Dual-filter"):
        protocol_metrics[protocol_id] = group_protocol_metrics[protocol_id]

    return protocol_metrics


def load_baseline_results(
    benchmark_meta: dict[str, dict],
    pair_benchmark_map: dict[str, dict],
) -> dict[str, dict[str, dict[str, dict]]]:
    results = {}

    for name, fname in BASELINE_DETAIL.items():
        path = TABLE13_DIR / "classic" / fname
        payload = _safe_load(path)
        if not payload:
            results[name] = empty_protocol_metrics()
            continue
        groups = payload.get("test", {}).get("details", [])
        results[name] = build_method_protocol_metrics(
            groups,
            "gt_ranks",
            benchmark_meta,
            pair_benchmark_map,
        )

    for suffix, display_name in QALIGN_MODES:
        path = TABLE13_DIR / "classic" / f"{suffix}_detail.json"
        payload = _safe_load(path)
        if not payload:
            results[display_name] = empty_protocol_metrics()
            continue
        groups = payload.get("test", {}).get("details", [])
        results[display_name] = build_method_protocol_metrics(
            groups,
            "gt_ranks",
            benchmark_meta,
            pair_benchmark_map,
        )

    return results


def load_agent_model_results(
    model_dirname: str,
    benchmark_meta: dict[str, dict],
    pair_benchmark_map: dict[str, dict],
) -> dict[str, dict[str, dict[str, dict]]]:
    results = {}
    model_root = TABLE13_DIR / model_dirname
    for method_name, dirname in AGENT_DIRS.items():
        path = model_root / dirname / "test_original.json"
        groups = _load_split_agent(path)
        if not groups:
            results[method_name] = empty_protocol_metrics()
            continue
        diagnostics = diagnose_order_template_contamination(
            groups,
            "ranks",
            pair_benchmark_map,
        )
        method_metrics = build_method_protocol_metrics(
            groups,
            "ranks",
            benchmark_meta,
            pair_benchmark_map,
        )
        if diagnostics["contaminated"]:
            reason = contamination_reason(diagnostics)
            print(
                f"Warning: flagged {model_dirname} / {method_name}: {reason}"
            )
            method_metrics = attach_protocol_warning(method_metrics, reason, diagnostics)
        results[method_name] = method_metrics
    return results


def gather_all_results(benchmark_meta: dict[str, dict], pair_benchmark_map: dict[str, dict]):
    baseline_results = load_baseline_results(benchmark_meta, pair_benchmark_map)
    model_results = {
        display_name: load_agent_model_results(
            model_dirname,
            benchmark_meta,
            pair_benchmark_map,
        )
        for display_name, model_dirname in MODEL_SECTIONS
    }
    return baseline_results, model_results


def iter_metric_values(baseline_results, model_results, protocol_id: str, split: str, metric_key: str):
    for method_name in BASELINES:
        method_metrics = baseline_results.get(method_name, {})
        value = (
            method_metrics
            .get(protocol_id, {})
            .get(split, {})
            .get(metric_key)
        )
        if value is not None:
            yield value
    for display_name, _ in MODEL_SECTIONS:
        for method_name in AGENTS:
            method_metrics = model_results.get(display_name, {}).get(method_name, {})
            value = (
                method_metrics
                .get(protocol_id, {})
                .get(split, {})
                .get(metric_key)
            )
            if value is not None:
                yield value


def compute_best_map(baseline_results, model_results):
    best = {}
    for protocol_id, _ in PROTOCOL_ORDER:
        for split in SPLITS:
            for metric_key in METRIC_KEYS:
                values = list(
                    iter_metric_values(
                        baseline_results,
                        model_results,
                        protocol_id,
                        split,
                        metric_key,
                    )
                )
                best[(protocol_id, split, metric_key)] = max(values) if values else None
    return best


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}"


def fmt_pair(metrics: dict, protocol_id: str, split: str, best_map: dict) -> str:
    pair_acc = metrics["pair_acc"]
    auc = metrics["auc"]
    if pair_acc is None and auc is None:
        return "--"

    pair_acc_str = fmt_pct(pair_acc)
    auc_str = fmt_pct(auc)

    best_pair_acc = best_map.get((protocol_id, split, "pair_acc"))
    if pair_acc is not None and best_pair_acc is not None and abs(pair_acc - best_pair_acc) < 1e-12:
        pair_acc_str = rf"\textbf{{{pair_acc_str}}}"

    best_auc = best_map.get((protocol_id, split, "auc"))
    if auc is not None and best_auc is not None and abs(auc - best_auc) < 1e-12:
        auc_str = rf"\textbf{{{auc_str}}}"

    return f"{pair_acc_str} / {auc_str}"


def row_line(method_name: str, protocol_metrics: dict[str, dict[str, dict]], best_map: dict) -> str:
    cells = []
    for protocol_id, _ in PROTOCOL_ORDER:
        for split in SPLITS:
            cells.append(fmt_pair(protocol_metrics[protocol_id][split], protocol_id, split, best_map))
    return f"{method_name:<28} & " + " & ".join(cells) + r" \\"


def generate_table(baseline_results, model_results, pair_counts) -> str:
    best_map = compute_best_map(baseline_results, model_results)

    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.62}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * (len(PROTOCOL_ORDER) * len(SPLITS)) + "}")
    lines.append(r"\toprule")
    lines.append(
        r"\multirow{3}{*}{\textbf{Method}}"
        + "".join(
            rf" & \multicolumn{{4}}{{c}}{{\textbf{{{protocol_label}}}}}"
            for _, protocol_label in PROTOCOL_ORDER
        )
        + r" \\"
    )
    cmidrules = []
    start = 2
    for _ in PROTOCOL_ORDER:
        end = start + 3
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append("".join(cmidrules))
    lines.append(
        r" & \textbf{Overall} & \textbf{Hard} & \textbf{Mid} & \textbf{Easy}"
        r" & \textbf{Overall} & \textbf{Hard} & \textbf{Mid} & \textbf{Easy}"
        r" & \textbf{Overall} & \textbf{Hard} & \textbf{Mid} & \textbf{Easy}"
        r" & \textbf{Overall} & \textbf{Hard} & \textbf{Mid} & \textbf{Easy} \\"
    )
    lines.append(
        r" & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC}"
        r" & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC}"
        r" & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC}"
        r" & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} & \textbf{PairAcc / AUROC} \\"
    )
    lines.append(r"\midrule")

    lines.append(r"\multicolumn{17}{c}{\textbf{Visual reward models}} \\")
    lines.append(r"\midrule")
    for method_name in BASELINES:
        lines.append(
            row_line(
                method_name,
                baseline_results.get(method_name, empty_protocol_metrics()),
                best_map,
            )
        )

    for display_name, _ in MODEL_SECTIONS:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{17}}{{c}}{{\textbf{{{display_name}}}}} \\")
        lines.append(r"\midrule")
        for method_name in AGENTS:
            lines.append(
                row_line(
                    method_name,
                    model_results[display_name].get(method_name, empty_protocol_metrics()),
                    best_map,
                )
            )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        r"\caption{\textbf{Unified pair-level audit of four hardness protocols on ImageReward (test).} "
        r"We compare four protocol families under the same downstream metrics, reporting \textbf{PairAcc / AUROC} in each cell. "
        r"\textbf{Pair-gap (ours)} directly bins valid image pairs by the human level gap $d(i,j)=|level_i-level_j|$, "
        rf"with fixed dataset-wide pair counts Hard = {pair_counts['Hard']}, Mid = {pair_counts['Mid']}, and Easy = {pair_counts['Easy']}. "
        r"\textbf{Human-only (group gap)} first bins each prompt group by the group-level human rank gap "
        r"$\Delta = \max(\mathrm{rank}) - \min(\mathrm{rank})$. "
        r"\textbf{Model-only (feature gap)} uses equal-frequency tertiles of the method-specific oracle gap "
        r"$\Delta_s = s(\mathrm{human\_top1}) - s(\mathrm{human\_top2})$. "
        r"\textbf{Dual-filter (PrefHard)} takes the intersection-style group protocol derived from Human-only and Model-only. "
        r"For the three group-level protocols, we assign Hard/Mid/Easy at the group level and then audit \emph{all valid pairs} inside the selected groups using the same pair-level metrics. "
        r"This lets us test whether each protocol induces a meaningful behavioral difficulty gradient rather than relying on group-level top-rank recovery alone. "
        r"The \textbf{Overall} column always reuses the full pair set, so it is numerically identical across the four protocols. "
        r"Best PairAcc and best AUROC in each column are shown in \textbf{bold}. "
        r"\textit{Caution:} the Llama3.2V-11B Single-agent (Pair) row is flagged by an order-template contamination diagnostic "
        r"(monotonic saved ranks, low top-1, repeated position-like rankings), so its unusually high pairwise scores should be interpreted as potentially inflated rather than clean model gains.}"
    )
    lines.append(r"\label{tab:hardness_protocols_full}")
    lines.append(r"\end{table*}")
    lines.append(r"\FloatBarrier")
    return "\n".join(lines) + "\n"


def print_summary(baseline_results, model_results):
    print("=" * 88)
    print("Table 16 — Unified pair-level audit for four hardness protocols")
    print("=" * 88)
    anchors = [
        ("ImageReward", baseline_results.get("ImageReward", empty_protocol_metrics())),
        (
            "Qwen3VL-4B / Single-agent (Pair)",
            model_results["Qwen3VL-4B"].get("Single-agent (Pair)", empty_protocol_metrics()),
        ),
        (
            "Qwen3VL-30B / Valuator",
            model_results["Qwen3VL-30B"].get("Valuator", empty_protocol_metrics()),
        ),
    ]
    for label, metrics in anchors:
        print(f"\n[{label}]")
        for protocol_id, protocol_label in PROTOCOL_ORDER:
            hard = metrics[protocol_id]["Hard"]
            print(
                f"  {protocol_label:<24} "
                f"Hard(n_groups={hard['n_groups']:<3}, n_pairs={hard['n_pairs']:<4}) "
                f"PairAcc={fmt_pct(hard['pair_acc'])} "
                f"AUROC={fmt_pct(hard['auc'])}"
            )


def main() -> None:
    pair_counts, pair_benchmark_map = build_benchmark_maps()
    benchmark_meta = build_benchmark_meta()
    print("Pair-gap benchmark bins:", pair_counts)
    print("Group-gap benchmark bins:", Counter(meta["human_bin"] for meta in benchmark_meta.values()))

    baseline_results, model_results = gather_all_results(benchmark_meta, pair_benchmark_map)
    print_summary(baseline_results, model_results)

    latex = generate_table(baseline_results, model_results, pair_counts)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"\nSaved LaTeX table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
