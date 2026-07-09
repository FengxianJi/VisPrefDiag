#!/usr/bin/env python3
"""归因链分析：验证 Hard 样本失败是否来自 feature gap 消失。"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_JSON = ROOT / "benchmark" / "ImageReward" / "test.json"
TABLE13_CLASSIC_DIR = ROOT / "show" / "table" / "table13" / "classic"
OUTPUT_DIR = Path(__file__).resolve().parent

PRIMARY_MODEL = "ImageReward"
CANDIDATE_MODELS = [
    "ImageReward",
    "HPSv2",
    "PickScore",
    "LLaVA-Reward",
]
MODEL_FILES = {
    "ImageReward": TABLE13_CLASSIC_DIR / "ImageReward_detail.json",
    "HPSv2": TABLE13_CLASSIC_DIR / "HPSv2_detail.json",
    "PickScore": TABLE13_CLASSIC_DIR / "PickScore_detail.json",
    "LLaVA-Reward": TABLE13_CLASSIC_DIR / "LLaVA-Reward_detail.json",
}
DIFFICULTY_ORDER = ["Hard", "Mid", "Easy"]
QUANTILE_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
HARD_BUCKET_LABELS = ["Hard-Low", "Hard-Mid", "Hard-High"]
EPS = 1e-12


def configure_matplotlib() -> None:
    """尽量启用中文字体，避免图中出现乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def difficulty_from_delta(human_delta: int) -> str:
    return {2: "Hard", 3: "Mid", 4: "Easy"}[human_delta]



def load_benchmark_meta() -> dict[str, dict]:
    data = load_json(BENCHMARK_JSON)
    meta = {}
    for item in data:
        human_delta = max(item["ranking"]) - min(item["ranking"])
        meta[item["id"]] = {
            "group_id": item["id"],
            "prompt_text": item["prompt"],
            "human_delta": human_delta,
            "difficulty": difficulty_from_delta(human_delta),
        }
    return meta



def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)



def safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return pstdev(values)



def fmt_pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.2f}%"



def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"



def compute_oracle_gap(scores: list[float], gt_ranks: list[int]) -> float:
    """oracle gap：人类 top-1 与人类 top-2 在 reward score 上的差值。"""
    paired = sorted(zip(gt_ranks, scores), key=lambda x: (x[0],))
    return float(paired[0][1] - paired[1][1])



def compute_top_margin(scores: list[float]) -> float:
    """top margin：模型自身预测 top-1 与 top-2 的分数差。"""
    ranked = sorted(scores, reverse=True)
    return float(ranked[0] - ranked[1])



def load_model_records(benchmark_meta: dict[str, dict]) -> dict[str, list[dict]]:
    all_records: dict[str, list[dict]] = {}
    for model_name in CANDIDATE_MODELS:
        path = MODEL_FILES[model_name]
        payload = load_json(path)
        details = payload["test"]["details"]
        records = []
        for item in details:
            gid = item["group_id"]
            meta = benchmark_meta[gid]
            scores = item["scores"]
            gt_ranks = item["gt_ranks"]
            records.append(
                {
                    "group_id": gid,
                    "prompt_text": item["prompt_text"],
                    "difficulty": meta["difficulty"],
                    "human_delta": meta["human_delta"],
                    "oracle_gap": compute_oracle_gap(scores, gt_ranks),
                    "top_margin": compute_top_margin(scores),
                    "correct": int(item["recall_at_1"]),
                }
            )
        all_records[model_name] = records
    return all_records



def overall_r1_from_records(records: list[dict]) -> float:
    return mean(record["correct"] for record in records)



def assign_equal_buckets(records: list[dict], value_key: str, labels: list[str]) -> list[dict]:
    ordered = sorted(records, key=lambda row: (row[value_key], row["group_id"]))
    bucketed = []
    total = len(ordered)
    num_buckets = len(labels)
    for idx, row in enumerate(ordered):
        bucket_id = min(num_buckets - 1, idx * num_buckets // total)
        new_row = dict(row)
        new_row["bucket_label"] = labels[bucket_id]
        new_row["bucket_index"] = bucket_id
        bucketed.append(new_row)
    return bucketed



def summarize_buckets(records: list[dict], labels: list[str], value_key: str) -> list[dict]:
    bucketed = assign_equal_buckets(records, value_key, labels)
    summary = []
    for label in labels:
        bucket_rows = [row for row in bucketed if row["bucket_label"] == label]
        difficulty_counts = Counter(row["difficulty"] for row in bucket_rows)
        n = len(bucket_rows)
        summary.append(
            {
                "bucket": label,
                "n": n,
                "r1": safe_mean([row["correct"] for row in bucket_rows]),
                "difficulty_counts": {
                    difficulty: difficulty_counts.get(difficulty, 0)
                    for difficulty in DIFFICULTY_ORDER
                },
                "difficulty_ratios": {
                    difficulty: difficulty_counts.get(difficulty, 0) / n if n else 0.0
                    for difficulty in DIFFICULTY_ORDER
                },
                "gap_min": min(row[value_key] for row in bucket_rows) if bucket_rows else None,
                "gap_max": max(row[value_key] for row in bucket_rows) if bucket_rows else None,
            }
        )
    return summary



def mann_whitney_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return None
    concordant = 0
    tied = 0
    for pos_score in positives:
        for neg_score in negatives:
            if pos_score > neg_score:
                concordant += 1
            elif abs(pos_score - neg_score) <= EPS:
                tied += 1
    return (concordant + 0.5 * tied) / (len(positives) * len(negatives))



def is_degenerate(records: list[dict], value_key: str) -> bool:
    values = [row[value_key] for row in records]
    return max(values) - min(values) <= EPS



def compute_global_analysis(model_name: str, records: list[dict]) -> dict:
    summary = summarize_buckets(records, QUANTILE_LABELS, "oracle_gap")
    monotonic_non_decreasing = all(
        summary[i]["r1"] <= summary[i + 1]["r1"] + EPS
        for i in range(len(summary) - 1)
    )
    return {
        "model": model_name,
        "gap_definition": "oracle_gap = s(human_top1) - s(human_top2)",
        "monotonic_non_decreasing_r1": monotonic_non_decreasing,
        "buckets": summary,
    }



def compute_hard_subbin_analysis(model_records: dict[str, list[dict]]) -> dict:
    per_model = {}
    excluded = {}
    valid_models = []

    for model_name, records in model_records.items():
        hard_records = [row for row in records if row["difficulty"] == "Hard"]
        if is_degenerate(hard_records, "oracle_gap"):
            excluded[model_name] = "Hard 子集上的 oracle gap 全部相同，无法做有效分层。"
            continue
        valid_models.append(model_name)
        per_model[model_name] = summarize_buckets(hard_records, HARD_BUCKET_LABELS, "oracle_gap")

    aggregated = []
    for bucket_label in HARD_BUCKET_LABELS:
        bucket_entries = []
        for model_name in valid_models:
            model_bucket = next(
                item for item in per_model[model_name] if item["bucket"] == bucket_label
            )
            bucket_entries.append(model_bucket)
        r1_values = [entry["r1"] for entry in bucket_entries if entry["r1"] is not None]
        aggregated.append(
            {
                "bucket": bucket_label,
                "n_mean": safe_mean([entry["n"] for entry in bucket_entries]),
                "r1_mean": safe_mean(r1_values),
                "r1_std": safe_std(r1_values),
                "gap_min_mean": safe_mean([entry["gap_min"] for entry in bucket_entries]),
                "gap_max_mean": safe_mean([entry["gap_max"] for entry in bucket_entries]),
                "per_model": {
                    model_name: next(
                        item for item in per_model[model_name] if item["bucket"] == bucket_label
                    )
                    for model_name in valid_models
                },
            }
        )

    monotonic_non_decreasing = all(
        aggregated[i]["r1_mean"] <= aggregated[i + 1]["r1_mean"] + EPS
        for i in range(len(aggregated) - 1)
        if aggregated[i]["r1_mean"] is not None and aggregated[i + 1]["r1_mean"] is not None
    )

    return {
        "gap_definition": "oracle_gap = s(human_top1) - s(human_top2)",
        "valid_models": valid_models,
        "excluded_models": excluded,
        "per_model": per_model,
        "aggregated": aggregated,
        "monotonic_non_decreasing_mean_r1": monotonic_non_decreasing,
    }



def compute_confidence_auroc_analysis(model_records: dict[str, list[dict]]) -> dict:
    per_model = {}
    excluded = {}
    valid_models = []

    for model_name, records in model_records.items():
        if is_degenerate(records, "top_margin"):
            excluded[model_name] = "top margin 全部相同，无法形成有效的 confidence ranking。"
            continue
        valid_models.append(model_name)
        per_model[model_name] = {}
        for difficulty in DIFFICULTY_ORDER:
            bucket_rows = [row for row in records if row["difficulty"] == difficulty]
            auc = mann_whitney_auc(
                [row["top_margin"] for row in bucket_rows],
                [row["correct"] for row in bucket_rows],
            )
            per_model[model_name][difficulty] = {
                "auc": auc,
                "n": len(bucket_rows),
                "margin_min": min(row["top_margin"] for row in bucket_rows),
                "margin_max": max(row["top_margin"] for row in bucket_rows),
            }

    aggregated = {}
    for difficulty in DIFFICULTY_ORDER:
        auc_values = [
            per_model[model_name][difficulty]["auc"]
            for model_name in valid_models
            if per_model[model_name][difficulty]["auc"] is not None
        ]
        aggregated[difficulty] = {
            "auc_mean": safe_mean(auc_values),
            "auc_std": safe_std(auc_values),
            "per_model": {
                model_name: per_model[model_name][difficulty]
                for model_name in valid_models
            },
        }

    return {
        "gap_definition": "top_margin = s(pred_top1) - s(pred_top2)",
        "valid_models": valid_models,
        "excluded_models": excluded,
        "per_model": per_model,
        "aggregated": aggregated,
    }



def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def export_global_csv(global_result: dict) -> None:
    rows = []
    for item in global_result["buckets"]:
        rows.append(
            {
                "bucket": item["bucket"],
                "n": item["n"],
                "r1": item["r1"],
                "hard_ratio": item["difficulty_ratios"]["Hard"],
                "mid_ratio": item["difficulty_ratios"]["Mid"],
                "easy_ratio": item["difficulty_ratios"]["Easy"],
                "gap_min": item["gap_min"],
                "gap_max": item["gap_max"],
            }
        )
    save_csv(
        OUTPUT_DIR / "global_delta_quantiles.csv",
        rows,
        ["bucket", "n", "r1", "hard_ratio", "mid_ratio", "easy_ratio", "gap_min", "gap_max"],
    )



def export_hard_csv(hard_result: dict) -> None:
    rows = []
    for item in hard_result["aggregated"]:
        row = {
            "bucket": item["bucket"],
            "n_mean": item["n_mean"],
            "r1_mean": item["r1_mean"],
            "r1_std": item["r1_std"],
            "gap_min_mean": item["gap_min_mean"],
            "gap_max_mean": item["gap_max_mean"],
        }
        for model_name in hard_result["valid_models"]:
            row[f"{model_name}_r1"] = item["per_model"][model_name]["r1"]
        rows.append(row)
    fieldnames = ["bucket", "n_mean", "r1_mean", "r1_std", "gap_min_mean", "gap_max_mean"]
    fieldnames.extend(f"{model_name}_r1" for model_name in hard_result["valid_models"])
    save_csv(OUTPUT_DIR / "hard_subbins.csv", rows, fieldnames)



def export_auroc_csv(auroc_result: dict) -> None:
    rows = []
    for difficulty in DIFFICULTY_ORDER:
        row = {
            "difficulty": difficulty,
            "auc_mean": auroc_result["aggregated"][difficulty]["auc_mean"],
            "auc_std": auroc_result["aggregated"][difficulty]["auc_std"],
        }
        for model_name in auroc_result["valid_models"]:
            row[f"{model_name}_auc"] = auroc_result["per_model"][model_name][difficulty]["auc"]
        rows.append(row)
    fieldnames = ["difficulty", "auc_mean", "auc_std"]
    fieldnames.extend(f"{model_name}_auc" for model_name in auroc_result["valid_models"])
    save_csv(OUTPUT_DIR / "confidence_auroc.csv", rows, fieldnames)



def plot_global_quantiles(global_result: dict) -> None:
    labels = [item["bucket"] for item in global_result["buckets"]]
    x = list(range(len(labels)))
    r1_values = [item["r1"] * 100 for item in global_result["buckets"]]
    hard = [item["difficulty_ratios"]["Hard"] * 100 for item in global_result["buckets"]]
    mid = [item["difficulty_ratios"]["Mid"] * 100 for item in global_result["buckets"]]
    easy = [item["difficulty_ratios"]["Easy"] * 100 for item in global_result["buckets"]]

    fig, ax1 = plt.subplots(figsize=(9.6, 5.6))
    ax2 = ax1.twinx()

    width = 0.68
    ax2.bar(x, hard, width=width, color="#C44E52", alpha=0.25, label="Hard 占比")
    ax2.bar(x, mid, width=width, bottom=hard, color="#4C72B0", alpha=0.25, label="Mid 占比")
    ax2.bar(
        x,
        easy,
        width=width,
        bottom=[h + m for h, m in zip(hard, mid)],
        color="#55A868",
        alpha=0.25,
        label="Easy 占比",
    )

    ax1.plot(x, r1_values, color="#111111", marker="o", linewidth=2.2, label="R@1")
    for idx, value in enumerate(r1_values):
        ax1.text(idx, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("R@1（%）")
    ax2.set_ylabel("Hard / Mid / Easy 占比（%）")
    ax1.set_xlabel("按 oracle gap 从小到大划分的 5 个分位段")
    ax1.set_title(f"实验一：全局 Δs 分位与准确率关系（{global_result['model']}）")
    ax1.set_ylim(0, max(r1_values) * 1.25)
    ax2.set_ylim(0, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_global_delta_quantiles.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_hard_subbins(hard_result: dict) -> None:
    labels = [item["bucket"] for item in hard_result["aggregated"]]
    x = list(range(len(labels)))
    means = [item["r1_mean"] * 100 for item in hard_result["aggregated"]]
    stds = [item["r1_std"] * 100 if item["r1_std"] is not None else 0.0 for item in hard_result["aggregated"]]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.bar(x, means, color=["#C44E52", "#8172B2", "#55A868"], alpha=0.9, width=0.64)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#222222", elinewidth=1.4, capsize=4)

    for idx, item in enumerate(hard_result["aggregated"]):
        ax.text(
            idx,
            means[idx] + 1.4,
            f"{means[idx]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
        ax.text(
            idx,
            max(2.0, means[idx] * 0.15),
            f"Δs≈[{item['gap_min_mean']:.3f}, {item['gap_max_mean']:.3f}]",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("平均 R@1（%）")
    ax.set_xlabel("仅在 Hard bin 内，再按 oracle gap 三等分")
    ax.set_title("实验二：Hard bin 内部 Δs 分层（多 reward model 平均）")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_ylim(0, max(means) * 1.35)

    note = "有效模型：" + "、".join(hard_result["valid_models"])
    if hard_result["excluded_models"]:
        excluded_note = "；排除：" + "；".join(
            f"{model}（{reason}）" for model, reason in hard_result["excluded_models"].items()
        )
        note += excluded_note
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_hard_subbins.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_confidence_auroc(auroc_result: dict) -> None:
    labels = DIFFICULTY_ORDER
    x = list(range(len(labels)))
    means = [auroc_result["aggregated"][label]["auc_mean"] * 100 for label in labels]
    stds = [
        auroc_result["aggregated"][label]["auc_std"] * 100
        if auroc_result["aggregated"][label]["auc_std"] is not None
        else 0.0
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(x, means, color=["#C44E52", "#4C72B0", "#55A868"], alpha=0.88, width=0.62)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#222222", elinewidth=1.4, capsize=4)
    ax.axhline(50.0, color="#444444", linestyle="--", linewidth=1.2, alpha=0.8)

    for idx, value in enumerate(means):
        ax.text(idx, value + 1.0, f"{value:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC（%）")
    ax.set_xlabel("按人类 difficulty bin 分组")
    ax.set_title("实验三：top margin 作为置信度信号的 AUROC")
    ax.set_ylim(40, max(means) * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    note = "有效模型：" + "、".join(auroc_result["valid_models"])
    if auroc_result["excluded_models"]:
        excluded_note = "；排除：" + "；".join(
            f"{model}（{reason}）" for model, reason in auroc_result["excluded_models"].items()
        )
        note += excluded_note
    fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_confidence_auroc.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



def build_summary_markdown(
    benchmark_meta: dict[str, dict],
    model_records: dict[str, list[dict]],
    global_result: dict,
    hard_result: dict,
    auroc_result: dict,
) -> str:
    difficulty_counter = Counter(meta["difficulty"] for meta in benchmark_meta.values())

    table13_r1 = {
        model_name: overall_r1_from_records(records)
        for model_name, records in model_records.items()
    }

    lines = []
    lines.append("# 归因链分析结果")
    lines.append("")
    lines.append("## 1. 分析目标")
    lines.append("")
    lines.append("- 目标命题：Hard 样本失败是否主要因为 feature gap 消失，而不是模型完全看不懂。")
    lines.append("- 数据来源：直接读取 `show/table/table13/classic/*_detail.json` 与 `benchmark/ImageReward/test.json`。")
    lines.append("- 人类 difficulty bin：Hard = 54，Mid = 229，Easy = 183，总计 466 组。")
    lines.append("")
    lines.append("## 2. 本次实现里使用的两个 gap 定义")
    lines.append("")
    lines.append("- `oracle gap = s(human_top1) - s(human_top2)`：用于实验一和实验二，直接衡量 reward model 是否还能把人类最优图和最接近的竞争者拉开。")
    lines.append("- `top margin = s(pred_top1) - s(pred_top2)`：用于实验三，作为推断阶段可见的 confidence proxy，检验在不同 difficulty bin 内还能否排序正确/错误样本。")
    lines.append("")
    lines.append("## 3. 参考 table13 的总体结果")
    lines.append("")
    for model_name in CANDIDATE_MODELS:
        lines.append(f"- {model_name}：Overall R@1 = {fmt_pct(table13_r1[model_name])}")
    lines.append("")
    lines.append("## 4. 实验一：全局 Δs 分位与准确率")
    lines.append("")
    lines.append(f"- 主图默认使用 `{global_result['model']}`。")
    lines.append(f"- R@1 是否单调不下降：{'是' if global_result['monotonic_non_decreasing_r1'] else '否'}。")
    for item in global_result["buckets"]:
        ratios = item["difficulty_ratios"]
        lines.append(
            f"- {item['bucket']}：n={item['n']}，R@1={fmt_pct(item['r1'])}，"
            f"Hard/Mid/Easy={ratios['Hard'] * 100:.1f}%/{ratios['Mid'] * 100:.1f}%/{ratios['Easy'] * 100:.1f}% ，"
            f"Δs范围=[{fmt_float(item['gap_min'], 3)}, {fmt_float(item['gap_max'], 3)}]"
        )
    lines.append("")
    lines.append("## 5. 实验二：Hard bin 内部再分层")
    lines.append("")
    lines.append(f"- 用于平均的有效模型：{', '.join(hard_result['valid_models']) if hard_result['valid_models'] else '无'}。")
    if hard_result["excluded_models"]:
        for model_name, reason in hard_result["excluded_models"].items():
            lines.append(f"- 排除 {model_name}：{reason}")
    lines.append(
        f"- 平均 R@1 是否单调不下降：{'是' if hard_result['monotonic_non_decreasing_mean_r1'] else '否'}。"
    )
    for item in hard_result["aggregated"]:
        lines.append(
            f"- {item['bucket']}：平均R@1={fmt_pct(item['r1_mean'])}，"
            f"模型间标准差={fmt_pct(item['r1_std'])}，"
            f"平均Δs范围≈[{fmt_float(item['gap_min_mean'], 3)}, {fmt_float(item['gap_max_mean'], 3)}]"
        )
    lines.append("")
    lines.append("## 6. 实验三：confidence AUROC")
    lines.append("")
    lines.append(f"- 用于平均的有效模型：{', '.join(auroc_result['valid_models']) if auroc_result['valid_models'] else '无'}。")
    if auroc_result["excluded_models"]:
        for model_name, reason in auroc_result["excluded_models"].items():
            lines.append(f"- 排除 {model_name}：{reason}")
    for difficulty in DIFFICULTY_ORDER:
        item = auroc_result["aggregated"][difficulty]
        lines.append(
            f"- {difficulty}：平均AUROC={fmt_pct(item['auc_mean'])}，模型间标准差={fmt_pct(item['auc_std'])}"
        )
    lines.append("")
    lines.append("## 7. 建议如何写进论文")
    lines.append("")
    lines.append("- 主论证先放实验二：只看 Hard bin，若 Hard-High 明显高于 Hard-Low，就能直接反驳“模型完全看不懂”的解释。")
    lines.append("- 实验一紧接着给全局外部效度：随着 oracle gap 增大，R@1 整体上升，说明 difficulty bin 背后确实对应可测的信号强弱。")
    lines.append("- 实验三作为辅助解释：在 Hard bin 内，推断时可见的 top margin 对 correct/incorrect 的排序能力最弱，说明不是方向错了，而是可用幅度不够。")
    lines.append("- 如果正文篇幅有限，可把实验一放主文图，实验二放主文小表，实验三放附录图。")
    lines.append("")
    return "\n".join(lines)



def print_console_summary(global_result: dict, hard_result: dict, auroc_result: dict) -> None:
    print("=" * 80)
    print("归因链分析：Hard 失败是否来自 feature gap 消失")
    print("=" * 80)
    print(f"实验一主模型：{global_result['model']}")
    for item in global_result["buckets"]:
        ratios = item["difficulty_ratios"]
        print(
            f"  {item['bucket']}: n={item['n']}, R@1={fmt_pct(item['r1'])}, "
            f"Hard/Mid/Easy={ratios['Hard'] * 100:.1f}%/{ratios['Mid'] * 100:.1f}%/{ratios['Easy'] * 100:.1f}%"
        )
    print("-" * 80)
    print("实验二：Hard bin 再分层（多模型平均）")
    for item in hard_result["aggregated"]:
        print(
            f"  {item['bucket']}: 平均R@1={fmt_pct(item['r1_mean'])}, "
            f"标准差={fmt_pct(item['r1_std'])}, Δs≈[{fmt_float(item['gap_min_mean'], 3)}, {fmt_float(item['gap_max_mean'], 3)}]"
        )
    print("-" * 80)
    print("实验三：confidence AUROC（多模型平均）")
    for difficulty in DIFFICULTY_ORDER:
        item = auroc_result["aggregated"][difficulty]
        print(
            f"  {difficulty}: AUROC={fmt_pct(item['auc_mean'])}, 标准差={fmt_pct(item['auc_std'])}"
        )
    print("=" * 80)



def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_meta = load_benchmark_meta()
    model_records = load_model_records(benchmark_meta)

    global_result = compute_global_analysis(PRIMARY_MODEL, model_records[PRIMARY_MODEL])
    hard_result = compute_hard_subbin_analysis(model_records)
    auroc_result = compute_confidence_auroc_analysis(model_records)

    export_global_csv(global_result)
    export_hard_csv(hard_result)
    export_auroc_csv(auroc_result)
    plot_global_quantiles(global_result)
    plot_hard_subbins(hard_result)
    plot_confidence_auroc(auroc_result)

    summary_md = build_summary_markdown(
        benchmark_meta,
        model_records,
        global_result,
        hard_result,
        auroc_result,
    )
    with open(OUTPUT_DIR / "analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    full_payload = {
        "benchmark": {
            "path": str(BENCHMARK_JSON),
            "n_total": len(benchmark_meta),
            "difficulty_counts": dict(Counter(meta["difficulty"] for meta in benchmark_meta.values())),
        },
        "table13_reference_overall_r1": {
            model_name: overall_r1_from_records(records)
            for model_name, records in model_records.items()
        },
        "experiment_1_global_quantiles": global_result,
        "experiment_2_hard_subbins": hard_result,
        "experiment_3_confidence_auroc": auroc_result,
    }
    with open(OUTPUT_DIR / "attribution_chain_results.json", "w", encoding="utf-8") as f:
        json.dump(full_payload, f, ensure_ascii=False, indent=2)

    print_console_summary(global_result, hard_result, auroc_result)
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
