#!/usr/bin/env python3
"""
Build Figure 1 bar charts as PNGs.

Outputs:
- show/figure/figure1_1.png
- show/figure/figure1_2.png

Each figure contains only the traditional reward model average bars
for one split, with metric-wise normalized heights and raw values annotated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TABLE13_PATH = ROOT / "table13_source.tex"

OUTPUT_HARD = ROOT / "hard_region_reward_average.png"
OUTPUT_EASY = ROOT / "easy_region_reward_average.png"

TRADITIONAL_METHODS = (
    "CLIPScore",
    "NIMA",
    "HPSv2",
    "PickScore",
    "ImageReward",
    "Q-Align (IQA)",
    "Q-Align (IAA)",
    "VQAscore",
)

METRICS = ("PairAcc", "NormGap", "AUROC")
COLORS = {
    "PairAcc": "#D96C75",
    "NormGap": "#D9A441",
    "AUROC": "#4C8DAE",
}


@dataclass(frozen=True)
class PanelSpec:
    split_name: str
    output_path: Path


PANELS = (
    PanelSpec(
        split_name="Hard",
        output_path=OUTPUT_HARD,
    ),
    PanelSpec(
        split_name="Easy",
        output_path=OUTPUT_EASY,
    ),
)


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def strip_latex(token: str) -> str:
    token = token.strip()
    token = re.sub(r"\\textbf\{([^}]*)\}", r"\1", token)
    token = re.sub(r"\\textit\{([^}]*)\}", r"\1", token)
    token = token.replace("\\", "").strip()
    return token


def parse_table13(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    rows: dict[str, dict[str, dict[str, float]]] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if "&" not in line or not line.endswith(r"\\"):
            continue
        if line.startswith(r"\textit{Random baseline}"):
            continue
        if line.startswith(r"\multicolumn"):
            continue

        parts = [strip_latex(part) for part in line[:-2].split("&")]
        method = parts[0]
        if method not in TRADITIONAL_METHODS:
            continue

        values = [float(part) for part in parts[1:13]]
        rows[method] = {
            "hard": {
                "PairAcc": values[3],
                "NormGap": values[4],
                "AUROC": values[5],
            },
            "easy": {
                "PairAcc": values[9],
                "NormGap": values[10],
                "AUROC": values[11],
            },
        }

    missing = [name for name in TRADITIONAL_METHODS if name not in rows]
    if missing:
        raise ValueError(f"Missing traditional methods in table13.tex: {missing}")

    return rows


def compute_summary(rows: dict[str, dict[str, dict[str, float]]]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    averages: dict[str, dict[str, float]] = {"hard": {}, "easy": {}}
    metric_max: dict[str, float] = {}

    for split in ("hard", "easy"):
        for metric in METRICS:
            values = [rows[method][split][metric] for method in TRADITIONAL_METHODS]
            averages[split][metric] = sum(values) / len(values)

    for metric in METRICS:
        metric_max[metric] = max(
            rows[method][split][metric]
            for method in TRADITIONAL_METHODS
            for split in ("hard", "easy")
        )

    return averages, metric_max


def format_metric(metric: str, value: float) -> str:
    if metric == "NormGap":
        return f"{value:.2f}"
    return f"{value:.1f}"


def draw_panel(
    spec: PanelSpec,
    averages: dict[str, dict[str, float]],
    metric_max: dict[str, float],
) -> None:
    split_key = spec.split_name.lower()
    raw_values = [averages[split_key][metric] for metric in METRICS]
    norm_values = [raw / metric_max[metric] for raw, metric in zip(raw_values, METRICS)]

    fig, chart_ax = plt.subplots(figsize=(5.4, 3.15), facecolor="white")

    bars = chart_ax.bar(
        range(len(METRICS)),
        norm_values,
        width=0.62,
        color=[COLORS[metric] for metric in METRICS],
        edgecolor="#2F2F2F",
        linewidth=1.0,
    )

    chart_ax.set_ylim(0, 1.08)
    chart_ax.set_xticks(range(len(METRICS)), METRICS, fontsize=12)
    chart_ax.set_yticks([])
    chart_ax.set_ylabel("Normalized height", fontsize=12)
    chart_ax.set_facecolor("#FBFBF8")
    chart_ax.grid(axis="y", color="#E7E7E0", linewidth=0.8)
    chart_ax.set_axisbelow(True)
    chart_ax.spines["top"].set_visible(False)
    chart_ax.spines["right"].set_visible(False)
    chart_ax.spines["left"].set_color("#BFBFB8")
    chart_ax.spines["bottom"].set_color("#BFBFB8")
    for bar, metric, raw_value, norm_value in zip(bars, METRICS, raw_values, norm_values):
        x = bar.get_x() + bar.get_width() / 2
        y = min(norm_value + 0.035, 1.03)
        chart_ax.text(
            x,
            y,
            format_metric(metric, raw_value),
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#1F1F1F",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(spec.output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    configure_matplotlib()

    rows = parse_table13(TABLE13_PATH)
    averages, metric_max = compute_summary(rows)

    for spec in PANELS:
        draw_panel(spec, averages, metric_max)
        print(f"Saved {spec.output_path}")

    print("Averages used:")
    for split in ("hard", "easy"):
        metrics = ", ".join(f"{metric}={format_metric(metric, averages[split][metric])}" for metric in METRICS)
        print(f"  {split}: {metrics}")


if __name__ == "__main__":
    main()
