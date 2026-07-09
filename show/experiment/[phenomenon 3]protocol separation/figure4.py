#!/usr/bin/env python3
"""
figure4.py - Compact bar-chart comparison of four hardness protocols.

To keep only the strongest evidence, this figure shows a single AUROC panel:
- four protocol groups
- Hard / Mid / Easy bars
- one aligned Overall dashed line with an inline label

This is the most direct visual support for the paper claim that Pair-gap
induces the clearest behavioral difficulty gradient.

Output:
- show/figure/figure4.pdf
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parent
TABLE16_PATH = ROOT / "table16_source.tex"
OUTPUT_PDF = ROOT / "figure4.pdf"

PROTOCOLS = [
    ("Pair-gap", "Pair-gap\n(ours)"),
    ("Human-only", "Human-only"),
    ("Model-only", "Model-only"),
    ("Dual-filter", "Dual-filter"),
]
BAR_ORDER = ["Hard", "Mid", "Easy"]
BAR_COLORS = {
    "Hard": "#D96C75",
    "Mid": "#D9A441",
    "Easy": "#4C8DAE",
}
VALUE_LABEL_OFFSETS = {
    "Hard": 0.25,
    "Mid": 1.35,
    "Easy": 0.75,
}
Y_MIN = 52.0
Y_MAX = 70.8


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def strip_latex(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    return text.strip()


def parse_table16() -> dict[str, dict[str, dict[str, tuple[float | None, float | None]]]]:
    rows = {}
    for raw_line in TABLE16_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.endswith(r"\\") or "&" not in line:
            continue
        if any(
            line.startswith(prefix)
            for prefix in [r"\multicolumn", r"\midrule", r"\toprule", r"\bottomrule", r"\multirow", "& "]
        ):
            continue

        parts = [strip_latex(part) for part in line[:-2].split("&")]
        if len(parts) != 17:
            continue

        method = parts[0]
        values = []
        for cell in parts[1:]:
            if cell == "--":
                values.append((None, None))
                continue
            left, right = [token.strip() for token in cell.split("/")]
            values.append((float(left), float(right)))

        rows[method] = {
            "Pair-gap": {
                "Overall": values[0],
                "Hard": values[1],
                "Mid": values[2],
                "Easy": values[3],
            },
            "Human-only": {
                "Overall": values[4],
                "Hard": values[5],
                "Mid": values[6],
                "Easy": values[7],
            },
            "Model-only": {
                "Overall": values[8],
                "Hard": values[9],
                "Mid": values[10],
                "Easy": values[11],
            },
            "Dual-filter": {
                "Overall": values[12],
                "Hard": values[13],
                "Mid": values[14],
                "Easy": values[15],
            },
        }
    return rows


def build_summary(rows: dict[str, dict[str, dict[str, tuple[float | None, float | None]]]]):
    summary = {}
    metric_idx = 1  # AUROC only
    for protocol_key, _ in PROTOCOLS:
        hard_vals = []
        mid_vals = []
        easy_vals = []
        overall_vals = []
        mono_count = 0
        valid_count = 0

        for protocol_metrics in rows.values():
            row = protocol_metrics[protocol_key]
            hard = row["Hard"][metric_idx]
            mid = row["Mid"][metric_idx]
            easy = row["Easy"][metric_idx]
            overall = row["Overall"][metric_idx]
            if None in (hard, mid, easy, overall):
                continue

            hard_vals.append(hard)
            mid_vals.append(mid)
            easy_vals.append(easy)
            overall_vals.append(overall)
            valid_count += 1
            if easy > mid > hard:
                mono_count += 1

        summary[protocol_key] = {
            "Hard": sum(hard_vals) / len(hard_vals),
            "Mid": sum(mid_vals) / len(mid_vals),
            "Easy": sum(easy_vals) / len(easy_vals),
            "Overall": sum(overall_vals) / len(overall_vals),
            "mono_rate": 100.0 * mono_count / valid_count,
            "mono_count": mono_count,
            "mono_total": valid_count,
        }
    return summary


def draw_panel(ax, stats: dict[str, dict]) -> None:
    label_fontsize = 26
    tick_fontsize = 24
    overall_tick_fontsize = 18
    overall_text_fontsize = 16
    value_fontsize = 21

    group_spacing = 1.75
    x_centers = [idx * group_spacing for idx in range(len(PROTOCOLS))]
    bar_width = 0.54
    offset_step = 0.52
    offsets = [-offset_step, 0.0, offset_step]

    overall = stats[PROTOCOLS[0][0]]["Overall"]
    ax.axhline(overall, color="#5A5A5A", linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.text(
        x_centers[0] + offsets[0] - 0.12,
        overall + 0.34,
        "Overall",
        ha="center",
        va="bottom",
        fontsize=overall_text_fontsize,
        color="#000000",
        zorder=5,
    )

    for center, (protocol_key, _protocol_label) in zip(x_centers, PROTOCOLS):
        values = stats[protocol_key]
        for offset, split_name in zip(offsets, BAR_ORDER):
            score = values[split_name]
            x = center + offset
            ax.bar(
                x,
                score - Y_MIN,
                width=bar_width * 0.92,
                bottom=Y_MIN,
                color=BAR_COLORS[split_name],
                edgecolor="#2F2F2F",
                linewidth=0.9,
                zorder=3,
            )
            ax.text(
                x,
                score + VALUE_LABEL_OFFSETS[split_name],
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                color="#000000",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.78},
                zorder=6,
            )

    ax.set_xticks(x_centers, [label for _, label in PROTOCOLS], fontsize=tick_fontsize)
    ax.set_xlim(x_centers[0] - 1.05, x_centers[-1] + 1.05)
    ax.tick_params(axis="both", labelsize=tick_fontsize, colors="#000000", width=0.8, length=4)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5, zorder=0)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.set_facecolor("white")
    ax.set_ylim(Y_MIN, Y_MAX)
    y_ticks = [55, 60, overall, 65, 70]
    y_tick_labels = ["55", "60", f"{overall:.2f}", "65", "70"]
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(y_tick_labels))
    for tick_label in ax.get_yticklabels():
        if tick_label.get_text() == f"{overall:.2f}":
            tick_label.set_fontsize(overall_tick_fontsize)
    ax.set_ylabel("AUROC", fontsize=label_fontsize, color="#000000", labelpad=4)
    ax.set_box_aspect(2 / 3)


def draw_figure(summary) -> None:
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(10.5, 7.6), facecolor="white")

    draw_panel(ax, summary)

    plt.tight_layout(pad=1.1, rect=(0.05, 0.01, 0.99, 0.97))
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = parse_table16()
    summary = build_summary(rows)
    draw_figure(summary)
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
