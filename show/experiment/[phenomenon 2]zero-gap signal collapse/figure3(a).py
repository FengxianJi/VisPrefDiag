#!/usr/bin/env python3
"""
figure3(a).py - Mechanism-effectiveness heatmap across difficulty regimes.

This figure reads AUROC values from table17.tex and renders a grouped heatmap
for representative methods:

1. Visual Reward Models
   - HPSv2
   - ImageReward

2. Model Scaling (Valuator)
   - Qwen3VL-4B / Valuator
   - Qwen3VL-30B / Valuator

3. Elicitation Strategy (Gemini-3-Flash)
   - Single-agent (Pair)
   - Valuator Judge
   - Single-agent (LLM)
   - Budgeted Single-agent

Outputs:
- show/figure/figure3(a).png
- show/figure/figure3(a).pdf
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parent
TABLE17_PATH = ROOT / "table17_source.tex"
OUTPUT_PNG = ROOT / "figure3(a).png"
OUTPUT_PDF = ROOT / "figure3(a).pdf"
OUTPUT_DPI = 600
TEXT_SCALE = 0.74

COLS = [
    ("Easy", "easy_auc"),
    ("Mid", "mid_auc"),
    ("Hard", "hard_auc"),
    ("Zero-gap", "zero_auc"),
]

ROW_SPECS = [
    ("Visual Reward Models", "HPSv2", "baselines", "HPSv2"),
    ("Visual Reward Models", "ImageReward", "baselines", "ImageReward"),
    ("Model Scaling (Valuator)", "Valuator (4B)", "Qwen3VL-4B", "Valuator"),
    ("Model Scaling (Valuator)", "Valuator (30B)", "Qwen3VL-30B", "Valuator"),
    ("Elicitation Strategy (Gemini-3-Flash)", "SA (Pair)", "Gemini-3-Flash", "Single-agent (Pair)"),
    ("Elicitation Strategy (Gemini-3-Flash)", "V. Judge", "Gemini-3-Flash", "Valuator Judge"),
    ("Elicitation Strategy (Gemini-3-Flash)", "SA (LLM)", "Gemini-3-Flash", "Single-agent (LLM)"),
    ("Elicitation Strategy (Gemini-3-Flash)", "Budgeted SA", "Gemini-3-Flash", "Budgeted Single-agent"),
]
FAMILY_DISPLAY = {
    "Visual Reward Models": "Visual\nReward\nModels",
    "Model Scaling (Valuator)": "Model\nScaling\nValuator",
    "Elicitation Strategy (Gemini-3-Flash)": "Elicitation\nGemini-3\nFlash",
}


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = OUTPUT_DPI
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def fs(size: float) -> float:
    return size * TEXT_SCALE


def strip_latex(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    return text.strip()


def parse_table17_auroc():
    section = "baselines"
    rows: dict[tuple[str, str], dict[str, float]] = {}

    for raw_line in TABLE17_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if "\\multicolumn{16}{c}{\\textbf{" in line:
            match = re.search(r"\\textbf\{([^}]*)\}", line)
            if match:
                section = match.group(1)
            continue

        if not line.endswith(r"\\") or "&" not in line:
            continue
        if line.startswith(r"\textit{Random baseline}"):
            continue
        if line.startswith(r"\multirow") or line.startswith("& "):
            continue
        if line.startswith(r"\midrule") or line.startswith(r"\bottomrule"):
            continue

        parts = [strip_latex(part) for part in line[:-2].split("&")]
        if len(parts) < 16:
            continue

        label = parts[0]
        if label.startswith(r"\multicolumn"):
            continue

        try:
            rows[(section, label)] = {
                "overall_auc": float(parts[3]),
                "hard_auc": float(parts[6]),
                "zero_auc": float(parts[9]),
                "mid_auc": float(parts[12]),
                "easy_auc": float(parts[15]),
            }
        except ValueError:
            continue

    return rows


def cell_color(delta: float) -> tuple[float, float, float]:
    weak = mcolors.to_rgb("#F6D8D5")
    neutral = mcolors.to_rgb("#F7F7F5")
    moderate = mcolors.to_rgb("#C7E3CC")
    strong = mcolors.to_rgb("#4F9967")

    if delta < 15:
        t = max(delta, 0.0) / 15.0
        return tuple((1 - t) * weak[i] + t * neutral[i] for i in range(3))
    if delta <= 30:
        t = (delta - 15.0) / 15.0
        return tuple((1 - t) * neutral[i] + t * moderate[i] for i in range(3))
    t = min((delta - 30.0) / 20.0, 1.0)
    return tuple((1 - t) * moderate[i] + t * strong[i] for i in range(3))


def build_layout():
    positions = []
    row_step = 0.42
    y = 0.0
    family_rows: dict[str, list[float]] = {}

    for family, display_name, section, method in ROW_SPECS:
        positions.append((family, display_name, section, method, y))
        family_rows.setdefault(family, []).append(y)
        y += row_step

    family_centers = {
        family: sum(vals) / len(vals)
        for family, vals in family_rows.items()
    }
    return positions, family_centers, y - row_step


def draw_heatmap(rows: dict[tuple[str, str], dict[str, float]]) -> None:
    configure_matplotlib()

    fig = plt.figure(figsize=(8.4, 8.4), facecolor="white")
    ax = fig.add_axes([0.32, 0.20, 0.48, 0.56], facecolor="white")

    positions, family_centers, max_y = build_layout()
    x_positions = list(range(len(COLS)))
    cell_scale = 1.33
    cell_w = 0.94 * cell_scale
    cell_h = 0.38 * cell_scale
    matrix_left = -0.47
    ax.set_xlim(-1.35, len(COLS) + 1.10)
    ax.set_ylim(max_y + 0.42, -0.72)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Column headers
    for x, (col_name, _) in enumerate(COLS):
        ax.text(
            x,
            -0.62,
            col_name,
            ha="center",
            va="center",
            fontsize=fs(13),
            fontweight="bold",
            color="#1F1F1F",
        )

    # Draw cells and method labels
    for family, display_name, section, method, y in positions:
        ax.text(
            -0.78,
            y,
            display_name,
            ha="right",
            va="center",
            fontsize=fs(10.2),
            color="#1F1F1F",
            rotation=-45,
            clip_on=False,
        )

        values = rows[(section, method)]
        for x, (_, key) in zip(x_positions, COLS):
            auroc = values[key]
            delta = auroc - 50.0
            rect = Rectangle(
                (x - cell_w / 2, y - cell_h / 2),
                cell_w,
                cell_h,
                facecolor=cell_color(delta),
                edgecolor="#D7D7D1",
                linewidth=1.0,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y - 0.055,
                f"{auroc:.2f}",
                ha="center",
                va="center",
                fontsize=fs(11.8),
                fontweight="bold",
                color="#1B1B1B",
            )
            ax.text(
                x,
                y + 0.10,
                f"+{delta:.2f}",
                ha="center",
                va="center",
                fontsize=fs(8.8),
                color="#374151",
            )

    # Family labels on the right
    for family, center_y in family_centers.items():
        ax.text(
            len(COLS) - 0.24,
            center_y,
            FAMILY_DISPLAY.get(family, family),
            ha="left",
            va="center",
            fontsize=fs(10.2),
            fontweight="bold",
            color="#2A2A2A",
            clip_on=False,
        )

    fig.suptitle(
        "Mechanism effectiveness across difficulty regimes (AUROC)",
        fontsize=fs(15),
        fontweight="bold",
        y=0.94,
    )

    legend_handles = [
        Patch(facecolor=cell_color(38.0), edgecolor="#D7D7D1", label="Strong (Δ > 30)"),
        Patch(facecolor=cell_color(22.0), edgecolor="#D7D7D1", label="Moderate (15 ≤ Δ ≤ 30)"),
        Patch(facecolor=cell_color(8.0), edgecolor="#D7D7D1", label="Weak (Δ < 15)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.57, 0.885),
        ncol=3,
        frameon=False,
        fontsize=fs(11),
        handlelength=1.4,
        columnspacing=1.4,
    )

    fig.text(
        0.5,
        0.08,
        "All methods degrade monotonically toward Zero-gap, confirming feature-gap collapse as the shared bottleneck.",
        ha="center",
        va="center",
        fontsize=fs(10.8),
        color="#3A3A3A",
    )

    fig.savefig(OUTPUT_PNG, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT_PDF, dpi=OUTPUT_DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    rows = parse_table17_auroc()
    draw_heatmap(rows)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
