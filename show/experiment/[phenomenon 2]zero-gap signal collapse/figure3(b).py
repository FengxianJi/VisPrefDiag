#!/usr/bin/env python3
"""
figure3(b).py - Plot average PairAcc curves across five difficulty regions.

The figure follows the averaging logic used by table18:
- Avg. Reward: average of HPSv2 and ImageReward
- Avg. Judge : average of four judge-method aggregates
- Avg. All   : average of reward and judge aggregates

Judge-method aggregates are computed across backbones with the same outlier
exclusions as table18:
- Single-agent (Pair): exclude Llama3.2V-11B
- Budgeted Single-agent: exclude Qwen3VL-30B

Outputs:
- show/figure/figure3(b).png
- show/figure/figure3(b).pdf
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parent
TABLE17_PATH = ROOT / "table17_source.tex"
OUTPUT_PNG = ROOT / "figure3(b).png"
OUTPUT_PDF = ROOT / "figure3(b).pdf"
OUTPUT_DPI = 600

REWARD_METHODS = ["HPSv2", "ImageReward"]
JUDGE_METHODS = [
    "Single-agent (LLM)",
    "Budgeted Single-agent",
    "Single-agent (Pair)",
    "Valuator Judge",
]
EXCLUDED_BACKBONES = {
    "Single-agent (Pair)": {"Llama3.2V-11B"},
    "Budgeted Single-agent": {"Qwen3VL-30B"},
}
SPLITS = ["Zero-gap", "Hard", "Mid", "Easy", "Ovr."]
SPLIT_KEYS = ["zero_gap", "hard", "mid", "easy", "overall"]
COLORS = {
    "Avg. Reward": "#C65D6B",
    "Avg. Judge": "#4C8DAE",
    "Avg. All": "#3C9D6E",
}
MARKERS = {
    "Avg. Reward": "o",
    "Avg. Judge": "s",
    "Avg. All": "D",
}


def configure_matplotlib() -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = OUTPUT_DPI
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def strip_latex(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textit\{([^}]*)\}", r"\1", text)
    return text.strip()


def parse_table17_pairacc():
    section = "baselines"
    baseline_rows: dict[str, dict[str, Decimal]] = {}
    model_rows: dict[str, dict[str, dict[str, Decimal]]] = {}

    with open(TABLE17_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if "\\multicolumn{16}{c}{\\textbf{" in line:
                match = re.search(r"\\textbf\{([^}]*)\}", line)
                if match:
                    section = match.group(1)
                    model_rows.setdefault(section, {})
                continue

            if not line or "&" not in line or not line.endswith(r"\\"):
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
                row = {
                    "overall": Decimal(parts[1]),
                    "hard": Decimal(parts[4]),
                    "zero_gap": Decimal(parts[7]),
                    "mid": Decimal(parts[10]),
                    "easy": Decimal(parts[13]),
                }
            except Exception:
                continue

            if section == "baselines":
                baseline_rows[label] = row
            else:
                model_rows.setdefault(section, {})[label] = row

    return baseline_rows, model_rows


def mean_decimal(values: list[Decimal]) -> Decimal:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values, Decimal("0")) / Decimal(len(values))


def aggregate_reward(baseline_rows: dict[str, dict[str, Decimal]]) -> dict[str, Decimal]:
    return {
        key: mean_decimal([baseline_rows[method][key] for method in REWARD_METHODS])
        for key in SPLIT_KEYS
    }


def aggregate_judge_method(
    method_name: str, model_rows: dict[str, dict[str, dict[str, Decimal]]]
) -> dict[str, Decimal]:
    excluded = EXCLUDED_BACKBONES.get(method_name, set())
    rows = []
    for backbone, methods in model_rows.items():
        if backbone in excluded:
            continue
        row = methods.get(method_name)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError(f"missing rows for {method_name}")

    return {
        key: mean_decimal([row[key] for row in rows])
        for key in SPLIT_KEYS
    }


def build_series():
    baseline_rows, model_rows = parse_table17_pairacc()

    reward_rows = {
        method_name: baseline_rows[method_name]
        for method_name in REWARD_METHODS
    }
    reward_avg = {
        key: mean_decimal([row[key] for row in reward_rows.values()])
        for key in SPLIT_KEYS
    }
    judge_method_avgs = {
        method_name: aggregate_judge_method(method_name, model_rows)
        for method_name in JUDGE_METHODS
    }
    judge_avg = {
        key: mean_decimal([metrics[key] for metrics in judge_method_avgs.values()])
        for key in SPLIT_KEYS
    }
    all_avg = {
        key: mean_decimal(
            [row[key] for row in reward_rows.values()]
            + [metrics[key] for metrics in judge_method_avgs.values()]
        )
        for key in SPLIT_KEYS
    }

    return {
        "Avg. Reward": reward_avg,
        "Avg. Judge": judge_avg,
        "Avg. All": all_avg,
    }


def draw_figure(series: dict[str, dict[str, Decimal]]) -> None:
    configure_matplotlib()

    label_fontsize = 26
    tick_fontsize = 24
    value_fontsize = 21
    legend_fontsize = 19

    fig, ax = plt.subplots(figsize=(10.5, 7.6), facecolor="white")
    ax.set_facecolor("white")

    x = [0.0, 1.15, 2.30, 3.45, 4.60]

    for name in ["Avg. Reward", "Avg. Judge", "Avg. All"]:
        values = [float(series[name][key]) for key in SPLIT_KEYS]
        ax.plot(
            x,
            values,
            color=COLORS[name],
            marker=MARKERS[name],
            linewidth=3.2,
            markersize=10.0,
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=name,
            zorder=3,
        )
        for xi, yi in zip(x, values):
            ax.text(
                xi,
                yi + 0.8,
                f"{yi:.2f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                color=COLORS[name],
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.78},
                zorder=6,
            )

    ax.set_xticks(x, SPLITS, fontsize=tick_fontsize)
    ax.set_ylabel("PairAcc", fontsize=label_fontsize, color="#000000", labelpad=4)
    ax.set_xlabel("Difficulty Region", fontsize=label_fontsize, color="#000000", labelpad=8)
    ax.set_ylim(35.0, 96.0)
    ax.set_xlim(x[0] - 0.45, x[-1] + 0.45)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
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

    legend = ax.legend(
        loc="upper left",
        fontsize=legend_fontsize,
        frameon=True,
        framealpha=0.85,
        edgecolor="0.7",
        handlelength=1.5,
        borderpad=0.4,
        handletextpad=0.6,
    )
    legend.get_frame().set_facecolor("white")

    plt.tight_layout(pad=1.1, rect=(0.05, 0.01, 0.99, 0.97))
    fig.savefig(OUTPUT_PNG, dpi=OUTPUT_DPI, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    series = build_series()
    draw_figure(series)

    print("Figure 3(b) series:")
    for name in ["Avg. Reward", "Avg. Judge", "Avg. All"]:
        values = ", ".join(
            f"{split}={float(series[name][key]):.2f}"
            for split, key in zip(SPLITS, SPLIT_KEYS)
        )
        print(f"  {name}: {values}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
