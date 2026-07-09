#!/usr/bin/env python3
"""
table18.py - Build a compact PairAcc-only summary table from table17.tex.

Rules:
- Keep reward-model rows: HPSv2, ImageReward
- Keep judge-method rows averaged across backbones:
  Single-agent (LLM), Budgeted Single-agent, Single-agent (Pair), Valuator Judge
- Exclude the Llama3.2V-11B outlier for Single-agent (Pair)
- Exclude the Qwen3VL-30B outlier for Budgeted Single-agent
- Non-hard Avg = simple average of Mid and Easy PairAcc
"""

from __future__ import annotations

import re
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


TABLE17_PATH = Path(__file__).resolve().parent / "table17_source.tex"
OUTPUT_PATH = Path(__file__).resolve().parent / "table18.tex"
PDF_PATH = Path(__file__).resolve().parent / "table18.pdf"
WRAPPER_PATH = Path(__file__).resolve().parent / "table18_wrapper.tex"
WRAPPER_PDF_PATH = Path(__file__).resolve().parent / "table18_wrapper.pdf"
TECTONIC_BIN = "/opt/homebrew/bin/tectonic"

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
DISPLAY_LABELS = {
    "Single-agent (LLM)": "SA (LLM)",
    "Budgeted Single-agent": "Budgeted SA",
    "Single-agent (Pair)": "SA (Pair)",
    "Valuator Judge": "V. Judge",
}


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
            if not line or "&" not in line:
                if "\\multicolumn{16}{c}{\\textbf{" in line:
                    match = re.search(r"\\textbf\{([^}]*)\}", line)
                    if match:
                        section = match.group(1)
                        model_rows.setdefault(section, {})
                continue

            if line.startswith(r"\textit{Random baseline}"):
                continue
            if line.startswith(r"\multirow") or line.startswith("& "):
                continue
            if line.startswith(r"\midrule") or line.startswith(r"\bottomrule"):
                continue

            parts = [strip_latex(part) for part in line.split("&")]
            if len(parts) < 14:
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
                    "easy": Decimal(parts[13].replace("\\\\", "").strip()),
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


def build_metrics(overall: Decimal, hard: Decimal, zero_gap: Decimal, mid: Decimal, easy: Decimal):
    return {
        "overall": overall,
        "hard": hard,
        "zero_gap": zero_gap,
        "non_hard": (mid + easy) / Decimal("2"),
    }


def aggregate_judge_method(method_name: str, model_rows: dict[str, dict[str, dict[str, Decimal]]]):
    rows = []
    excluded = EXCLUDED_BACKBONES.get(method_name, set())
    for backbone, methods in model_rows.items():
        if backbone in excluded:
            continue
        row = methods.get(method_name)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError(f"missing rows for {method_name}")

    return build_metrics(
        mean_decimal([row["overall"] for row in rows]),
        mean_decimal([row["hard"] for row in rows]),
        mean_decimal([row["zero_gap"] for row in rows]),
        mean_decimal([row["mid"] for row in rows]),
        mean_decimal([row["easy"] for row in rows]),
    )


def fmt(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def render_table(rows: list[tuple[str, dict[str, Decimal]]]) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{2.7pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.16}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Ovr.} & \textbf{Hard} & \textbf{Zero} & \textbf{NH} \\")
    lines.append(r"\midrule")

    for idx, (label, metrics) in enumerate(rows):
        if idx == 2 or idx == 6:
            lines.append(r"\midrule")
        lines.append(
            f"{display_label(label):<24} & "
            f"{fmt(metrics['overall'])} & "
            f"{fmt(metrics['hard'])} & "
            f"{fmt(metrics['zero_gap'])} & "
            f"{fmt(metrics['non_hard'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{\textbf{Representative methods under Zero-gap and other difficulty regions on ImageReward (test).} "
        r"We report \textbf{PairAcc} only. \textbf{Zero-gap} denotes pairs with "
        r"$\Delta_{\mathrm{overall}}=0$, \textbf{Hard} denotes pairs with "
        r"$\Delta_{\mathrm{overall}}=1$, and \textbf{Non-hard Avg} averages the Mid and Easy "
        r"regions ($\Delta_{\mathrm{overall}}\geq2$). For LLM-as-a-judge methods, rows are "
        r"averaged across backbones; for \textit{Single-agent (Pair)}, the "
        r"Llama3.2V-11B outlier is excluded, and for \textit{Budgeted Single-agent}, the "
        r"Qwen3VL-30B outlier is excluded.}"
    )
    lines.append(r"\label{tab:zero_gap_main}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_rows():
    baseline_rows, model_rows = parse_table17_pairacc()

    selected_rows: list[tuple[str, dict[str, Decimal]]] = []

    reward_metrics = {}
    for method_name in REWARD_METHODS:
        row = baseline_rows[method_name]
        reward_metrics[method_name] = build_metrics(
            row["overall"], row["hard"], row["zero_gap"], row["mid"], row["easy"]
        )
        selected_rows.append((method_name, reward_metrics[method_name]))

    judge_metrics = {}
    for method_name in JUDGE_METHODS:
        judge_metrics[method_name] = aggregate_judge_method(method_name, model_rows)
        selected_rows.append((method_name, judge_metrics[method_name]))

    avg_reward = {
        key: mean_decimal([metrics[key] for metrics in reward_metrics.values()])
        for key in ("overall", "hard", "zero_gap", "non_hard")
    }
    avg_judge = {
        key: mean_decimal([metrics[key] for metrics in judge_metrics.values()])
        for key in ("overall", "hard", "zero_gap", "non_hard")
    }
    avg_all = {
        key: mean_decimal(
            [metrics[key] for metrics in list(reward_metrics.values()) + list(judge_metrics.values())]
        )
        for key in ("overall", "hard", "zero_gap", "non_hard")
    }

    selected_rows.extend(
        [
            ("Avg. Reward", avg_reward),
            ("Avg. Judge", avg_judge),
            ("Avg. All", avg_all),
        ]
    )

    return selected_rows


def compile_pdf():
    wrapper = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage[margin=0.75in]{geometry}",
            r"\usepackage{booktabs}",
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"\input{table18.tex}",
            r"\end{document}",
            "",
        ]
    )
    with open(WRAPPER_PATH, "w", encoding="utf-8") as f:
        f.write(wrapper)

    try:
        subprocess.run(
            [TECTONIC_BIN, WRAPPER_PATH.name, "--outdir", "."],
            cwd=OUTPUT_PATH.parent,
            check=True,
        )
        if WRAPPER_PDF_PATH.exists():
            WRAPPER_PDF_PATH.replace(PDF_PATH)
    finally:
        for suffix in (".tex", ".aux", ".log", ".xdv", ".synctex.gz"):
            path = OUTPUT_PATH.parent / f"table18_wrapper{suffix}"
            if path.exists():
                path.unlink()


def main():
    selected_rows = build_rows()
    latex = render_table(selected_rows)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(latex)

    compile_pdf()

    print("Table 18 rows:")
    for label, metrics in selected_rows:
        print(
            f"  {label:<20} "
            f"Overall={fmt(metrics['overall'])} "
            f"Hard={fmt(metrics['hard'])} "
            f"Zero-gap={fmt(metrics['zero_gap'])} "
            f"Non-hard Avg={fmt(metrics['non_hard'])}"
        )
    print(f"\nWritten to {OUTPUT_PATH}")
    print(f"Written to {PDF_PATH}")
    print("\n" + latex)

if __name__ == "__main__":
    main()
