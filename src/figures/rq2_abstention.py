from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import seaborn as sns

from figures.style import BLUE, INK, ORANGE, apply_style, save

EVALUATION_PATH = Path("data/evaluation/generated_answer_evaluation.json")

CONDITIONS = [
    ("no_enrichment", "No enrichment"),
    ("parent_only", "Parent-only"),
    ("xref_only", "Xref-only"),
    ("combined", "Combined"),
    ("volume_matched", "Volume-matched"),
]

SERIES = [
    ("answer_abstains", "Abstention", BLUE),
    ("either_judge_unjustified_abstention", "Unjustified abstention", ORANGE),
]


def read_rates(path: Path) -> dict[str, dict[str, tuple[float, int]]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    rates: dict[str, dict[str, tuple[float, int]]] = {}
    for condition, _ in CONDITIONS:
        rows = [r for r in records if r["condition_id"] == condition]
        rates[condition] = {
            field: (
                100 * sum(bool(r[field]) for r in rows) / len(rows),
                sum(bool(r[field]) for r in rows),
            )
            for field, _, _ in SERIES
        }
    return rates


def write_rq2_abstention(
    *, output_path: Path, evaluation_path: Path = EVALUATION_PATH
) -> dict[str, Any]:
    apply_style()
    rates = read_rates(evaluation_path)

    figure, axis = plt.subplots(figsize=(7.0, 2.6))
    height = 0.34
    for offset, (field, label, colour) in zip((-0.5, 0.5), SERIES):
        ys = [row + offset * height for row in range(len(CONDITIONS))]
        values = [rates[c][field] for c, _ in CONDITIONS]
        axis.barh(ys, [v for v, _ in values], height=height, color=colour, label=label)
        for y, (percent, count) in zip(ys, values):
            axis.text(
                percent + 0.22,
                y,
                f"{percent:.1f}\u2009% ({count})",
                va="center",
                fontsize=7,
                color=INK,
            )

    axis.set_yticks(range(len(CONDITIONS)))
    axis.set_yticklabels([label for _, label in CONDITIONS], fontsize=8)
    axis.set_ylim(len(CONDITIONS) - 0.5, -0.5)
    axis.set_xlim(0, 14)
    axis.set_xlabel("% share of the questions (counts in brackets)")
    axis.grid(axis="x", linewidth=0.5)
    axis.legend(loc="lower right", fontsize=7.5, frameon=False)
    sns.despine(ax=axis, left=True)
    figure.subplots_adjust(left=0.17, right=0.99, top=0.96, bottom=0.20)
    return {
        "rates": rates,
        "written": [str(path) for path in save(figure, output_path)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the RQ2 abstention figure.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/figures/rq2_abstention_by_condition"),
    )
    parser.add_argument("--evaluation-path", type=Path, default=EVALUATION_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_rq2_abstention(
        output_path=args.output_path, evaluation_path=args.evaluation_path
    )
    print("RQ2 abstention figure complete: " + ", ".join(summary["written"]))


if __name__ == "__main__":
    main()
