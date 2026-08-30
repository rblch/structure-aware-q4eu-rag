from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from figures.style import (
    AQUA,
    BLUE,
    GRID,
    INK,
    ORANGE,
    RED,
    YELLOW,
    apply_style,
    save,
)

FAITHFULNESS_PATH = Path("data/evaluation/faithfulness_scores.json")
CORRECTNESS_PATH = Path("data/evaluation/correctness_scores.json")

CONDITIONS = [
    ("no_enrichment", "No enrichment", BLUE, "o"),
    ("parent_only", "Parent-only", ORANGE, "s"),
    ("xref_only", "Xref-only", AQUA, "^"),
    ("combined", "Combined", YELLOW, "D"),
    ("volume_matched", "Volume-matched", RED, "v"),
]


def read_scores(path: Path, key: str) -> dict[str, Counter]:
    records = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, Counter] = {c: Counter() for c, _, _, _ in CONDITIONS}
    for record in records:
        condition = record["condition_id"]
        if condition in counts:
            counts[condition][round(float(record[key]), 1)] += 1
    return counts


def condition_mean(scores: Counter) -> float:
    return sum(level * n for level, n in scores.items()) / sum(scores.values())


def draw_means(axis, faith: dict[str, Counter], corr: dict[str, Counter]) -> None:
    for row, (condition, label, _, _) in enumerate(CONDITIONS):
        f_mean = condition_mean(faith[condition])
        c_mean = condition_mean(corr[condition])
        axis.plot([c_mean, f_mean], [row, row], color=GRID, linewidth=1.6, zorder=1)
        axis.plot(c_mean, row, marker="s", markersize=6, color=ORANGE, zorder=3)
        axis.plot(f_mean, row, marker="o", markersize=6, color=BLUE, zorder=3)
        axis.text(
            c_mean, row - 0.34, f"{c_mean:.3f}", ha="center", fontsize=7, color=INK
        )
        axis.text(
            f_mean, row - 0.34, f"{f_mean:.3f}", ha="center", fontsize=7, color=INK
        )
        axis.text(
            -0.02,
            row,
            label,
            va="center",
            ha="right",
            fontsize=8,
            color=INK,
            transform=axis.get_yaxis_transform(),
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(len(CONDITIONS) - 0.4, -0.85)
    axis.set_yticks([])
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.grid(axis="x", linewidth=0.5)
    axis.set_xlabel("Mean judged score")
    axis.set_title("(a) Mean scores by condition", loc="left", pad=5)
    axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=BLUE,
                marker="o",
                linestyle="none",
                markersize=6,
                label="Faithfulness",
            ),
            Line2D(
                [],
                [],
                color=ORANGE,
                marker="s",
                linestyle="none",
                markersize=6,
                label="Correctness",
            ),
        ],
        loc="upper left",
        fontsize=7.5,
        frameon=False,
        ncol=1,
        handletextpad=0.3,
    )
    sns.despine(ax=axis, left=True)


def draw_distribution(axis, counts: dict[str, Counter], title: str) -> None:
    pooled: Counter = Counter()
    for scores in counts.values():
        pooled.update(scores)
    levels = sorted(pooled)
    total = sum(pooled.values())
    axis.bar(
        range(len(levels)),
        [pooled[level] for level in levels],
        color=BLUE,
        width=0.68,
    )
    for index, level in enumerate(levels):
        share = pooled[level] / total
        axis.text(
            index,
            pooled[level] + total * 0.018,
            f"{share:.0%}" if share >= 0.05 else "",
            ha="center",
            fontsize=7,
            color=INK,
        )
    axis.set_xticks(range(len(levels)))
    axis.set_xticklabels([f"{level:.1f}" for level in levels])
    axis.set_xlabel("Judged faithfulness score")
    axis.set_ylabel(f"Responses (of {total})")
    axis.set_ylim(0, max(pooled.values()) * 1.16)
    axis.set_title(title, loc="left", pad=5)
    sns.despine(ax=axis)


def write_rq2_answer_quality(
    *,
    output_path: Path,
    faithfulness_path: Path = FAITHFULNESS_PATH,
    correctness_path: Path = CORRECTNESS_PATH,
) -> dict[str, Any]:
    apply_style()
    faith = read_scores(faithfulness_path, "faithfulness_score")
    corr = read_scores(correctness_path, "correctness_score")

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), width_ratios=(1.45, 1))
    draw_means(axes[0], faith, corr)
    draw_distribution(axes[1], faith, "(b) Faithfulness scores, pooled")
    figure.subplots_adjust(left=0.135, right=0.985, top=0.89, bottom=0.21, wspace=0.32)
    return {
        "conditions": len(CONDITIONS),
        "written": [str(path) for path in save(figure, output_path)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the RQ2 answer-quality figure.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/figures/rq2_answer_quality_by_condition"),
    )
    parser.add_argument("--faithfulness-path", type=Path, default=FAITHFULNESS_PATH)
    parser.add_argument("--correctness-path", type=Path, default=CORRECTNESS_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_rq2_answer_quality(
        output_path=args.output_path,
        faithfulness_path=args.faithfulness_path,
        correctness_path=args.correctness_path,
    )
    print("RQ2 answer-quality figure complete: " + ", ".join(summary["written"]))


if __name__ == "__main__":
    main()
