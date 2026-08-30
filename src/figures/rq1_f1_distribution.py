from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figures.style import (
    AXIS,
    CONFIRMATORY,
    FAMILY_COLOURS,
    FAMILY_LABELS,
    INK,
    apply_style,
    save,
)

QUERY_METRICS_PATH = Path("data/results/tables/retrieval_query_metrics.csv")
EXCLUDED = {"hier_paragraph_contextualized"}
ORDER = [
    ("fixed_size", ["fs_64_12", "fs_128_25", "fs_256_50"]),
    (
        "semantic",
        [
            "sem_50_64",
            "sem_70_64",
            "sem_50_128",
            "sem_70_128",
            "sem_50_256",
            "sem_70_256",
        ],
    ),
    (
        "hierarchical",
        [
            "hier_subparagraph",
            "hier_paragraph",
            "hier_paragraph_contextualized",
            "hier_article",
        ],
    ),
]


def read_query_f1(
    query_metrics_path: Path,
    *,
    search_scope: str,
    top_k: int,
) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    with query_metrics_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["search_scope"] != search_scope or int(row["top_k"]) != top_k:
                continue
            scores[row["config_id"]].append(float(row["f1"]))
    return scores


def write_rq1_f1_distribution(
    *,
    output_path: Path,
    query_metrics_path: Path = QUERY_METRICS_PATH,
    search_scope: str = "all_acts",
    top_k: int = 10,
) -> dict[str, Any]:
    apply_style()
    scores = read_query_f1(query_metrics_path, search_scope=search_scope, top_k=top_k)
    configs = [
        c for _, members in ORDER for c in members if c in scores and c not in EXCLUDED
    ]
    families = {c: family for family, members in ORDER for c in members}
    values = [scores[c] for c in configs]
    colours = [FAMILY_COLOURS[families[c]] for c in configs]

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    sns.boxplot(
        data=values,
        orient="h",
        ax=axis,
        palette=colours,
        width=0.62,
        linewidth=0.8,
        fliersize=1.8,
        saturation=1.0,
    )
    axis.set_yticks(range(len(configs)))
    axis.set_yticklabels(configs)
    for tick, config in zip(axis.get_yticklabels(), configs):
        tick.set_fontfamily("monospace")
        tick.set_fontsize(7.5)
        if config in CONFIRMATORY:
            tick.set_color(INK)
            tick.set_fontweight("bold")
        else:
            tick.set_color(AXIS)
    means = [statistics.mean(values[index]) for index in range(len(configs))]
    axis.scatter(
        means,
        range(len(configs)),
        marker="D",
        s=16,
        facecolor="white",
        edgecolor=INK,
        linewidth=0.8,
        zorder=3,
    )
    reference = means[configs.index("hier_paragraph")]
    axis.axvline(reference, color=AXIS, linewidth=0.7, linestyle="--", zorder=0)

    boundaries = []
    position = 0
    for _, members in ORDER[:-1]:
        position += len([c for c in members if c in scores and c not in EXCLUDED])
        boundaries.append(position - 0.5)
    for boundary in boundaries:
        axis.axhline(boundary, color=AXIS, linewidth=0.5, linestyle=":")
    handles = [
        Patch(facecolor=FAMILY_COLOURS[family], label=FAMILY_LABELS[family])
        for family, _ in ORDER
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markersize=4,
            label="Mean",
        )
    )
    handles.append(
        Line2D([], [], color=AXIS, linestyle="--", label="hier_paragraph mean")
    )
    axis.legend(
        handles=handles,
        loc="upper right",
        fontsize=7.5,
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="white",
        framealpha=1.0,
    )

    axis.set_xlabel(f"Per-question F1@{top_k}")
    axis.set_xlim(left=0)
    sns.despine(ax=axis, left=True)
    figure.tight_layout()
    return {
        "configurations": len(configs),
        "written": [str(path) for path in save(figure, output_path)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the RQ1 F1 distribution figure."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/figures/rq1_all_acts_f1_at_10_distribution"),
    )
    parser.add_argument("--query-metrics-path", type=Path, default=QUERY_METRICS_PATH)
    parser.add_argument("--search-scope", default="all_acts")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_rq1_f1_distribution(
        output_path=args.output_path,
        query_metrics_path=args.query_metrics_path,
        search_scope=args.search_scope,
        top_k=args.top_k,
    )
    print("RQ1 F1 distribution complete: " + ", ".join(summary["written"]))


if __name__ == "__main__":
    main()
