from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from figures.style import AXIS, BLUE, NEUTRAL, RED, apply_style, save

QUERY_METRICS_PATH = Path("data/results/tables/retrieval_query_metrics.csv")
TIE_DECIMALS = 12
PRIMARY_CONFIG = "hier_paragraph"
BASELINES = [("fs_256_50", "fixed-size"), ("sem_50_256", "semantic")]


def read_query_f1(
    query_metrics_path: Path,
    *,
    search_scope: str,
    top_k: int,
) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    with query_metrics_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["search_scope"] != search_scope or int(row["top_k"]) != top_k:
                continue
            scores[row["query_id"]][row["config_id"]] = float(row["f1"])
    return scores


def paired_differences(
    scores: dict[str, dict[str, float]],
    baseline_config: str,
) -> list[float]:
    differences = [
        round(row[PRIMARY_CONFIG] - row[baseline_config], TIE_DECIMALS)
        for row in scores.values()
        if PRIMARY_CONFIG in row and baseline_config in row
    ]
    return sorted(differences, reverse=True)


def write_rq1_paired_differences(
    *,
    output_path: Path,
    query_metrics_path: Path = QUERY_METRICS_PATH,
    search_scope: str = "all_acts",
    top_k: int = 10,
) -> dict[str, Any]:
    apply_style()
    scores = read_query_f1(query_metrics_path, search_scope=search_scope, top_k=top_k)
    panels = {config: paired_differences(scores, config) for config, _ in BASELINES}
    limit = max(abs(value) for values in panels.values() for value in values)
    limit = round(limit + 0.05, 1)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    summary: dict[str, Any] = {"search_scope": search_scope, "top_k": top_k}
    for axis, (config, label) in zip(axes, BASELINES):
        values = panels[config]
        better = sum(value > 0 for value in values)
        worse = sum(value < 0 for value in values)
        tied = sum(value == 0 for value in values)
        colours = [BLUE if v > 0 else RED if v < 0 else NEUTRAL for v in values]
        axis.bar(range(len(values)), values, width=1.0, color=colours, linewidth=0)
        axis.axhline(0, color=AXIS, linewidth=0.7)
        axis.set_title(f"Hierarchical − {label}", loc="left", pad=6)
        axis.set_xlabel(f"{len(values)} questions, sorted by difference")
        axis.set_ylim(-limit, limit)
        axis.margins(x=0.01)
        axis.text(
            0.98,
            0.95,
            f"{better} favour hierarchical",
            transform=axis.transAxes,
            color=BLUE,
            fontsize=8,
            ha="right",
            va="top",
        )
        axis.text(
            0.02,
            0.05,
            f"{worse} favour baseline · {tied} tied",
            transform=axis.transAxes,
            color=RED,
            fontsize=8,
            va="bottom",
        )
        summary[config] = {"better": better, "worse": worse, "tied": tied}
    axes[0].set_ylabel("Paired F1@10 difference")
    figure.tight_layout()
    summary["written"] = [str(path) for path in save(figure, output_path)]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the RQ1 paired per-question F1 difference figure."
    )
    parser.add_argument(
        "--query-metrics-path",
        type=Path,
        default=QUERY_METRICS_PATH,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/figures/rq1_paired_differences"),
    )
    parser.add_argument("--search-scope", default="all_acts")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_rq1_paired_differences(
        query_metrics_path=args.query_metrics_path,
        output_path=args.output_path,
        search_scope=args.search_scope,
        top_k=args.top_k,
    )
    print("RQ1 paired-difference figure complete: " + ", ".join(summary["written"]))


if __name__ == "__main__":
    main()
