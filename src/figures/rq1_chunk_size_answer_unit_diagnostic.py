from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from figures.style import (
    AXIS,
    CONFIRMATORY,
    FAMILY_COLOURS,
    FAMILY_LABELS,
    FAMILY_MARKERS,
    FAMILY_ORDER,
    INK,
    apply_style,
    family_of,
    save,
)

CHUNK_METRICS_PATH = Path("data/results/tables/chunk_size_sensitivity.csv")
RETRIEVAL_SUMMARY_PATH = Path("data/results/tables/retrieval_summary.csv")
EXCLUDED = {"hier_paragraph_contextualized"}


def read_diagnostic(
    chunk_metrics_path: Path,
    retrieval_summary_path: Path,
    *,
    search_scope: str,
    top_k: int,
) -> dict[str, list[tuple[float, float, float, str]]]:
    with chunk_metrics_path.open(encoding="utf-8") as handle:
        chunks = {row["config_id"]: row for row in csv.DictReader(handle)}
    points: dict[str, list[tuple[float, float, float, str]]] = {
        family: [] for family in FAMILY_ORDER
    }
    with retrieval_summary_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            config_id = row["config_id"]
            if row["search_scope"] != search_scope or int(row["top_k"]) != top_k:
                continue
            if config_id in EXCLUDED or config_id not in chunks:
                continue
            points[family_of(config_id)].append(
                (
                    float(chunks[config_id]["mean_chunk_tokens"]),
                    float(chunks[config_id]["mean_answer_units_per_chunk"]),
                    float(row["retrieved_answer_unit_count"]),
                    float(row["relevant_retrieved_count"]),
                    float(row["precision"]),
                    float(row["recall"]),
                    config_id,
                )
            )
    return {family: sorted(values) for family, values in points.items()}


def draw_panel(axis, points, value_index: int) -> None:
    for family in FAMILY_ORDER:
        values = points[family]
        colour = FAMILY_COLOURS[family]
        axis.plot(
            [value[0] for value in values],
            [value[value_index] for value in values],
            color=colour,
            linewidth=1.1,
            zorder=1,
        )
        for value in values:
            confirmatory = value[-1] in CONFIRMATORY
            axis.plot(
                value[0],
                value[value_index],
                marker=FAMILY_MARKERS[family],
                markersize=6 if confirmatory else 4.5,
                markerfacecolor=colour if confirmatory else "white",
                markeredgecolor=colour,
                markeredgewidth=1.1,
                zorder=2,
            )


def write_rq1_chunk_size_answer_unit_diagnostic(
    *,
    output_path: Path,
    chunk_metrics_path: Path = CHUNK_METRICS_PATH,
    retrieval_summary_path: Path = RETRIEVAL_SUMMARY_PATH,
    search_scope: str = "all_acts",
    top_k: int = 10,
) -> dict[str, Any]:
    apply_style()
    points = read_diagnostic(
        chunk_metrics_path,
        retrieval_summary_path,
        search_scope=search_scope,
        top_k=top_k,
    )

    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), sharex=True)

    axes[0].axhline(1.0, color=AXIS, linewidth=0.7, linestyle="--", zorder=0)
    draw_panel(axes[0], points, 1)
    axes[0].set_title("(a) Units covered per chunk", loc="left", pad=6)
    axes[0].set_ylabel("Mean answer units per chunk")
    axes[0].text(262, 1.035, "one unit per chunk", color=AXIS, fontsize=7, ha="right")

    draw_panel(axes[1], points, 2)
    axes[1].set_ylim(0, 22.5)
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].set_title(f"(b) All units retrieved at top-{top_k}", loc="left", pad=6)
    axes[1].set_ylabel("Mean answer units")

    draw_panel(axes[2], points, 3)
    axes[2].set_ylim(0, 3.0)
    axes[2].set_title(f"(c) Gold units retrieved at top-{top_k}", loc="left", pad=6)
    axes[2].set_ylabel("Mean gold answer units")

    handles = [
        Line2D(
            [],
            [],
            color=FAMILY_COLOURS[family],
            marker=FAMILY_MARKERS[family],
            markersize=4.5,
            markerfacecolor="white",
            label=FAMILY_LABELS[family],
        )
        for family in FAMILY_ORDER
    ]
    handles.append(
        Line2D(
            [],
            [],
            color=INK,
            marker="o",
            linestyle="none",
            markersize=6,
            label="Confirmatory configuration",
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )

    figure.supxlabel("Mean chunk length (tokens)", fontsize=9, y=0.115)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.27, top=0.90, wspace=0.33)
    return {
        "families": {family: len(values) for family, values in points.items()},
        "written": [str(path) for path in save(figure, output_path)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the chunk-size and answer-unit-volume diagnostic."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/figures/rq1_chunk_size_answer_unit_diagnostic"),
    )
    parser.add_argument("--chunk-metrics-path", type=Path, default=CHUNK_METRICS_PATH)
    parser.add_argument(
        "--retrieval-summary-path", type=Path, default=RETRIEVAL_SUMMARY_PATH
    )
    parser.add_argument("--search-scope", default="all_acts")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_rq1_chunk_size_answer_unit_diagnostic(
        output_path=args.output_path,
        chunk_metrics_path=args.chunk_metrics_path,
        retrieval_summary_path=args.retrieval_summary_path,
        search_scope=args.search_scope,
        top_k=args.top_k,
    )
    print("Chunk-size diagnostic complete: " + ", ".join(summary["written"]))


if __name__ == "__main__":
    main()
