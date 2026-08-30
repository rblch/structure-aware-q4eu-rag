from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter, LogLocator, NullFormatter, PercentFormatter

from figures.style import (
    AXIS,
    FAMILY_COLOURS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    INK,
    apply_style,
    save,
)

FIXED_SIZE_CHUNKS_PATH = Path("data/chunks/fixed_size/fs_256_50.json")
SEMANTIC_CHUNKS_PATH = Path("data/chunks/semantic/sem_50_256.json")
HIERARCHICAL_CHUNKS_PATH = Path("data/chunks/hierarchical/hier_paragraph.json")

# Shared logarithmic bins make the three panels horizontally comparable while
# retaining the long right tail of the hierarchical collection.
LOG_BINS = np.geomspace(3, 2048, 17)


def read_token_counts(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = payload["chunks"] if isinstance(payload, dict) else payload
    values = [int(chunk["token_count"]) for chunk in chunks]
    if not values:
        raise ValueError(f"Chunk collection is empty: {path}")
    return values


def read_confirmatory_chunk_lengths(
    *,
    fixed_size_path: Path,
    semantic_path: Path,
    hierarchical_path: Path,
) -> dict[str, list[int]]:
    return {
        "fixed_size": read_token_counts(fixed_size_path),
        "semantic": read_token_counts(semantic_path),
        "hierarchical": read_token_counts(hierarchical_path),
    }


def write_rq1_chunk_length_distributions(
    *,
    output_path: Path,
    fixed_size_path: Path = FIXED_SIZE_CHUNKS_PATH,
    semantic_path: Path = SEMANTIC_CHUNKS_PATH,
    hierarchical_path: Path = HIERARCHICAL_CHUNKS_PATH,
) -> dict[str, Any]:
    apply_style("whitegrid")
    lengths = read_confirmatory_chunk_lengths(
        fixed_size_path=fixed_size_path,
        semantic_path=semantic_path,
        hierarchical_path=hierarchical_path,
    )

    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), sharex=True)
    summary: dict[str, Any] = {"configurations": {}}

    for panel_index, (axis, family) in enumerate(zip(axes, FAMILY_ORDER)):
        values = lengths[family]
        colour = FAMILY_COLOURS[family]
        median = statistics.median(values)
        weights = np.full(len(values), 100.0 / len(values))

        axis.hist(
            values,
            bins=LOG_BINS,
            weights=weights,
            color=colour,
            edgecolor="white",
            linewidth=0.45,
        )
        axis.axvline(
            median,
            color=AXIS,
            linewidth=0.9,
            linestyle="--",
            zorder=3,
        )
        axis.set_xscale("log")
        axis.set_xlim(LOG_BINS[0], LOG_BINS[-1])
        axis.xaxis.set_major_locator(LogLocator(base=10))
        axis.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=True))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.set_ylim(bottom=0)
        axis.grid(axis="x", visible=False)
        axis.set_title(
            f"({chr(97 + panel_index)}) {FAMILY_LABELS[family]}",
            loc="left",
            pad=6,
        )
        axis.set_ylabel("Share of chunks")
        annotation_x = 0.72 if family == "fixed_size" else 0.97
        annotation_alignment = "left" if family == "fixed_size" else "right"
        axis.text(
            annotation_x,
            0.94,
            f"n={len(values):,}\nmedian={median:g}",
            transform=axis.transAxes,
            color=INK,
            fontsize=7.5,
            ha=annotation_alignment,
            va="top",
        )
        summary["configurations"][family] = {
            "chunks": len(values),
            "median_tokens": float(median),
            "maximum_tokens": max(values),
        }

    figure.supxlabel("Chunk length (tokens, logarithmic scale)", fontsize=9, y=0.08)
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.25,
        top=0.88,
        wspace=0.34,
    )
    summary["written"] = [str(path) for path in save(figure, output_path)]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build confirmatory chunk-length distribution histograms."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/figures/rq1_chunk_length_distributions"),
    )
    parser.add_argument("--fixed-size-path", type=Path, default=FIXED_SIZE_CHUNKS_PATH)
    parser.add_argument("--semantic-path", type=Path, default=SEMANTIC_CHUNKS_PATH)
    parser.add_argument(
        "--hierarchical-path", type=Path, default=HIERARCHICAL_CHUNKS_PATH
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_rq1_chunk_length_distributions(
        output_path=args.output_path,
        fixed_size_path=args.fixed_size_path,
        semantic_path=args.semantic_path,
        hierarchical_path=args.hierarchical_path,
    )
    print("RQ1 chunk-length distributions complete: " + ", ".join(summary["written"]))


if __name__ == "__main__":
    main()
