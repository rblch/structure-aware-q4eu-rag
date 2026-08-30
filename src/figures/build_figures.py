from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures.gdpr_similarity_profile import write_gdpr_similarity_profile
from figures.rq1_chunk_length_distributions import (
    write_rq1_chunk_length_distributions,
)
from figures.rq1_chunk_size_answer_unit_diagnostic import (
    write_rq1_chunk_size_answer_unit_diagnostic,
)
from figures.rq1_f1_distribution import write_rq1_f1_distribution
from figures.rq1_paired_differences import write_rq1_paired_differences
from figures.rq2_abstention import write_rq2_abstention
from figures.rq2_answer_quality import write_rq2_answer_quality

FIGURES: dict[str, Callable[..., dict[str, Any]]] = {
    "gdpr_similarity_profile": write_gdpr_similarity_profile,
    "rq1_chunk_length_distributions": write_rq1_chunk_length_distributions,
    "rq1_chunk_size_answer_unit_diagnostic": (
        write_rq1_chunk_size_answer_unit_diagnostic
    ),
    "rq1_all_acts_f1_at_10_distribution": write_rq1_f1_distribution,
    "rq1_paired_differences": write_rq1_paired_differences,
    "rq2_answer_quality_by_condition": write_rq2_answer_quality,
    "rq2_abstention_by_condition": write_rq2_abstention,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build thesis figures.")
    parser.add_argument("--name", choices=sorted(FIGURES), action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    names = args.name or sorted(FIGURES)
    for name in names:
        summary = FIGURES[name](output_path=args.output_dir / name)
        print(f"{name}: " + ", ".join(summary["written"]))


if __name__ == "__main__":
    main()
