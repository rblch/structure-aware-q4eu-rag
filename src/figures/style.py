from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"

CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW]
DIVERGING = (BLUE, RED)

INK = "#33322f"
AXIS = "#66655f"
GRID = "#dedcd6"
NEUTRAL = "#9a9a95"

FAMILY_ORDER = ("fixed_size", "semantic", "hierarchical")
FAMILY_COLOURS = {"fixed_size": BLUE, "semantic": ORANGE, "hierarchical": AQUA}
FAMILY_LABELS = {
    "fixed_size": "Fixed-size",
    "semantic": "Semantic",
    "hierarchical": "Hierarchical",
}
FAMILY_MARKERS = {"fixed_size": "o", "semantic": "s", "hierarchical": "^"}
CONFIRMATORY = ("fs_256_50", "sem_50_256", "hier_paragraph")


def family_of(config_id: str) -> str:
    if config_id.startswith("fs"):
        return "fixed_size"
    return "semantic" if config_id.startswith("sem") else "hierarchical"


OUTPUT_FORMATS = (".pdf", ".png")


def apply_style(style: str = "ticks") -> None:
    sns.set_theme(
        style=style,
        context="paper",
        palette=CATEGORICAL,
        font="Helvetica",
        rc={
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "grid.color": GRID,
            "text.color": INK,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
        },
    )
    sns.set_color_codes()
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def save(figure: plt.Figure, output_path: Path) -> list[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in OUTPUT_FORMATS:
        target = output_path.with_suffix(suffix)
        figure.savefig(target, bbox_inches="tight")
        written.append(target)
    plt.close(figure)
    return written
