"""Shared matplotlib style for the notebooks (reference data-viz palette, light mode).
Categorical hues are assigned in fixed order and follow the entity, never its rank."""
import matplotlib as mpl

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEUTRAL = "#8a8984"


def use():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
        "lines.linewidth": 2, "lines.markersize": 7, "font.size": 10,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "legend.frameon": False,
        "figure.dpi": 110,
    })
