"""Consistent, presentation-ready chart style for the whole project.

Dark retro sci-fi look (terminal / HUD aesthetic).
The palette avoids red-green combinations and was checked with a
red-green color blindness simulation (all class pairs stay distinguishable).

Usage in a notebook:
    from personality.plot_style import apply_style, plot_class_shares
    apply_style()
"""
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from personality.config import CLASS_ORDER, FIGURES_DIR

# --- Base colors -------------------------------------------------------------
BACKGROUND = "#0B0E14"   # near-black, slightly blue
PANEL = "#121722"        # for boxes / table backgrounds
TEXT = "#D6E2EA"         # pale terminal white
MUTED = "#5C6878"        # axes, grid, secondary text
ACCENT = "#FFB000"       # amber HUD accent (titles, highlights)

# --- One fixed color per personality type (red-green safe) --------------------
CLASS_COLORS = dict(zip(
    CLASS_ORDER,
    ["#8A93A8",   # Moderate: neutral steel (the "default" type)
     "#00D9FF",   # Resilient: calm cyan
     "#A970FF",   # Overcontroller: violet
     "#FFB000"],  # Undercontroller: amber
))

# --- Colormaps for heatmaps / correlations ------------------------------------
# Sequential: dark -> cyan (for counts, importance, confusion matrix)
SEQ_CMAP = LinearSegmentedColormap.from_list("hud_seq", [BACKGROUND, "#0A4A5C", "#00D9FF"])
# Diverging: violet <- dark -> amber (for correlations, no red/green)
DIV_CMAP = LinearSegmentedColormap.from_list("hud_div", ["#A970FF", BACKGROUND, "#FFB000"])

FONT = "DejaVu Sans Mono"  # ships with matplotlib, works on every machine


def apply_style() -> None:
    """Set global matplotlib/seaborn defaults. Call once at the top of a notebook."""
    sns.set_theme(style="dark", context="talk")
    plt.rcParams.update({
        # Canvas
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        # Text
        "font.family": "monospace",
        "font.monospace": [FONT],
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": TEXT,
        "axes.titlecolor": ACCENT,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlesize": 17,
        "axes.titlepad": 14,
        # Frame and grid
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#1E2633",
        "grid.linewidth": 0.8,
        # Bars / patches without white outline
        "patch.edgecolor": BACKGROUND,
        "patch.linewidth": 0,
        # Legend
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        # Default color cycle = class colors
        "axes.prop_cycle": plt.cycler(color=list(CLASS_COLORS.values())),
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG into reports/figures for the presentation."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=200)


def plot_class_shares(shares, title: str, name: str | None = None):
    """Horizontal bar chart of class shares (0 to 1) with % labels.

    shares: pandas Series indexed by class name, e.g. value_counts(normalize=True)
    name:   if given, the figure is saved to reports/figures/<name>.png
    """
    shares = shares.reindex(CLASS_ORDER)
    colors = [CLASS_COLORS[c] for c in CLASS_ORDER]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(shares.index, shares, color=colors, height=0.6)
    ax.bar_label(bars, labels=[f"{p * 100:.1f} %" for p in shares], padding=8)

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title(title.upper())

    if name:
        save_fig(fig, name)
    return fig, ax
