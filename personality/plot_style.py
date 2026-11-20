"""Consistent, presentation-ready chart style for the whole project.

Usage in a notebook:
    from personality.plot_style import apply_style, CLASS_COLORS, save_fig
    apply_style()
"""
import matplotlib.pyplot as plt
import seaborn as sns

from personality.config import CLASS_ORDER, FIGURES_DIR

# One fixed color per personality type, used in every chart
CLASS_COLORS = dict(zip(
    CLASS_ORDER,
    ["#8E9AAF", "#2A9D8F", "#E76F51", "#E9C46A"],
))
ACCENT = "#264653"
NEUTRAL = "#B0B0B0"


def apply_style() -> None:
    """Set global matplotlib/seaborn defaults. Call once at the top of a notebook."""
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlesize": 18,
        "axes.labelcolor": "#333333",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "legend.frameon": False,
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save a figure as PNG into reports/figures for the presentation."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
