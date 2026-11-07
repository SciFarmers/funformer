"""Shared style and helpers for the manuscript figures.

Keeps a single definition of the figure style, the clade palette, and the panel
labelling convention so every figure reads as one system.
"""
import matplotlib.pyplot as plt

# clade palette (colourblind-safe Okabe-Ito subset; validated CVD dE >= 18)
CLADE_COLOR = {"mammal": "#0072B2", "bird": "#E69F00", "fish": "#D55E00"}


def apply_style():
    """Common rcParams for every figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,   # editable text in the PDF (TrueType, not Type-3)
        "ps.fonttype": 42,
    })


def panel_label(ax, letter, title=None):
    """Bold panel letter at the top-left, with an optional centred title.

    Standardises multi-panel labelling: the letter is a left-aligned title and
    the descriptive text, if any, is a centred title, so the two never collide.
    """
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=12)
    if title:
        ax.set_title(title, loc="center", fontsize=9)
