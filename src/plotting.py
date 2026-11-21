"""Estilo de gráficos consistente (paleta categórica validada para daltonismo)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Orden fijo: el cluster i siempre tiene el mismo color en todas las figuras.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#6250d6", "#e34948"]
NOISE_COLOR = "#898781"
TEXT = "#0b0b0b"
MUTED = "#898781"

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#2a78d6", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_blue_orange", ["#1c5cab", "#f0efec", "#d95926"])
CAT_CMAP = ListedColormap(PALETTE)


def set_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.dpi": 100,
    })


def cluster_colors(labels):
    return [NOISE_COLOR if l < 0 else PALETTE[l % len(PALETTE)] for l in labels]


def scatter_clusters(ax, X2, labels, title=""):
    """Dispersión 2D coloreada por cluster con leyenda (identidad no solo por color)."""
    import numpy as np
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(X2[m, 0], X2[m, 1], s=8, alpha=0.6, linewidths=0,
                   color=NOISE_COLOR if c < 0 else PALETTE[c % len(PALETTE)],
                   label=f"Cluster {c}" if c >= 0 else "Ruido")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8, loc="best")
    return ax
