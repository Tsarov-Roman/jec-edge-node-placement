import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "Liberation Serif",
    "font.size": 12,
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
})
CM = 1 / 2.54


def box(ax, x, y, w, h, text, fc="white", ec="black", lw=1.0, fs=11, style="round,pad=0.02", bold=False, ls="-"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, wrap=True,
            fontweight="bold" if bold else "normal")
    return (x, y, w, h)


def arrow(ax, p1, p2, text=None, fs=10, ls="-", lw=1.0, rad=0.0, off=(0, 0.02), color="black", ha="center"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, lw=lw, ls=ls, color=color,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1]
        ax.text(mx, my, text, ha=ha, va="bottom", fontsize=fs)


def scheme(w_cm=16, h_cm=9):
    fig, ax = plt.subplots(figsize=(w_cm * CM, h_cm * CM))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax
