"""Generate figures for chapter 01 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter01_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch01_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def _base_event_axes(ax: plt.Axes) -> tuple[Circle, Circle]:
    """Draw Omega, A and B as a reusable event diagram."""
    ax.add_patch(Rectangle((0.05, 0.08), 0.9, 0.82, facecolor="#f8f9fa", edgecolor="#5f6368", lw=1.5))
    a = Circle((0.42, 0.5), 0.24, facecolor="none", edgecolor="#1a73e8", lw=2.0)
    b = Circle((0.6, 0.5), 0.24, facecolor="none", edgecolor="#d93025", lw=2.0)
    ax.add_patch(a)
    ax.add_patch(b)
    ax.text(0.08, 0.86, "Omega", fontsize=10, color="#202124")
    ax.text(0.3, 0.72, "A", fontsize=11, color="#1a73e8")
    ax.text(0.69, 0.72, "B", fontsize=11, color="#d93025")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    return a, b


def fig_event_operations() -> None:
    """Plot event operations inside the sample space."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0))
    titles = ["A union B", "A intersection B", "A complement", "A minus B"]

    for ax, title in zip(axes.flat, titles):
        a, b = _base_event_axes(ax)
        if title == "A union B":
            a.set_facecolor("#8ab4f8")
            a.set_alpha(0.75)
            b.set_facecolor("#f28b82")
            b.set_alpha(0.65)
        elif title == "A intersection B":
            a.set_facecolor("#fce8e6")
            a.set_alpha(0.35)
            b.set_facecolor("#fce8e6")
            b.set_alpha(0.35)
            overlap = Circle((0.51, 0.5), 0.15, facecolor="#a142f4", edgecolor="none", alpha=0.8)
            ax.add_patch(overlap)
        elif title == "A complement":
            ax.add_patch(Rectangle((0.05, 0.08), 0.9, 0.82, facecolor="#c2e7ff", edgecolor="none", alpha=0.65))
            a.set_facecolor("#ffffff")
            a.set_alpha(1.0)
            b.set_facecolor("none")
        elif title == "A minus B":
            a.set_facecolor("#8ab4f8")
            a.set_alpha(0.78)
            b.set_facecolor("#ffffff")
            b.set_alpha(1.0)
            b.set_zorder(5)
            ax.add_patch(Circle((0.6, 0.5), 0.24, facecolor="none", edgecolor="#d93025", lw=2.0))
        ax.set_title(title)

    fig.suptitle("Events are subsets of the sample space", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch01_event_operations.png", bbox_inches="tight")
    plt.close(fig)


def fig_classical_tree() -> None:
    """Plot a simple classical model tree for two coin tosses."""
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "start": (0.08, 0.5),
        "h1": (0.35, 0.72),
        "t1": (0.35, 0.28),
        "hh": (0.68, 0.84),
        "ht": (0.68, 0.6),
        "th": (0.68, 0.4),
        "tt": (0.68, 0.16),
    }
    edges = [
        ("start", "h1", "H : 1/2"),
        ("start", "t1", "T : 1/2"),
        ("h1", "hh", "H : 1/2"),
        ("h1", "ht", "T : 1/2"),
        ("t1", "th", "H : 1/2"),
        ("t1", "tt", "T : 1/2"),
    ]

    for src, dst, label in edges:
        (x0, y0), (x1, y1) = nodes[src], nodes[dst]
        ax.plot([x0, x1], [y0, y1], color="#5f6368", lw=1.8)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(xm - 0.02, ym + 0.03, label, fontsize=10, color="#202124")

    leaf_info = [
        ("hh", "HH", "#1a73e8"),
        ("ht", "HT", "#34a853"),
        ("th", "TH", "#f9ab00"),
        ("tt", "TT", "#d93025"),
    ]

    for key, label, color in leaf_info:
        x, y = nodes[key]
        ax.scatter([x], [y], s=120, color=color, zorder=5)
        ax.text(x + 0.03, y, f"{label}, P = 1/4", va="center", fontsize=11)

    for key, label in [("start", "start"), ("h1", "1st toss: H"), ("t1", "1st toss: T")]:
        x, y = nodes[key]
        ax.scatter([x], [y], s=80, color="#202124", zorder=5)
        ax.text(x - 0.01, y + 0.06, label, fontsize=10)

    ax.text(0.08, 0.93, "Classical model example: two fair coin tosses", fontsize=14, weight="bold")
    ax.text(0.08, 0.06, "Every leaf is equally likely, so counting leaves gives probabilities.", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch01_classical_tree.png", bbox_inches="tight")
    plt.close(fig)


def fig_geometric_probability() -> None:
    """Plot a geometric probability example in the unit square."""
    x = np.linspace(0, 1, 400)
    y = np.sqrt(np.clip(1 - x**2, 0, None))

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.fill_between(x, 0, y, color="#8ab4f8", alpha=0.85, label="event: x^2 + y^2 <= 1")
    ax.plot(x, y, color="#1a73e8", lw=2.2, label="quarter circle")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#202124", lw=1.6))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Geometric probability in the unit square")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="lower left")
    ax.annotate(
        "area(event) = pi / 4\narea(square) = 1\nP(event) = pi / 4",
        xy=(0.62, 0.78),
        xytext=(0.35, 0.28),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch01_geometric_probability.png", bbox_inches="tight")
    plt.close(fig)


def fig_frequency_stabilization() -> None:
    """Plot the stabilization of relative frequency."""
    rng = np.random.default_rng(42)
    n = 2000
    p = 0.6
    samples = rng.binomial(1, p, size=n)
    freq = np.cumsum(samples) / np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(np.arange(1, n + 1), freq, color="#1a73e8", lw=1.5, label="relative frequency")
    ax.axhline(p, color="#d93025", lw=2.0, ls="--", label="true probability p = 0.6")
    ax.fill_between(np.arange(1, n + 1), p - 0.05, p + 0.05, color="#fce8e6", alpha=0.6, label="stable band")
    ax.set_xlim(1, n)
    ax.set_ylim(0.3, 0.9)
    ax.set_xlabel("number of trials")
    ax.set_ylabel("relative frequency of success")
    ax.set_title("Frequency stabilizes as trials accumulate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper right")
    ax.annotate(
        "early trials fluctuate a lot",
        xy=(60, freq[59]),
        xytext=(260, 0.82),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    ax.annotate(
        "later the curve stays near p",
        xy=(1650, freq[1649]),
        xytext=(1100, 0.44),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch01_frequency_stabilization.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 01 figures."""
    fig_event_operations()
    fig_classical_tree()
    fig_geometric_probability()
    fig_frequency_stabilization()


if __name__ == "__main__":
    main()
