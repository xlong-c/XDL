"""Generate figures for chapter 02 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter02_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch02_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def fig_conditional_probability() -> None:
    """Visualize that conditioning shrinks the reference space to B."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8))

    # Left: original sample space with A and B.
    ax = axes[0]
    ax.add_patch(Rectangle((0.05, 0.08), 0.9, 0.82, facecolor="#f8f9fa", edgecolor="#5f6368", lw=1.5))
    ax.add_patch(Rectangle((0.18, 0.18), 0.54, 0.54, facecolor="#8ab4f8", alpha=0.45, edgecolor="#1a73e8", lw=2.0))
    ax.add_patch(Rectangle((0.42, 0.32), 0.40, 0.42, facecolor="#f28b82", alpha=0.45, edgecolor="#d93025", lw=2.0))
    ax.add_patch(Rectangle((0.42, 0.32), 0.30, 0.40, facecolor="#a142f4", alpha=0.7, edgecolor="none"))
    ax.text(0.08, 0.86, "Omega", color="#202124")
    ax.text(0.28, 0.75, "B", color="#1a73e8", fontsize=12)
    ax.text(0.76, 0.76, "A", color="#d93025", fontsize=12)
    ax.text(0.49, 0.52, "A intersection B", color="white", fontsize=10, ha="center")
    ax.set_title("Before conditioning")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Right: inside B only, intersection becomes the new favorable region.
    ax = axes[1]
    ax.add_patch(Rectangle((0.12, 0.18), 0.76, 0.54, facecolor="#8ab4f8", alpha=0.35, edgecolor="#1a73e8", lw=2.0))
    ax.add_patch(Rectangle((0.42, 0.18), 0.30, 0.54, facecolor="#a142f4", alpha=0.75, edgecolor="none"))
    ax.text(0.14, 0.76, "reference space is now B", color="#1a73e8")
    ax.text(0.57, 0.47, "A intersection B", color="white", fontsize=10, ha="center")
    ax.text(0.18, 0.08, "P(A | B) = area(A intersection B) / area(B)", fontsize=11)
    ax.set_title("After conditioning on B")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("Conditioning means we renormalize inside the event B", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch02_conditional_probability.png", bbox_inches="tight")
    plt.close(fig)


def fig_bayes_update() -> None:
    """Flow chart for prior -> likelihood -> evidence -> posterior."""
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                facecolor=fc,
                edgecolor=ec,
                lw=2.0,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=11)

    def arrow(x0: float, y0: float, x1: float, y1: float, text: str) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=2.2, color="#5f6368"),
        )
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.07, text, ha="center", fontsize=10, color="#202124")

    box(0.13, 0.52, 0.18, 0.2, "Prior\nP(H)\nwhat we believed\nbefore data", "#e8f0fe", "#1a73e8")
    box(0.39, 0.52, 0.20, 0.2, "Likelihood\nP(E | H)\nhow compatible\nevidence is", "#fce8e6", "#d93025")
    box(0.65, 0.52, 0.18, 0.2, "Evidence\nP(E)\nall ways the\nevidence can appear", "#fff7d6", "#f9ab00")
    box(0.89, 0.52, 0.18, 0.2, "Posterior\nP(H | E)\nupdated belief", "#e6f4ea", "#34a853")

    arrow(0.22, 0.52, 0.29, 0.52, "combine")
    arrow(0.49, 0.52, 0.56, 0.52, "normalize by")
    arrow(0.74, 0.52, 0.80, 0.52, "get")

    ax.text(
        0.5,
        0.16,
        "Bayes rule:  P(H | E) = P(E | H) P(H) / P(E)",
        ha="center",
        fontsize=14,
        weight="bold",
        color="#202124",
    )
    ax.text(
        0.5,
        0.06,
        "Screening example: prior 1%, sensitivity 95%, false positive 5% -> posterior about 16.1%",
        ha="center",
        fontsize=10.5,
    )

    fig.tight_layout()
    fig.savefig(OUT / "ps_ch02_bayes_update.png", bbox_inches="tight")
    plt.close(fig)


def fig_independence_heatmaps() -> None:
    """Compare independent and dependent joint distributions."""
    independent = np.array([[0.25, 0.25], [0.25, 0.25]])
    dependent = np.array([[0.45, 0.05], [0.05, 0.45]])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    for ax, mat, title in zip(
        axes,
        [independent, dependent],
        ["Independent: joint = marginal product", "Dependent: same marginals, different joint"],
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="#202124", fontsize=12)
        ax.set_xticks([0, 1], labels=["B = 0", "B = 1"])
        ax.set_yticks([0, 1], labels=["A = 0", "A = 1"])
        ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, label="joint probability")
    fig.savefig(OUT / "ps_ch02_independence_heatmaps.png", bbox_inches="tight")
    plt.close(fig)


def fig_conditional_independence() -> None:
    """Fork graph for conditional independence intuition."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "D": (0.5, 0.75),
        "S1": (0.28, 0.3),
        "S2": (0.72, 0.3),
    }
    colors = {"D": "#1a73e8", "S1": "#34a853", "S2": "#f9ab00"}
    labels = {"D": "hidden cause\nDisease", "S1": "symptom 1\nFever", "S2": "symptom 2\nCough"}

    for src, dst in [("D", "S1"), ("D", "S2")]:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        ax.annotate(
            "",
            xy=(x1, y1 + 0.08),
            xytext=(x0, y0 - 0.08),
            arrowprops=dict(arrowstyle="->", lw=2.4, color="#5f6368"),
        )

    for key, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), 0.11, facecolor=colors[key], edgecolor="white", lw=2.0, alpha=0.95))
        ax.text(x, y, labels[key], ha="center", va="center", color="white", fontsize=12, weight="bold")

    ax.text(0.5, 0.92, "Conditional independence often comes from a shared hidden cause", ha="center", fontsize=14)
    ax.text(0.5, 0.08, "Given Disease, Fever and Cough can be modeled as approximately independent.", ha="center", fontsize=11)
    ax.text(0.5, 0.02, "Without conditioning, the two symptoms look dependent because both are driven by the same cause.", ha="center", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(OUT / "ps_ch02_conditional_independence.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 02 figures."""
    fig_conditional_probability()
    fig_bayes_update()
    fig_independence_heatmaps()
    fig_conditional_independence()


if __name__ == "__main__":
    main()
