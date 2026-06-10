"""Generate figures for chapter 06 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter06_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch06_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    """Return standard Normal density."""
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)


def fig_sample_mean_convergence() -> None:
    """Plot sample mean convergence for Bernoulli trials."""
    rng = np.random.default_rng(7)
    p = 0.6
    n = 2500
    trials = 6
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i in range(trials):
        samples = rng.binomial(1, p, size=n)
        means = np.cumsum(samples) / x
        ax.plot(x, means, lw=1.3, alpha=0.82, label=f"path {i + 1}")

    ax.axhline(p, color="#202124", ls="--", lw=2.0, label="true mean p = 0.6")
    ax.fill_between(x, p - 0.03, p + 0.03, color="#e8f0fe", alpha=0.65, label="stable band")
    ax.set_xscale("log")
    ax.set_ylim(0.35, 0.85)
    ax.set_xlabel("number of trials, log scale")
    ax.set_ylabel("sample mean")
    ax.set_title("Law of large numbers: sample mean stabilizes")
    ax.grid(alpha=0.24)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch06_sample_mean_convergence.png", bbox_inches="tight")
    plt.close(fig)


def fig_clt_standardized_histogram() -> None:
    """Plot standardized sums approaching standard Normal."""
    rng = np.random.default_rng(11)
    reps = 12000
    p = 0.35
    panels = [(5, "#f9ab00"), (30, "#34a853"), (120, "#1a73e8")]
    zgrid = np.linspace(-4.0, 4.0, 500)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), sharey=True)
    for ax, (n, color) in zip(axes, panels):
        samples = rng.binomial(1, p, size=(reps, n))
        sums = samples.sum(axis=1)
        z = (sums - n * p) / np.sqrt(n * p * (1.0 - p))
        ax.hist(z, bins=34, density=True, color=color, alpha=0.78, edgecolor="white")
        ax.plot(zgrid, _normal_pdf(zgrid), color="#202124", lw=2.0, label="N(0,1)")
        ax.set_title(f"n = {n}")
        ax.set_xlabel("standardized sum")
        ax.grid(alpha=0.22)
        ax.text(-3.7, 0.35, f"mean={z.mean():.2f}\nstd={z.std(ddof=0):.2f}", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("density")
        ax.legend(fontsize=8)

    fig.suptitle("Central limit theorem: standardized sums become bell-shaped", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch06_clt_standardized_histogram.png", bbox_inches="tight")
    plt.close(fig)


def fig_concentration_bounds() -> None:
    """Plot empirical tail probability and Hoeffding bound."""
    rng = np.random.default_rng(17)
    p = 0.5
    eps = 0.08
    reps = 18000
    ns = np.array([25, 50, 100, 200, 400, 800, 1600])
    empirical = []
    hoeffding = []
    empirical_plot = []

    for n in ns:
        samples = rng.binomial(1, p, size=(reps, n))
        means = samples.mean(axis=1)
        tail = float(np.mean(np.abs(means - p) >= eps))
        raw_bound = float(2.0 * np.exp(-2.0 * n * eps**2))
        empirical.append(tail)
        # Use a small plotting floor for zero-observed tails on a log axis.
        empirical_plot.append(max(tail, 0.5 / reps))
        hoeffding.append(min(1.0, raw_bound))

    empirical = np.array(empirical)
    empirical_plot = np.array(empirical_plot)
    hoeffding = np.array(hoeffding)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(ns, empirical_plot, "o-", color="#1a73e8", lw=2.2, label="empirical tail probability")
    ax.plot(ns, hoeffding, "s--", color="#d93025", lw=2.0, label="Hoeffding bound, clipped at 1")
    for n, tail, y in zip(ns, empirical, empirical_plot):
        if tail == 0.0:
            ax.text(n, y * 1.55, "0 observed", ha="center", fontsize=8, color="#1a73e8")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("sample size n, log scale")
    ax.set_ylabel("P(|sample mean - p| >= epsilon), log scale")
    ax.set_title("Concentration: larger samples make large deviations rare")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    ax.text(35, 0.018, "epsilon = 0.08\nBernoulli p = 0.5", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch06_concentration_bounds.png", bbox_inches="tight")
    plt.close(fig)


def fig_markov_chain_states() -> None:
    """Draw a simple Markov chain transition diagram."""
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "Sunny": (0.22, 0.58),
        "Cloudy": (0.52, 0.78),
        "Rainy": (0.78, 0.42),
    }
    colors = {
        "Sunny": "#f9ab00",
        "Cloudy": "#1a73e8",
        "Rainy": "#34a853",
    }
    transitions = [
        ("Sunny", "Sunny", "0.70", 0.18, (0.00, 0.00)),
        ("Sunny", "Cloudy", "0.20", 0.16, (-0.04, 0.05)),
        ("Sunny", "Rainy", "0.10", -0.16, (-0.03, -0.08)),
        ("Cloudy", "Sunny", "0.25", -0.16, (0.03, -0.06)),
        ("Cloudy", "Cloudy", "0.45", 0.16, (0.00, 0.00)),
        ("Cloudy", "Rainy", "0.30", 0.16, (0.05, 0.04)),
        ("Rainy", "Sunny", "0.15", -0.24, (0.02, -0.11)),
        ("Rainy", "Cloudy", "0.35", -0.16, (-0.04, -0.05)),
        ("Rainy", "Rainy", "0.50", 0.16, (0.00, 0.00)),
    ]

    for name, (x, y) in nodes.items():
        ax.add_patch(Circle((x, y), 0.09, facecolor=colors[name], edgecolor="white", lw=2.2, alpha=0.96, zorder=5))
        ax.text(x, y, name, color="white", weight="bold", ha="center", va="center", fontsize=12, zorder=6)

    for src, dst, label, rad, offset in transitions:
        x0, y0 = nodes[src]
        x1, y1 = nodes[dst]
        if src == dst:
            loop = FancyArrowPatch(
                (x0 - 0.045, y0 + 0.095),
                (x0 + 0.045, y0 + 0.095),
                connectionstyle="arc3,rad=1.7",
                arrowstyle="->",
                mutation_scale=13,
                lw=1.7,
                color="#5f6368",
                zorder=3,
            )
            ax.add_patch(loop)
            ax.text(x0, y0 + 0.18, label, ha="center", fontsize=9)
        else:
            arrow = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="->",
                mutation_scale=13,
                lw=1.7,
                color="#5f6368",
                shrinkA=70,
                shrinkB=70,
                zorder=2,
            )
            ax.add_patch(arrow)
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ax.text(
                xm + offset[0],
                ym + rad * 0.55 + offset[1],
                label,
                fontsize=9,
                ha="center",
                color="#202124",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
            )

    ax.text(0.5, 0.08, "Each row of the transition matrix sums to 1. Future state depends on current state.", ha="center", fontsize=11)
    ax.set_title("A simple Markov chain: weather state transitions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch06_markov_chain_states.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 06 figures."""
    fig_sample_mean_convergence()
    fig_clt_standardized_histogram()
    fig_concentration_bounds()
    fig_markov_chain_states()


if __name__ == "__main__":
    main()
