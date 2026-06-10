"""Generate figures for chapter 11 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter11_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch11_*.png
"""

from math import gamma, pi, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
    return (1.0 / sqrt(2.0 * pi)) * np.exp(-0.5 * x**2)


def _student_t_pdf(x: np.ndarray, df: float) -> np.ndarray:
    """Return Student t(df) density."""
    coef = gamma((df + 1.0) / 2.0) / (sqrt(df * pi) * gamma(df / 2.0))
    return coef * (1.0 + x**2 / df) ** (-(df + 1.0) / 2.0)


def fig_bootstrap_flow() -> None:
    """Draw bootstrap resampling workflow with a small distribution sketch."""
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2.0, y - h / 2.0),
                w,
                h,
                boxstyle="round,pad=0.018,rounding_size=0.022",
                facecolor=fc,
                edgecolor=ec,
                lw=2.0,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=10.5, color="#202124")

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="->",
                mutation_scale=16,
                lw=2.1,
                color="#5f6368",
            )
        )

    items = [
        (0.12, "Original sample\nx1,...,xn"),
        (0.34, "Resample\nwith replacement"),
        (0.56, "Compute statistic\nT*"),
        (0.78, "Repeat B times\nT1*,...,TB*"),
    ]
    colors = [
        ("#e8f0fe", "#1a73e8"),
        ("#e6f4ea", "#34a853"),
        ("#fff7d6", "#f9ab00"),
        ("#fce8e6", "#d93025"),
    ]
    for (x, text), (fc, ec) in zip(items, colors):
        box(x, 0.68, 0.18, 0.21, text, fc, ec)
    for i in range(len(items) - 1):
        arrow(items[i][0] + 0.095, 0.68, items[i + 1][0] - 0.095, 0.68)

    rng = np.random.default_rng(211)
    values = rng.normal(0.0, 1.0, size=2500)
    hist, edges = np.histogram(values, bins=36, density=True)
    hist = hist / hist.max() * 0.22
    centers = (edges[:-1] + edges[1:]) / 2.0
    x_scaled = 0.22 + (centers - centers.min()) / (centers.max() - centers.min()) * 0.56
    ax.fill_between(x_scaled, 0.17, 0.17 + hist, color="#1a73e8", alpha=0.35)
    ax.plot(x_scaled, 0.17 + hist, color="#1a73e8", lw=2.0)
    ax.text(0.50, 0.10, "Bootstrap distribution approximates sampling variability from the observed sample.", ha="center", fontsize=11)
    ax.set_title("Bootstrap approximates repeated sampling by resampling the data", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch11_bootstrap_flow.png", bbox_inches="tight")
    plt.close(fig)


def fig_rank_vs_parametric() -> None:
    """Compare raw values and rank-transformed values for two groups."""
    rng = np.random.default_rng(223)
    group_a = rng.lognormal(mean=0.0, sigma=0.45, size=24)
    group_b = rng.lognormal(mean=0.18, sigma=0.45, size=24)
    group_b[-1] = 5.8
    combined = np.concatenate([group_a, group_b])
    order = np.argsort(combined)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, combined.size + 1)
    ranks_a = ranks[: group_a.size]
    ranks_b = ranks[group_a.size :]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    jitter_a = rng.normal(0.0, 0.025, size=group_a.size)
    jitter_b = rng.normal(0.0, 0.025, size=group_b.size)
    axes[0].scatter(np.full_like(group_a, 0.95) + jitter_a, group_a, color="#1a73e8", alpha=0.85, label="A")
    axes[0].scatter(np.full_like(group_b, 2.05) + jitter_b, group_b, color="#d93025", alpha=0.85, label="B")
    axes[0].hlines(group_a.mean(), 0.75, 1.15, color="#1a73e8", lw=3.0, label="mean")
    axes[0].hlines(group_b.mean(), 1.85, 2.25, color="#d93025", lw=3.0)
    axes[0].hlines(np.median(group_a), 0.75, 1.15, color="#202124", lw=2.0, ls="--", label="median")
    axes[0].hlines(np.median(group_b), 1.85, 2.25, color="#202124", lw=2.0, ls="--")
    axes[0].set_xlim(0.5, 2.5)
    axes[0].set_xticks([1.0, 2.0], ["group A", "group B"])
    axes[0].set_title("Raw values: mean is sensitive to outliers")
    axes[0].set_ylabel("observed value")
    axes[0].grid(alpha=0.22, axis="y")
    axes[0].legend(fontsize=8)

    axes[1].scatter(np.full_like(ranks_a, 0.95) + jitter_a, ranks_a, color="#1a73e8", alpha=0.85, label="A ranks")
    axes[1].scatter(np.full_like(ranks_b, 2.05) + jitter_b, ranks_b, color="#d93025", alpha=0.85, label="B ranks")
    axes[1].hlines(ranks_a.mean(), 0.75, 1.15, color="#1a73e8", lw=3.0, label="mean rank")
    axes[1].hlines(ranks_b.mean(), 1.85, 2.25, color="#d93025", lw=3.0)
    axes[1].set_xlim(0.5, 2.5)
    axes[1].set_xticks([1.0, 2.0], ["group A", "group B"])
    axes[1].set_title("Rank view: compare relative order")
    axes[1].set_ylabel("rank in pooled sample")
    axes[1].grid(alpha=0.22, axis="y")
    axes[1].legend(fontsize=8)

    fig.suptitle("Rank methods use order information instead of raw scale assumptions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch11_rank_vs_parametric.png", bbox_inches="tight")
    plt.close(fig)


def fig_outlier_mean_median() -> None:
    """Show how a moving outlier affects mean and median."""
    rng = np.random.default_rng(227)
    base = rng.normal(0.0, 1.0, size=31)
    outliers = np.linspace(0.0, 18.0, 220)
    means = []
    medians = []
    trimmed = []
    for value in outliers:
        sample = np.concatenate([base, [value]])
        sorted_sample = np.sort(sample)
        trim = 3
        means.append(float(sample.mean()))
        medians.append(float(np.median(sample)))
        trimmed.append(float(sorted_sample[trim:-trim].mean()))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(outliers, means, color="#d93025", lw=2.4, label="mean")
    ax.plot(outliers, medians, color="#1a73e8", lw=2.4, label="median")
    ax.plot(outliers, trimmed, color="#34a853", lw=2.4, label="trimmed mean")
    ax.set_title("Outliers pull the mean much more than the median")
    ax.set_xlabel("value of one added outlier")
    ax.set_ylabel("location estimate")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(9.5, 0.4, "robust estimators change slowly\nwhen one point becomes extreme", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch11_outlier_mean_median.png", bbox_inches="tight")
    plt.close(fig)


def fig_heavy_tail_distribution() -> None:
    """Compare Normal and heavy-tailed t distribution."""
    x = np.linspace(-7.0, 7.0, 1000)
    normal = _normal_pdf(x)
    t3 = _student_t_pdf(x, 3.0)
    t5 = _student_t_pdf(x, 5.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    axes[0].plot(x, normal, color="#202124", lw=2.2, label="Normal")
    axes[0].plot(x, t5, color="#1a73e8", lw=2.2, label="t df=5")
    axes[0].plot(x, t3, color="#d93025", lw=2.2, label="t df=3")
    axes[0].set_title("Center and shoulder")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=9)

    mask = x >= 2.5
    axes[1].plot(x[mask], normal[mask], color="#202124", lw=2.2, label="Normal")
    axes[1].plot(x[mask], t5[mask], color="#1a73e8", lw=2.2, label="t df=5")
    axes[1].plot(x[mask], t3[mask], color="#d93025", lw=2.2, label="t df=3")
    axes[1].set_yscale("log")
    axes[1].set_title("Right tail on log scale")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("density, log scale")
    axes[1].grid(alpha=0.22, which="both")
    axes[1].legend(fontsize=9)

    fig.suptitle("Heavy tails make extreme observations much more common than Normal intuition suggests", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch11_heavy_tail_distribution.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 11 figures."""
    fig_bootstrap_flow()
    fig_rank_vs_parametric()
    fig_outlier_mean_median()
    fig_heavy_tail_distribution()


if __name__ == "__main__":
    main()
