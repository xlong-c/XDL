"""Generate figures for chapter 07 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter07_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch07_*.png
"""

from math import erf, gamma, pi, sqrt
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


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return Normal(mu, sigma^2) density."""
    return (1.0 / (sqrt(2.0 * pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def _chisq_pdf(x: np.ndarray, df: float) -> np.ndarray:
    """Return Chi-square(df) density."""
    y = np.zeros_like(x, dtype=float)
    mask = x > 0
    coef = 1.0 / ((2.0 ** (df / 2.0)) * gamma(df / 2.0))
    y[mask] = coef * (x[mask] ** (df / 2.0 - 1.0)) * np.exp(-x[mask] / 2.0)
    return y


def _t_pdf(x: np.ndarray, df: float) -> np.ndarray:
    """Return Student t(df) density."""
    coef = gamma((df + 1.0) / 2.0) / (sqrt(df * pi) * gamma(df / 2.0))
    return coef * (1.0 + x**2 / df) ** (-(df + 1.0) / 2.0)


def _f_pdf(x: np.ndarray, d1: float, d2: float) -> np.ndarray:
    """Return F(d1, d2) density."""
    y = np.zeros_like(x, dtype=float)
    mask = x > 0
    a = d1 / 2.0
    b = d2 / 2.0
    beta = gamma(a) * gamma(b) / gamma(a + b)
    coef = (d1 / d2) ** a / beta
    y[mask] = coef * (x[mask] ** (a - 1.0)) * (1.0 + d1 * x[mask] / d2) ** (-(a + b))
    return y


def fig_sampling_flow() -> None:
    """Draw population -> sample -> statistic -> sampling distribution flow."""
    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2.0, y - h / 2.0),
                w,
                h,
                boxstyle="round,pad=0.018,rounding_size=0.025",
                facecolor=fc,
                edgecolor=ec,
                lw=2.0,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=11, color="#202124")

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="->",
                mutation_scale=16,
                lw=2.2,
                color="#5f6368",
            )
        )
        ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.055, label, ha="center", fontsize=9)

    box(0.13, 0.60, 0.20, 0.24, "Population\nunknown full law\nparameter: mu, sigma^2", "#e8f0fe", "#1a73e8")
    box(0.38, 0.60, 0.20, 0.24, "Sample\nX1,...,Xn\nrandom data", "#e6f4ea", "#34a853")
    box(0.63, 0.60, 0.20, 0.24, "Statistic\nT(X1,...,Xn)\nmean, variance", "#fff7d6", "#f9ab00")
    box(0.87, 0.60, 0.20, 0.24, "Sampling\ndistribution\nlaw of T", "#fce8e6", "#d93025")

    arrow(0.24, 0.60, 0.27, 0.60, "draw")
    arrow(0.49, 0.60, 0.52, 0.60, "compute")
    arrow(0.74, 0.60, 0.76, 0.60, "repeat")

    rng = np.random.default_rng(5)
    pop = rng.normal(size=(120, 2)) * np.array([0.055, 0.08]) + np.array([0.13, 0.22])
    sample = pop[rng.choice(len(pop), size=14, replace=False)] + np.array([0.25, 0.0])
    ax.scatter(pop[:, 0], pop[:, 1], s=14, color="#1a73e8", alpha=0.25)
    ax.scatter(sample[:, 0], sample[:, 1], s=32, color="#34a853", alpha=0.9, edgecolors="white", linewidth=0.5)
    ax.text(0.50, 0.20, "A statistic is random because the sample is random.", ha="center", fontsize=12)
    ax.set_title("From population to sampling distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch07_sampling_flow.png", bbox_inches="tight")
    plt.close(fig)


def fig_sample_mean_distribution() -> None:
    """Simulate sampling distributions of the sample mean."""
    rng = np.random.default_rng(23)
    reps = 16000
    population = rng.exponential(scale=1.0, size=120000)
    means_5 = rng.exponential(scale=1.0, size=(reps, 5)).mean(axis=1)
    means_30 = rng.exponential(scale=1.0, size=(reps, 30)).mean(axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), sharey=False)
    axes[0].hist(population, bins=70, density=True, color="#9aa0a6", alpha=0.8, edgecolor="white")
    axes[0].set_title("Original population\nExponential(1)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("density")
    axes[0].set_xlim(0, 6)
    axes[0].grid(alpha=0.22)

    for ax, means, n, color in [
        (axes[1], means_5, 5, "#f9ab00"),
        (axes[2], means_30, 30, "#1a73e8"),
    ]:
        ax.hist(means, bins=55, density=True, color=color, alpha=0.82, edgecolor="white")
        grid = np.linspace(0.0, 2.2, 400)
        ax.plot(grid, _normal_pdf(grid, 1.0, 1.0 / sqrt(n)), color="#202124", lw=2.0, label="Normal approx")
        ax.axvline(1.0, color="#d93025", ls="--", lw=1.4, label="population mean")
        ax.set_title(f"Sampling distribution of sample mean\nn = {n}")
        ax.set_xlabel("sample mean")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8)
        ax.text(1.45, ax.get_ylim()[1] * 0.72, f"std approx = {1/sqrt(n):.2f}", fontsize=9)

    fig.suptitle("Sampling distribution is the distribution of a statistic under repeated sampling", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch07_sample_mean_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_classical_sampling_distributions() -> None:
    """Plot Chi-square, t and F distribution families."""
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    x_chi = np.linspace(0.001, 18.0, 700)
    for df, color in [(2, "#d93025"), (5, "#f9ab00"), (10, "#1a73e8")]:
        axes[0].plot(x_chi, _chisq_pdf(x_chi, df), color=color, lw=2.2, label=f"df={df}")
    axes[0].set_title("Chi-square")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8)

    x_t = np.linspace(-5.0, 5.0, 700)
    for df, color in [(2, "#d93025"), (5, "#f9ab00"), (30, "#1a73e8")]:
        axes[1].plot(x_t, _t_pdf(x_t, df), color=color, lw=2.2, label=f"df={df}")
    axes[1].plot(x_t, _normal_pdf(x_t, 0.0, 1.0), color="#202124", lw=1.6, ls="--", label="N(0,1)")
    axes[1].set_title("Student t")
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)

    x_f = np.linspace(0.001, 5.0, 700)
    for d1, d2, color in [(5, 5, "#d93025"), (5, 20, "#f9ab00"), (20, 20, "#1a73e8")]:
        axes[2].plot(x_f, _f_pdf(x_f, d1, d2), color=color, lw=2.2, label=f"d1={d1}, d2={d2}")
    axes[2].set_title("F")
    axes[2].set_xlabel("x")
    axes[2].grid(alpha=0.22)
    axes[2].legend(fontsize=8)

    fig.suptitle("Classical sampling distributions change shape with degrees of freedom", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch07_classical_sampling_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig_empirical_cdf() -> None:
    """Plot an empirical CDF against the true CDF."""
    rng = np.random.default_rng(31)
    n = 35
    sample = np.sort(rng.normal(loc=0.0, scale=1.0, size=n))
    y = np.arange(1, n + 1) / n
    xgrid = np.linspace(-3.2, 3.2, 500)
    true_cdf = 0.5 * (1.0 + np.vectorize(lambda z: erf(z / sqrt(2.0)))(xgrid))

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.step(sample, y, where="post", color="#1a73e8", lw=2.2, label="empirical CDF")
    ax.scatter(sample, y, color="#1a73e8", s=18, zorder=5)
    ax.plot(xgrid, true_cdf, color="#d93025", lw=2.0, ls="--", label="true Normal CDF")
    ax.set_title("Empirical distribution function is a step function")
    ax.set_xlabel("x")
    ax.set_ylabel("F_n(x)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(-3.0, 0.86, "each sample point adds a jump of 1/n", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch07_empirical_cdf.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 07 figures."""
    fig_sampling_flow()
    fig_sample_mean_distribution()
    fig_classical_sampling_distributions()
    fig_empirical_cdf()


if __name__ == "__main__":
    main()
