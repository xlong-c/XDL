"""Generate figures for chapter 08 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter08_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch08_*.png
"""

from math import pi, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def fig_likelihood_curve() -> None:
    """Plot likelihood and log-likelihood for a Normal mean."""
    data = np.array([4.8, 5.3, 5.1, 4.9, 5.4, 5.0, 5.2, 4.7, 5.1, 5.3])
    sigma = 0.35
    mu_grid = np.linspace(4.35, 5.65, 500)
    log_likelihood = -0.5 * np.sum(((data[:, None] - mu_grid[None, :]) / sigma) ** 2, axis=0)
    log_likelihood = log_likelihood - len(data) * np.log(sqrt(2.0 * pi) * sigma)
    likelihood_scaled = np.exp(log_likelihood - log_likelihood.max())
    mle = float(data.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    axes[0].plot(mu_grid, likelihood_scaled, color="#1a73e8", lw=2.4)
    axes[0].axvline(mle, color="#d93025", ls="--", lw=1.7, label=f"MLE = {mle:.2f}")
    axes[0].scatter(data, np.full_like(data, 0.03), color="#202124", s=30, label="observed data")
    axes[0].set_title("Likelihood as support for parameter values")
    axes[0].set_xlabel("candidate mean mu")
    axes[0].set_ylabel("scaled likelihood")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8)

    axes[1].plot(mu_grid, log_likelihood, color="#34a853", lw=2.4)
    axes[1].axvline(mle, color="#d93025", ls="--", lw=1.7, label="maximum")
    axes[1].set_title("Log-likelihood has the same maximizer")
    axes[1].set_xlabel("candidate mean mu")
    axes[1].set_ylabel("log-likelihood")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=8)

    fig.suptitle("For Normal data with known sigma, MLE of the mean is the sample mean", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch08_likelihood_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig_confidence_interval_coverage() -> None:
    """Draw repeated 95% confidence intervals for a Normal mean."""
    rng = np.random.default_rng(41)
    mu = 0.0
    sigma = 1.0
    n = 25
    reps = 80
    z = 1.96
    samples = rng.normal(mu, sigma, size=(reps, n))
    means = samples.mean(axis=1)
    half_width = z * sigma / sqrt(n)
    lower = means - half_width
    upper = means + half_width
    cover = (lower <= mu) & (mu <= upper)

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    y = np.arange(reps)
    for i in range(reps):
        color = "#1a73e8" if cover[i] else "#d93025"
        ax.hlines(y[i], lower[i], upper[i], color=color, lw=1.7, alpha=0.9)
        ax.plot(means[i], y[i], "o", color=color, ms=3.0)
    ax.axvline(mu, color="#202124", lw=2.0, label="true mean")
    ax.set_title("Repeated 95% confidence intervals")
    ax.set_xlabel("mean value")
    ax.set_ylabel("repeated samples")
    ax.set_yticks([])
    ax.grid(alpha=0.18, axis="x")
    ax.text(
        0.98,
        0.04,
        f"covered: {cover.sum()}/{reps}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dadce0", "boxstyle": "round,pad=0.25"},
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch08_confidence_interval_coverage.png", bbox_inches="tight")
    plt.close(fig)


def fig_sample_size_interval_width() -> None:
    """Plot confidence interval width as sample size increases."""
    sigma_values = [0.6, 1.0, 1.5]
    ns = np.arange(10, 401)
    z = 1.96

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for sigma, color in zip(sigma_values, ["#34a853", "#1a73e8", "#d93025"]):
        width = 2.0 * z * sigma / np.sqrt(ns)
        ax.plot(ns, width, color=color, lw=2.2, label=f"sigma={sigma}")
    ax.set_title("Confidence interval width shrinks like 1/sqrt(n)")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("95% CI total width")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.annotate(
        "4x samples\nabout half width",
        xy=(200, 2.0 * z / sqrt(200)),
        xytext=(235, 0.55),
        arrowprops={"arrowstyle": "->", "color": "#5f6368", "lw": 1.5},
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch08_sample_size_interval_width.png", bbox_inches="tight")
    plt.close(fig)


def fig_estimator_variability() -> None:
    """Compare repeated-sampling variability of several estimators."""
    rng = np.random.default_rng(47)
    reps = 14000
    n = 25
    data = rng.normal(loc=0.0, scale=1.0, size=(reps, n))
    sorted_data = np.sort(data, axis=1)
    trim = int(0.1 * n)
    estimators = {
        "sample mean": data.mean(axis=1),
        "sample median": np.median(data, axis=1),
        "10% trimmed mean": sorted_data[:, trim : n - trim].mean(axis=1),
    }

    grid = np.linspace(-0.9, 0.9, 500)
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = ["#1a73e8", "#d93025", "#34a853"]
    for (name, values), color in zip(estimators.items(), colors):
        ax.hist(values, bins=70, density=True, histtype="step", lw=2.2, color=color, label=name)
        ax.text(
            0.58,
            ax.get_ylim()[1] * (0.86 - 0.08 * colors.index(color)),
            f"{name}: var={values.var(ddof=1):.3f}",
            color=color,
            fontsize=9,
        )
    ax.plot(grid, _normal_pdf(grid, 0.0, 1.0 / sqrt(n)), color="#202124", ls="--", lw=1.6, label="mean theory")
    ax.axvline(0.0, color="#202124", lw=1.2)
    ax.set_title("Different unbiased-ish estimators can have different variability")
    ax.set_xlabel("estimate of mean")
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch08_estimator_variability.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 08 figures."""
    fig_likelihood_curve()
    fig_confidence_interval_coverage()
    fig_sample_size_interval_width()
    fig_estimator_variability()


if __name__ == "__main__":
    main()
