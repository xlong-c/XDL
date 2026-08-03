"""Generate figures for chapter 09 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter09_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch09_*.png
"""

from math import erf, pi, sqrt
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


def _normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Return Normal(mu, sigma^2) density."""
    return (1.0 / (sqrt(2.0 * pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def _normal_cdf_scalar(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Return Normal(mu, sigma^2) CDF for scalar x."""
    z = (x - mu) / (sigma * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def fig_rejection_regions() -> None:
    """Plot one-sided and two-sided rejection regions."""
    x = np.linspace(-4.0, 4.0, 900)
    y = _normal_pdf(x)
    z_two = 1.96
    z_one = 1.645

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharey=True)
    axes[0].plot(x, y, color="#202124", lw=2.2)
    axes[0].fill_between(x, y, where=(x <= -z_two), color="#d93025", alpha=0.55, label="reject H0")
    axes[0].fill_between(x, y, where=(x >= z_two), color="#d93025", alpha=0.55)
    axes[0].axvline(-z_two, color="#d93025", ls="--", lw=1.5)
    axes[0].axvline(z_two, color="#d93025", ls="--", lw=1.5)
    axes[0].set_title("Two-sided z test, alpha = 0.05")
    axes[0].set_xlabel("test statistic z")
    axes[0].set_ylabel("density under H0")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=9)
    axes[0].text(-3.55, 0.08, "2.5%", color="#d93025", fontsize=10)
    axes[0].text(2.82, 0.08, "2.5%", color="#d93025", fontsize=10)

    axes[1].plot(x, y, color="#202124", lw=2.2)
    axes[1].fill_between(x, y, where=(x >= z_one), color="#d93025", alpha=0.55, label="reject H0")
    axes[1].axvline(z_one, color="#d93025", ls="--", lw=1.5)
    axes[1].set_title("Right-sided z test, alpha = 0.05")
    axes[1].set_xlabel("test statistic z")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=9)
    axes[1].text(2.35, 0.08, "5%", color="#d93025", fontsize=10)

    fig.suptitle("Rejection regions are chosen before looking at the data", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch09_rejection_regions.png", bbox_inches="tight")
    plt.close(fig)


def fig_type_errors_power() -> None:
    """Plot type I error, type II error and power for a one-sided test."""
    x = np.linspace(-4.0, 5.5, 1000)
    mu0 = 0.0
    mu1 = 1.4
    threshold = 1.645
    y0 = _normal_pdf(x, mu0, 1.0)
    y1 = _normal_pdf(x, mu1, 1.0)
    alpha = 1.0 - _normal_cdf_scalar(threshold, mu0, 1.0)
    beta = _normal_cdf_scalar(threshold, mu1, 1.0)
    power = 1.0 - beta

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.plot(x, y0, color="#1a73e8", lw=2.4, label="H0 distribution")
    ax.plot(x, y1, color="#34a853", lw=2.4, label="H1 distribution")
    ax.axvline(threshold, color="#202124", ls="--", lw=1.7, label="critical value")
    ax.fill_between(x, y0, where=(x >= threshold), color="#d93025", alpha=0.45, label=f"type I error alpha={alpha:.3f}")
    ax.fill_between(x, y1, where=(x < threshold), color="#f9ab00", alpha=0.45, label=f"type II error beta={beta:.3f}")
    ax.fill_between(x, y1, where=(x >= threshold), color="#34a853", alpha=0.22, label=f"power={power:.3f}")
    ax.set_title("Type I error, type II error and power")
    ax.set_xlabel("test statistic")
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(-3.5, 0.31, "control alpha\nbefore the test", color="#1a73e8", fontsize=10)
    ax.text(2.55, 0.22, "larger effect or larger n\nusually increases power", color="#34a853", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch09_type_errors_power.png", bbox_inches="tight")
    plt.close(fig)


def fig_p_value_tail() -> None:
    """Plot two-sided p-value as tail area beyond observed statistic."""
    x = np.linspace(-4.0, 4.0, 900)
    y = _normal_pdf(x)
    z_obs = 2.2
    p_value = 2.0 * (1.0 - _normal_cdf_scalar(z_obs))

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(x, y, color="#202124", lw=2.3)
    ax.fill_between(x, y, where=(x <= -z_obs), color="#d93025", alpha=0.55, label="tail as or more extreme")
    ax.fill_between(x, y, where=(x >= z_obs), color="#d93025", alpha=0.55)
    ax.axvline(-z_obs, color="#d93025", ls="--", lw=1.4)
    ax.axvline(z_obs, color="#d93025", ls="--", lw=1.4)
    ax.axvline(0.0, color="#5f6368", lw=1.0)
    ax.set_title("Two-sided p-value under H0")
    ax.set_xlabel("test statistic z")
    ax.set_ylabel("density under H0")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(0.05, 0.33, f"observed |z| = {z_obs:.1f}\np-value = {p_value:.4f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch09_p_value_tail.png", bbox_inches="tight")
    plt.close(fig)


def fig_ab_test_flow() -> None:
    """Draw a practical A/B test workflow."""
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
                boxstyle="round,pad=0.016,rounding_size=0.02",
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
                mutation_scale=15,
                lw=2.0,
                color="#5f6368",
            )
        )

    top_y = 0.68
    items = [
        (0.11, "Question\nmetric, effect\nrisk"),
        (0.29, "Design\nrandomize\nsample size"),
        (0.47, "Run\navoid peeking\nlog quality"),
        (0.65, "Analyze\neffect size\np-value, CI"),
        (0.83, "Decide\nship, stop\nor iterate"),
    ]
    colors = [
        ("#e8f0fe", "#1a73e8"),
        ("#e6f4ea", "#34a853"),
        ("#fff7d6", "#f9ab00"),
        ("#fce8e6", "#d93025"),
        ("#f3e8fd", "#9334e6"),
    ]
    for (x, text), (fc, ec) in zip(items, colors):
        box(x, top_y, 0.14, 0.22, text, fc, ec)
    for i in range(len(items) - 1):
        arrow(items[i][0] + 0.075, top_y, items[i + 1][0] - 0.075, top_y)

    box(0.30, 0.25, 0.30, 0.18, "Before test\nwrite H0/H1, alpha,\nprimary metric", "#f8f9fa", "#9aa0a6")
    box(0.70, 0.25, 0.30, 0.18, "After test\nreport uncertainty,\nnot only significance", "#f8f9fa", "#9aa0a6")
    arrow(0.42, 0.35, 0.55, 0.56)
    arrow(0.58, 0.56, 0.67, 0.35)
    ax.text(
        0.50,
        0.08,
        "Good experimental design protects the meaning of the test; formulas cannot repair biased data.",
        ha="center",
        fontsize=11.5,
    )
    ax.set_title("A/B testing is a full workflow, not just a p-value", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch09_ab_test_flow.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 09 figures."""
    fig_rejection_regions()
    fig_type_errors_power()
    fig_p_value_tail()
    fig_ab_test_flow()


if __name__ == "__main__":
    main()
