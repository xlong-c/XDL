"""Generate figures for chapter 03 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter03_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch03_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import comb, erf, exp, factorial, pi, sqrt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def fig_random_variable_mapping() -> None:
    """Show that a random variable maps outcomes to numbers."""
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str, fs: int = 11) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor=fc,
                edgecolor=ec,
                lw=2.0,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#202124")

    sample_nodes = {
        "HH": (0.14, 0.78),
        "HT": (0.14, 0.58),
        "TH": (0.14, 0.38),
        "TT": (0.14, 0.18),
    }
    value_nodes = {
        "2": (0.52, 0.78),
        "1a": (0.52, 0.48),
        "0": (0.52, 0.18),
    }

    ax.text(0.14, 0.93, "sample outcomes", ha="center", fontsize=13, weight="bold")
    ax.text(0.52, 0.93, "X = number of heads", ha="center", fontsize=13, weight="bold")
    ax.text(0.83, 0.93, "distribution of X", ha="center", fontsize=13, weight="bold")

    for label, (x, y) in sample_nodes.items():
        box(x, y, 0.12, 0.1, label, "#e8f0fe", "#1a73e8", fs=13)
    box(0.52, 0.78, 0.12, 0.1, "2", "#fce8e6", "#d93025", fs=14)
    box(0.52, 0.48, 0.12, 0.1, "1", "#fce8e6", "#d93025", fs=14)
    box(0.52, 0.18, 0.12, 0.1, "0", "#fce8e6", "#d93025", fs=14)

    arrows = [("HH", "2"), ("HT", "1a"), ("TH", "1a"), ("TT", "0")]
    for s, t in arrows:
        x0, y0 = sample_nodes[s]
        x1, y1 = value_nodes[t]
        ax.annotate(
            "",
            xy=(x1 - 0.07, y1),
            xytext=(x0 + 0.07, y0),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="#5f6368"),
        )

    ks = np.array([0, 1, 2])
    ps = np.array([0.25, 0.5, 0.25])
    xleft = 0.69
    width = 0.08
    heights = 0.5 * ps / ps.max()
    colors = ["#34a853", "#1a73e8", "#f9ab00"]
    for i, (k, p, h, c) in enumerate(zip(ks, ps, heights, colors)):
        x = xleft + i * 0.1
        ax.add_patch(FancyBboxPatch((x, 0.12), width, h, boxstyle="square,pad=0.0", facecolor=c, edgecolor="#202124"))
        ax.text(x + width / 2, 0.08, f"{k}", ha="center", fontsize=11)
        ax.text(x + width / 2, 0.14 + h, f"{p:.2f}", ha="center", fontsize=10)
    ax.text(0.73, 0.68, "PMF", fontsize=12, weight="bold")
    ax.text(0.82, 0.03, "k", fontsize=11)
    ax.text(0.66, 0.38, "p_X(k)", fontsize=11, rotation=90)
    ax.text(0.5, 0.02, "A random variable compresses outcomes into numerical values, then the distribution records how often each value appears.", ha="center", fontsize=10.5)

    fig.tight_layout()
    fig.savefig(OUT / "ps_ch03_random_variable_mapping.png", bbox_inches="tight")
    plt.close(fig)


def fig_cdf_pmf_pdf_alignment() -> None:
    """Align PMF/PDF with their CDFs."""
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))

    # Discrete PMF and CDF: Binomial(4, 0.5)
    n = 4
    p = 0.5
    k = np.arange(0, n + 1)
    pmf = np.array([comb(n, int(i)) * (p**i) * ((1 - p) ** (n - i)) for i in k], dtype=float)
    cdf = np.cumsum(pmf)

    ax = axes[0, 0]
    ax.bar(k, pmf, color="#8ab4f8", edgecolor="#1a73e8", width=0.7)
    ax.set_title("Discrete PMF: Binomial(4, 0.5)")
    ax.set_xlabel("k")
    ax.set_ylabel("P(X = k)")
    ax.set_xticks(k)
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    xstep = np.r_[-0.5, k, 4.5]
    ystep = np.r_[0.0, cdf, 1.0]
    ax.step(xstep, ystep, where="post", color="#1a73e8", lw=2.2)
    ax.scatter(k, cdf, color="#d93025", zorder=5)
    ax.set_title("Discrete CDF: F(k) = P(X <= k)")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(k)
    ax.grid(alpha=0.2)

    # Continuous PDF and CDF: Exponential(1)
    x = np.linspace(0, 5, 400)
    lam = 1.0
    pdf = lam * np.exp(-lam * x)
    cdf_cont = 1 - np.exp(-lam * x)
    x0 = 1.2

    ax = axes[1, 0]
    mask = x <= x0
    ax.plot(x, pdf, color="#1a73e8", lw=2.2)
    ax.fill_between(x[mask], 0, pdf[mask], color="#8ab4f8", alpha=0.75)
    ax.axvline(x0, color="#d93025", ls="--", lw=1.8)
    ax.text(2.15, 0.62, "shaded area = P(X <= 1.2)", fontsize=10)
    ax.set_title("Continuous PDF: Exponential(1)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.plot(x, cdf_cont, color="#34a853", lw=2.2)
    ax.axvline(x0, color="#d93025", ls="--", lw=1.8)
    ax.axhline(1 - exp(-x0), color="#d93025", ls=":", lw=1.6)
    ax.text(2.4, 0.38, "F(1.2) = 1 - exp(-1.2)", fontsize=10)
    ax.set_title("Continuous CDF: F(x) = integral from -inf to x")
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.2)

    fig.suptitle("PMF/PDF describe local behavior; CDF accumulates probability", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch03_cdf_pmf_pdf_alignment.png", bbox_inches="tight")
    plt.close(fig)


def fig_binomial_poisson_approx() -> None:
    """Compare Binomial and its Poisson approximation."""
    n = 50
    p = 0.08
    lam = n * p
    k = np.arange(0, 15)
    binom = np.array([comb(n, int(i)) * (p**i) * ((1 - p) ** (n - i)) for i in k], dtype=float)
    pois = np.array([exp(-lam) * (lam**i) / factorial(int(i)) for i in k], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(k - 0.18, binom, width=0.36, color="#8ab4f8", edgecolor="#1a73e8", label="Binomial(50, 0.08)")
    ax.bar(k + 0.18, pois, width=0.36, color="#fce8e6", edgecolor="#d93025", label="Poisson(lambda = 4)")
    ax.set_title("Poisson approximates Binomial when n is large and p is small")
    ax.set_xlabel("k")
    ax.set_ylabel("probability")
    ax.set_xticks(k)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    ax.text(8.1, max(binom.max(), pois.max()) * 0.82, "here lambda = np = 4", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch03_binomial_poisson_approx.png", bbox_inches="tight")
    plt.close(fig)


def fig_normal_family() -> None:
    """Show how mu and sigma change the Normal density."""
    x = np.linspace(-6, 6, 700)

    def normal_pdf(xv: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return (1.0 / (sqrt(2 * pi) * sigma)) * np.exp(-((xv - mu) ** 2) / (2 * sigma**2))

    curves = [
        (0.0, 1.0, "#1a73e8", "N(0, 1)"),
        (0.0, 2.0, "#34a853", "N(0, 4)"),
        (2.0, 1.0, "#d93025", "N(2, 1)"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for mu, sigma, color, label in curves:
        ax.plot(x, normal_pdf(x, mu, sigma), color=color, lw=2.4, label=label)
    ax.axvline(0, color="#9aa0a6", ls="--", lw=1.0)
    ax.axvline(2, color="#9aa0a6", ls=":", lw=1.0)
    ax.text(-2.6, 0.34, "larger sigma -> flatter and wider", fontsize=10, color="#34a853")
    ax.text(2.15, 0.24, "changing mu shifts the center", fontsize=10, color="#d93025")
    ax.set_title("The Normal family: mean sets location, variance sets spread")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch03_normal_family.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 03 figures."""
    fig_random_variable_mapping()
    fig_cdf_pmf_pdf_alignment()
    fig_binomial_poisson_approx()
    fig_normal_family()


if __name__ == "__main__":
    main()
