"""Generate figures for chapter 17 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter17_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch17_*.png
"""

from math import pi, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

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


def _gmm_data() -> np.ndarray:
    """Return deterministic one-dimensional mixture data."""
    rng = np.random.default_rng(701)
    n = 360
    z = rng.binomial(1, 0.38, size=n)
    x = np.where(z == 0, rng.normal(-1.35, 0.45, size=n), rng.normal(1.55, 0.70, size=n))
    return x


def _run_em(x: np.ndarray, steps: int = 14) -> list[tuple[float, float, float, float, float]]:
    """Run EM for a two-component 1D Gaussian mixture.

    Returns tuples (pi1, mu1, sigma1, mu2, sigma2), where pi2 = 1 - pi1.
    """
    pi1 = 0.55
    mu1, mu2 = -0.35, 0.45
    sigma1, sigma2 = 1.25, 1.10
    history = []
    for _ in range(steps):
        comp1 = pi1 * _normal_pdf(x, mu1, sigma1)
        comp2 = (1.0 - pi1) * _normal_pdf(x, mu2, sigma2)
        r1 = comp1 / (comp1 + comp2 + 1e-12)
        n1 = r1.sum()
        n2 = x.size - n1
        pi1 = n1 / x.size
        mu1 = float((r1 * x).sum() / n1)
        mu2 = float(((1.0 - r1) * x).sum() / n2)
        sigma1 = float(np.sqrt((r1 * (x - mu1) ** 2).sum() / n1))
        sigma2 = float(np.sqrt(((1.0 - r1) * (x - mu2) ** 2).sum() / n2))
        history.append((float(pi1), mu1, sigma1, mu2, sigma2))
    return history


def fig_generative_discriminative_flow() -> None:
    """Draw generative vs discriminative modeling flow."""
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
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
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=16, lw=2.0, color="#5f6368"))

    box(0.18, 0.66, 0.25, 0.18, "Generative model\nlearn p(x,y)\nor p(x|y)p(y)", "#e8f0fe", "#1a73e8")
    box(0.50, 0.66, 0.22, 0.18, "Bayes rule\np(y|x)", "#fff7d6", "#f9ab00")
    box(0.82, 0.66, 0.25, 0.18, "Classification\npredict y", "#e6f4ea", "#34a853")
    arrow(0.31, 0.66, 0.39, 0.66)
    arrow(0.61, 0.66, 0.69, 0.66)

    box(0.18, 0.30, 0.25, 0.18, "Discriminative model\nlearn p(y|x)\nor decision score", "#fce8e6", "#d93025")
    box(0.50, 0.30, 0.22, 0.18, "Direct boundary\nor probability", "#f3e8fd", "#9334e6")
    box(0.82, 0.30, 0.25, 0.18, "Classification\npredict y", "#e6f4ea", "#34a853")
    arrow(0.31, 0.30, 0.39, 0.30)
    arrow(0.61, 0.30, 0.69, 0.30)
    ax.text(0.50, 0.08, "Generative models describe how data may be produced; discriminative models focus on prediction.", ha="center", fontsize=11.5)
    ax.set_title("Generative and discriminative models answer different questions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch17_generative_discriminative_flow.png", bbox_inches="tight")
    plt.close(fig)


def fig_gmm_mixture_density() -> None:
    """Plot Gaussian mixture data, components and total density."""
    x = _gmm_data()
    grid = np.linspace(-3.5, 4.0, 700)
    weights = [0.62, 0.38]
    comp1 = weights[0] * _normal_pdf(grid, -1.35, 0.45)
    comp2 = weights[1] * _normal_pdf(grid, 1.55, 0.70)
    total = comp1 + comp2

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.hist(x, bins=42, density=True, color="#9aa0a6", alpha=0.45, edgecolor="white", label="observed data")
    ax.plot(grid, comp1, color="#1a73e8", lw=2.3, label="component 1")
    ax.plot(grid, comp2, color="#d93025", lw=2.3, label="component 2")
    ax.plot(grid, total, color="#202124", lw=2.8, label="mixture density")
    ax.set_title("Gaussian mixture model: observed data comes from hidden components")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch17_gmm_mixture_density.png", bbox_inches="tight")
    plt.close(fig)


def fig_em_iteration() -> None:
    """Plot EM parameter trajectory for a two-component GMM."""
    x = _gmm_data()
    history = _run_em(x)
    steps = np.arange(1, len(history) + 1)
    arr = np.array(history)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    axes[0].plot(steps, arr[:, 1], "o-", color="#1a73e8", lw=2.2, label="mu1")
    axes[0].plot(steps, arr[:, 3], "s-", color="#d93025", lw=2.2, label="mu2")
    axes[0].axhline(-1.35, color="#1a73e8", ls="--", lw=1.1, alpha=0.65)
    axes[0].axhline(1.55, color="#d93025", ls="--", lw=1.1, alpha=0.65)
    axes[0].set_title("M-step updates component means")
    axes[0].set_xlabel("EM iteration")
    axes[0].set_ylabel("mean")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=9)

    axes[1].plot(steps, arr[:, 0], "o-", color="#34a853", lw=2.2, label="mixing weight pi1")
    axes[1].plot(steps, arr[:, 2], "s-", color="#1a73e8", lw=2.0, label="sigma1")
    axes[1].plot(steps, arr[:, 4], "^-", color="#d93025", lw=2.0, label="sigma2")
    axes[1].set_title("Weights and scales stabilize")
    axes[1].set_xlabel("EM iteration")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=9)
    fig.suptitle("EM alternates soft assignment (E-step) and parameter update (M-step)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch17_em_iteration.png", bbox_inches="tight")
    plt.close(fig)


def fig_hmm_state_transition() -> None:
    """Draw a small hidden Markov model."""
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    hidden = [(0.20, 0.67, "z1"), (0.50, 0.67, "z2"), (0.80, 0.67, "z3")]
    observed = [(0.20, 0.27, "x1"), (0.50, 0.27, "x2"), (0.80, 0.27, "x3")]
    for x, y, label in hidden:
        ax.add_patch(Circle((x, y), 0.075, facecolor="#e8f0fe", edgecolor="#1a73e8", lw=2.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=12)
    for x, y, label in observed:
        ax.add_patch(Circle((x, y), 0.075, facecolor="#fff7d6", edgecolor="#f9ab00", lw=2.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=12)

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str) -> None:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=15, lw=2.0, color="#5f6368"))
        ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.035, label, ha="center", fontsize=9)

    arrow(0.275, 0.67, 0.425, 0.67, "transition")
    arrow(0.575, 0.67, 0.725, 0.67, "transition")
    for x, _, _ in hidden:
        arrow(x, 0.59, x, 0.35, "emission")
    ax.text(0.50, 0.08, "HMM assumes hidden states evolve Markovianly and each observation is emitted from the current state.", ha="center", fontsize=11)
    ax.set_title("Hidden Markov model: latent states generate observed sequence", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch17_hmm_state_transition.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 17 figures."""
    fig_generative_discriminative_flow()
    fig_gmm_mixture_density()
    fig_em_iteration()
    fig_hmm_state_transition()


if __name__ == "__main__":
    main()
