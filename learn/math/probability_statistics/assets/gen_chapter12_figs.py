"""Generate figures for chapter 12 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter12_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch12_*.png
"""

from math import gamma, lgamma
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


def _beta_pdf(p: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Return Beta(alpha, beta) density on p in (0, 1)."""
    coef = gamma(alpha + beta) / (gamma(alpha) * gamma(beta))
    return coef * (p ** (alpha - 1.0)) * ((1.0 - p) ** (beta - 1.0))


def _binom_coeff_log(n: int, k: int) -> float:
    """Return log binomial coefficient."""
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _beta_binomial_pmf(k: np.ndarray, m: int, alpha: float, beta: float) -> np.ndarray:
    """Return Beta-Binomial(m, alpha, beta) PMF for integer array k."""
    values = []
    base = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    for value in k:
        log_p = (
            _binom_coeff_log(m, int(value))
            + lgamma(int(value) + alpha)
            + lgamma(m - int(value) + beta)
            - lgamma(m + alpha + beta)
            - base
        )
        values.append(np.exp(log_p))
    return np.array(values, dtype=float)


def fig_prior_to_posterior_update() -> None:
    """Plot prior, scaled likelihood and posterior for a Bernoulli rate."""
    p = np.linspace(0.001, 0.999, 600)
    alpha, beta = 2.0, 2.0
    n, successes = 18, 13
    failures = n - successes
    prior = _beta_pdf(p, alpha, beta)
    likelihood = (p**successes) * ((1.0 - p) ** failures)
    likelihood = likelihood / likelihood.max() * prior.max()
    posterior = _beta_pdf(p, alpha + successes, beta + failures)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(p, prior, color="#1a73e8", lw=2.3, label="prior Beta(2,2)")
    ax.plot(p, likelihood, color="#f9ab00", lw=2.3, ls="--", label="scaled likelihood, 13/18")
    ax.plot(p, posterior, color="#d93025", lw=2.6, label="posterior Beta(15,7)")
    ax.axvline(successes / n, color="#202124", lw=1.2, ls=":", label="sample rate")
    ax.set_title("Bayesian update: posterior is proportional to prior times likelihood")
    ax.set_xlabel("success probability p")
    ax.set_ylabel("density, likelihood scaled")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(0.06, posterior.max() * 0.72, "prior + data\n=> posterior", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch12_prior_to_posterior_update.png", bbox_inches="tight")
    plt.close(fig)


def fig_prior_influence() -> None:
    """Plot how different priors influence the posterior under the same data."""
    p = np.linspace(0.001, 0.999, 700)
    n, successes = 12, 9
    priors = [
        (1.0, 1.0, "#1a73e8", "weak prior Beta(1,1)"),
        (2.0, 8.0, "#d93025", "skeptical prior Beta(2,8)"),
        (20.0, 20.0, "#34a853", "strong prior Beta(20,20)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=False)
    for a, b, color, label in priors:
        axes[0].plot(p, _beta_pdf(p, a, b), color=color, lw=2.2, label=label)
        axes[1].plot(p, _beta_pdf(p, a + successes, b + n - successes), color=color, lw=2.2, label=label.replace("prior", "posterior"))
    for ax in axes:
        ax.axvline(successes / n, color="#202124", lw=1.2, ls=":", label="sample rate")
        ax.set_xlabel("success probability p")
        ax.grid(alpha=0.22)
    axes[0].set_title("Before data: different priors")
    axes[0].set_ylabel("density")
    axes[1].set_title("After same data: 9 successes in 12 trials")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Prior influence is strongest when the prior is strong or data is scarce", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch12_prior_influence.png", bbox_inches="tight")
    plt.close(fig)


def fig_conjugate_update_flow() -> None:
    """Draw a conjugate-prior update flow."""
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

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="->",
                mutation_scale=16,
                lw=2.0,
                color="#5f6368",
            )
        )
        if label:
            ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.045, label, ha="center", fontsize=9)

    box(0.16, 0.65, 0.22, 0.24, "Prior\nsame family as\nposterior", "#e8f0fe", "#1a73e8")
    box(0.42, 0.65, 0.22, 0.24, "Likelihood\ndata enters as\nsufficient counts", "#fff7d6", "#f9ab00")
    box(0.68, 0.65, 0.22, 0.24, "Posterior\nupdated parameters\nsame family", "#fce8e6", "#d93025")
    arrow(0.28, 0.65, 0.30, 0.65, "multiply")
    arrow(0.54, 0.65, 0.56, 0.65, "normalize")

    examples = [
        "Beta + Binomial -> Beta",
        "Gamma + Poisson -> Gamma",
        "Normal + Normal -> Normal",
    ]
    y_positions = [0.31, 0.22, 0.13]
    for text, y in zip(examples, y_positions):
        box(0.50, y, 0.42, 0.065, text, "#f8f9fa", "#9aa0a6")
    ax.text(0.50, 0.43, "Conjugacy is a computational convenience, not a requirement.", ha="center", fontsize=12)
    ax.set_title("Conjugate priors keep posterior updates in a familiar family", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch12_conjugate_update_flow.png", bbox_inches="tight")
    plt.close(fig)


def fig_posterior_predictive() -> None:
    """Plot posterior predictive distribution for future successes."""
    prior_alpha, prior_beta = 2.0, 2.0
    n, successes = 18, 13
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + n - successes
    future_m = 20
    k = np.arange(0, future_m + 1)
    posterior_pred = _beta_binomial_pmf(k, future_m, posterior_alpha, posterior_beta)
    plug_in_rate = posterior_alpha / (posterior_alpha + posterior_beta)
    plug_in = np.array(
        [
            np.exp(_binom_coeff_log(future_m, int(value)))
            * (plug_in_rate ** int(value))
            * ((1.0 - plug_in_rate) ** (future_m - int(value)))
            for value in k
        ]
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(k - 0.18, posterior_pred, width=0.36, color="#1a73e8", alpha=0.82, label="posterior predictive")
    ax.bar(k + 0.18, plug_in, width=0.36, color="#f9ab00", alpha=0.72, label="plug-in Binomial")
    ax.axvline(future_m * plug_in_rate, color="#202124", lw=1.4, ls="--", label="posterior mean prediction")
    ax.set_title("Posterior predictive averages over parameter uncertainty")
    ax.set_xlabel("future successes in 20 trials")
    ax.set_ylabel("predictive probability")
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=9)
    ax.text(1.1, posterior_pred.max() * 0.72, "Beta-Binomial\nis wider than\nplug-in Binomial", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch12_posterior_predictive.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 12 figures."""
    fig_prior_to_posterior_update()
    fig_prior_influence()
    fig_conjugate_update_flow()
    fig_posterior_predictive()


if __name__ == "__main__":
    main()
