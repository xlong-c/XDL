"""Generate figures for chapter 13 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter13_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch13_*.png
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


def _target_density(x: np.ndarray) -> np.ndarray:
    """Return an unnormalized bimodal target density."""
    return 0.55 * _normal_pdf(x, -1.3, 0.55) + 0.45 * _normal_pdf(x, 1.25, 0.75)


def fig_mle_map_comparison() -> None:
    """Plot likelihood, prior and posterior to compare MLE and MAP."""
    theta = np.linspace(-1.0, 2.4, 700)
    data_mean = 1.1
    like_sigma = 0.34
    prior_mu = 0.0
    prior_sigma = 0.55
    likelihood = _normal_pdf(theta, data_mean, like_sigma)
    prior = _normal_pdf(theta, prior_mu, prior_sigma)
    posterior_unnorm = likelihood * prior
    posterior = posterior_unnorm / np.trapezoid(posterior_unnorm, theta)
    mle = theta[np.argmax(likelihood)]
    map_est = theta[np.argmax(posterior)]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(theta, likelihood / likelihood.max(), color="#1a73e8", lw=2.4, label="scaled likelihood")
    ax.plot(theta, prior / prior.max(), color="#34a853", lw=2.4, label="scaled prior")
    ax.plot(theta, posterior / posterior.max(), color="#d93025", lw=2.7, label="scaled posterior")
    ax.axvline(mle, color="#1a73e8", ls="--", lw=1.5, label=f"MLE={mle:.2f}")
    ax.axvline(map_est, color="#d93025", ls="--", lw=1.5, label=f"MAP={map_est:.2f}")
    ax.set_title("MAP combines likelihood with prior information")
    ax.set_xlabel("parameter theta")
    ax.set_ylabel("scaled density")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    ax.text(-0.85, 0.78, "prior pulls posterior mode\naway from MLE", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch13_mle_map_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_mcmc_trace_histogram() -> None:
    """Run a simple random-walk Metropolis sampler and plot trace plus histogram."""
    rng = np.random.default_rng(313)
    n_steps = 5000
    proposal_sd = 0.85
    chain = np.empty(n_steps)
    chain[0] = -2.5
    accepted = 0

    def log_target(value: float) -> float:
        density = float(_target_density(np.array([value]))[0])
        return np.log(max(density, 1e-300))

    current_log = log_target(float(chain[0]))
    for i in range(1, n_steps):
        proposal = chain[i - 1] + rng.normal(0.0, proposal_sd)
        proposal_log = log_target(float(proposal))
        if np.log(rng.uniform()) < proposal_log - current_log:
            chain[i] = proposal
            current_log = proposal_log
            accepted += 1
        else:
            chain[i] = chain[i - 1]
    accept_rate = accepted / (n_steps - 1)

    grid = np.linspace(-4.0, 4.0, 700)
    target = _target_density(grid)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
    axes[0].plot(chain, color="#1a73e8", lw=0.8, alpha=0.82)
    axes[0].set_title("MCMC trace")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("theta")
    axes[0].grid(alpha=0.22)
    axes[0].text(200, 3.0, f"accept rate={accept_rate:.2f}", fontsize=10)

    burn = 600
    axes[1].hist(chain[burn:], bins=55, density=True, color="#f9ab00", alpha=0.75, edgecolor="white", label="MCMC samples")
    axes[1].plot(grid, target, color="#202124", lw=2.4, label="target posterior")
    axes[1].set_title("Samples approximate the target distribution")
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.22)
    axes[1].legend(fontsize=9)

    fig.suptitle("MCMC targets a distribution by constructing a dependent sample path", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch13_mcmc_trace_histogram.png", bbox_inches="tight")
    plt.close(fig)


def fig_vi_approximation() -> None:
    """Plot target posterior and a simple unimodal variational approximation."""
    x = np.linspace(-4.0, 4.0, 900)
    target = _target_density(x)
    q_mean = -0.52
    q_sd = 1.15
    q = _normal_pdf(x, q_mean, q_sd)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(x, target, color="#202124", lw=2.5, label="target posterior p(theta|D)")
    ax.plot(x, q, color="#d93025", lw=2.4, ls="--", label="variational q(theta)")
    ax.fill_between(x, target, q, color="#1a73e8", alpha=0.16, label="approximation gap")
    ax.set_title("Variational inference approximates the posterior with a tractable family")
    ax.set_xlabel("theta")
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(1.3, 0.30, "simple q may miss\nmulti-modal structure", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch13_vi_approximation.png", bbox_inches="tight")
    plt.close(fig)


def fig_elbo_decomposition() -> None:
    """Draw a compact ELBO decomposition diagram."""
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left = 0.18
    bottom = 0.25
    total_w = 0.64
    h = 0.24
    elbo_w = 0.44
    kl_w = total_w - elbo_w
    ax.add_patch(plt.Rectangle((left, bottom), elbo_w, h, facecolor="#1a73e8", alpha=0.75, edgecolor="white"))
    ax.add_patch(plt.Rectangle((left + elbo_w, bottom), kl_w, h, facecolor="#d93025", alpha=0.70, edgecolor="white"))
    ax.text(left + elbo_w / 2.0, bottom + h / 2.0, "ELBO\noptimized lower bound", ha="center", va="center", color="white", fontsize=12)
    ax.text(left + elbo_w + kl_w / 2.0, bottom + h / 2.0, "KL gap\nq vs posterior", ha="center", va="center", color="white", fontsize=12)
    ax.annotate(
        "log evidence log p(D)",
        xy=(left + total_w / 2.0, bottom + h + 0.03),
        xytext=(left + total_w / 2.0, 0.78),
        arrowprops={"arrowstyle": "-[", "lw": 2.0, "color": "#202124"},
        ha="center",
        fontsize=13,
    )
    ax.text(0.50, 0.14, "log p(D) = ELBO(q) + KL(q(theta) || p(theta|D))", ha="center", fontsize=12)
    ax.text(0.50, 0.06, "Maximizing ELBO is equivalent to shrinking the KL gap when log p(D) is fixed.", ha="center", fontsize=10.5)
    ax.set_title("ELBO turns posterior approximation into an optimization problem", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch13_elbo_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 13 figures."""
    fig_mle_map_comparison()
    fig_mcmc_trace_histogram()
    fig_vi_approximation()
    fig_elbo_decomposition()


if __name__ == "__main__":
    main()
