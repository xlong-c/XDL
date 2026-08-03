"""Generate figures for chapter 14 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter14_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch14_*.png
"""

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


def _bayes_regression_data() -> tuple[np.ndarray, np.ndarray, float]:
    """Return small data for Bayesian linear regression."""
    rng = np.random.default_rng(401)
    x = np.linspace(-3.0, 3.0, 22)
    sigma = 0.55
    y = 0.8 + 1.25 * x + rng.normal(0.0, sigma, size=x.size)
    return x, y, sigma


def _bayes_linear_posterior(x: np.ndarray, y: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return posterior mean and covariance for beta under a Normal prior."""
    design = np.column_stack([np.ones_like(x), x])
    prior_var = 9.0
    prior_precision = np.eye(2) / prior_var
    precision = prior_precision + (design.T @ design) / (sigma**2)
    cov = np.linalg.inv(precision)
    mean = cov @ (design.T @ y) / (sigma**2)
    return mean, cov


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Return logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def fig_bayesian_regression_band() -> None:
    """Plot Bayesian linear regression mean and uncertainty bands."""
    x, y, sigma = _bayes_regression_data()
    beta_mean, beta_cov = _bayes_linear_posterior(x, y, sigma)
    grid = np.linspace(-4.2, 4.2, 300)
    design_grid = np.column_stack([np.ones_like(grid), grid])
    pred_mean = design_grid @ beta_mean
    epistemic_var = np.sum((design_grid @ beta_cov) * design_grid, axis=1)
    epistemic_sd = np.sqrt(epistemic_var)
    predictive_sd = np.sqrt(epistemic_var + sigma**2)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.scatter(x, y, color="#202124", s=38, alpha=0.82, label="observed data")
    ax.plot(grid, pred_mean, color="#1a73e8", lw=2.4, label="posterior mean")
    ax.fill_between(grid, pred_mean - 1.96 * epistemic_sd, pred_mean + 1.96 * epistemic_sd, color="#1a73e8", alpha=0.22, label="parameter uncertainty")
    ax.fill_between(grid, pred_mean - 1.96 * predictive_sd, pred_mean + 1.96 * predictive_sd, color="#f9ab00", alpha=0.20, label="predictive interval")
    ax.set_title("Bayesian linear regression separates parameter and predictive uncertainty")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(-4.0, 4.8, "predictive band = parameter uncertainty + observation noise", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch14_bayesian_regression_band.png", bbox_inches="tight")
    plt.close(fig)


def fig_hierarchical_partial_pooling() -> None:
    """Draw a hierarchical model and partial pooling intuition."""
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
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
                mutation_scale=15,
                lw=2.0,
                color="#5f6368",
            )
        )

    box(0.50, 0.80, 0.26, 0.16, "Population level\nmu, tau", "#e8f0fe", "#1a73e8")
    group_x = [0.22, 0.50, 0.78]
    group_names = ["group A\ntheta_A", "group B\ntheta_B", "group C\ntheta_C"]
    for x, name in zip(group_x, group_names):
        box(x, 0.52, 0.20, 0.15, name, "#e6f4ea", "#34a853")
        arrow(0.50, 0.72, x, 0.60)
        box(x, 0.25, 0.20, 0.15, "data\nwithin group", "#fff7d6", "#f9ab00")
        arrow(x, 0.44, x, 0.33)
    ax.text(0.50, 0.08, "Partial pooling shrinks noisy group estimates toward a shared population pattern.", ha="center", fontsize=12)
    ax.set_title("Hierarchical models share information across related groups", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch14_hierarchical_partial_pooling.png", bbox_inches="tight")
    plt.close(fig)


def fig_calibration_curve() -> None:
    """Plot calibration curve for calibrated and overconfident predictions."""
    rng = np.random.default_rng(409)
    n = 9000
    raw_score = rng.normal(0.0, 1.0, size=n)
    true_prob = _sigmoid(raw_score)
    y = rng.binomial(1, true_prob)
    pred_good = true_prob
    pred_bad = _sigmoid(1.75 * raw_score)
    bins = np.linspace(0.0, 1.0, 11)

    def curve(pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        centers = []
        observed = []
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (pred >= left) & (pred < right)
            if mask.sum() >= 30:
                centers.append(float(pred[mask].mean()))
                observed.append(float(y[mask].mean()))
        return np.array(centers), np.array(observed)

    good_x, good_y = curve(pred_good)
    bad_x, bad_y = curve(pred_bad)

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot([0, 1], [0, 1], color="#202124", ls="--", lw=1.6, label="perfect calibration")
    ax.plot(good_x, good_y, "o-", color="#1a73e8", lw=2.2, label="calibrated model")
    ax.plot(bad_x, bad_y, "s-", color="#d93025", lw=2.2, label="overconfident model")
    ax.set_title("Calibration checks whether predicted probabilities mean what they say")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch14_calibration_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig_uncertainty_decomposition() -> None:
    """Plot epistemic and aleatoric uncertainty across input space."""
    x, y, sigma = _bayes_regression_data()
    beta_mean, beta_cov = _bayes_linear_posterior(x, y, sigma)
    grid = np.linspace(-5.0, 5.0, 300)
    design_grid = np.column_stack([np.ones_like(grid), grid])
    epistemic_sd = np.sqrt(np.sum((design_grid @ beta_cov) * design_grid, axis=1))
    aleatoric_sd = np.full_like(grid, sigma)
    total_sd = np.sqrt(epistemic_sd**2 + aleatoric_sd**2)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(grid, epistemic_sd, color="#1a73e8", lw=2.3, label="epistemic: parameter uncertainty")
    ax.plot(grid, aleatoric_sd, color="#d93025", lw=2.3, label="aleatoric: observation noise")
    ax.plot(grid, total_sd, color="#202124", lw=2.3, ls="--", label="total predictive sd")
    ax.axvspan(x.min(), x.max(), color="#e8f0fe", alpha=0.55, label="training data range")
    ax.set_title("Epistemic uncertainty grows away from observed data")
    ax.set_xlabel("x")
    ax.set_ylabel("standard deviation")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch14_uncertainty_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 14 figures."""
    fig_bayesian_regression_band()
    fig_hierarchical_partial_pooling()
    fig_calibration_curve()
    fig_uncertainty_decomposition()


if __name__ == "__main__":
    main()
