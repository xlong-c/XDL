"""Generate figures for chapter 10 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter10_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch10_*.png
"""

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


def _linear_data() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic simulated linear-regression data."""
    rng = np.random.default_rng(101)
    x = np.linspace(0.0, 10.0, 70)
    noise = rng.normal(0.0, 1.05, size=x.size)
    y = 1.5 + 0.78 * x + noise
    return x, y


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit simple OLS and return intercept, slope, fitted values and residuals."""
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residuals = y - fitted
    return float(beta[0]), float(beta[1]), fitted, residuals


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Return logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def fig_linear_fit() -> None:
    """Plot scatter data and fitted regression line."""
    x, y = _linear_data()
    intercept, slope, fitted, _ = _ols_fit(x, y)
    x_grid = np.linspace(x.min(), x.max(), 300)
    y_grid = intercept + slope * x_grid

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.scatter(x, y, s=38, color="#1a73e8", alpha=0.82, edgecolors="white", linewidth=0.5, label="observed data")
    ax.plot(x_grid, y_grid, color="#d93025", lw=2.5, label=f"OLS fit: y={intercept:.2f}+{slope:.2f}x")
    ax.set_title("Simple linear regression fits a conditional mean line")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(0.4, 9.8, "model: Y = beta0 + beta1 x + error", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch10_linear_fit.png", bbox_inches="tight")
    plt.close(fig)


def fig_residual_diagnostics() -> None:
    """Plot residuals versus fitted values and residual histogram."""
    x, y = _linear_data()
    _, _, fitted, residuals = _ols_fit(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    axes[0].scatter(fitted, residuals, s=34, color="#1a73e8", alpha=0.85, edgecolors="white", linewidth=0.5)
    axes[0].axhline(0.0, color="#202124", lw=1.6, ls="--")
    axes[0].set_title("Residuals versus fitted values")
    axes[0].set_xlabel("fitted value")
    axes[0].set_ylabel("residual")
    axes[0].grid(alpha=0.22)
    axes[0].text(fitted.min() + 0.2, residuals.max() - 0.35, "look for pattern,\ncurvature,\nchanging spread", fontsize=9)

    axes[1].hist(residuals, bins=16, density=True, color="#34a853", alpha=0.78, edgecolor="white")
    axes[1].axvline(0.0, color="#202124", lw=1.4, ls="--")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("residual")
    axes[1].set_ylabel("density")
    axes[1].grid(alpha=0.22)
    axes[1].text(0.10, 0.90, f"mean={residuals.mean():.2f}\nstd={residuals.std(ddof=2):.2f}", transform=axes[1].transAxes, fontsize=9)

    fig.suptitle("Residual diagnostics check whether the model assumptions look plausible", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch10_residual_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def fig_variance_decomposition() -> None:
    """Visualize SST = SSR + SSE for one fitted regression example."""
    x, y = _linear_data()
    _, _, fitted, residuals = _ols_fit(x, y)
    ybar = float(y.mean())
    sst = float(np.sum((y - ybar) ** 2))
    ssr = float(np.sum((fitted - ybar) ** 2))
    sse = float(np.sum(residuals**2))
    r2 = 1.0 - sse / sst

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    idx = np.linspace(0, len(x) - 1, 11, dtype=int)
    axes[0].scatter(x, y, color="#1a73e8", s=26, alpha=0.65, label="observed y")
    axes[0].plot(x, fitted, color="#d93025", lw=2.2, label="fitted mean")
    axes[0].axhline(ybar, color="#202124", ls="--", lw=1.5, label="sample mean")
    for i in idx:
        axes[0].vlines(x[i], ybar, fitted[i], color="#34a853", lw=1.8, alpha=0.85)
        axes[0].vlines(x[i], fitted[i], y[i], color="#f9ab00", lw=1.8, alpha=0.85)
    axes[0].set_title("Explained part and residual part")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.22)
    axes[0].legend(fontsize=8)

    labels = ["SST\ntotal", "SSR\nexplained", "SSE\nresidual"]
    values = [sst, ssr, sse]
    colors = ["#9aa0a6", "#34a853", "#f9ab00"]
    axes[1].bar(labels, values, color=colors, edgecolor="white")
    axes[1].set_title("Variance decomposition")
    axes[1].set_ylabel("sum of squares")
    axes[1].grid(alpha=0.22, axis="y")
    axes[1].text(1.05, max(values) * 0.82, f"SST = SSR + SSE\nR^2 = {r2:.3f}", fontsize=11)

    fig.suptitle("ANOVA view of linear regression: decompose total variation", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch10_variance_decomposition.png", bbox_inches="tight")
    plt.close(fig)


def fig_logistic_curve() -> None:
    """Plot logistic regression curves and binary observations."""
    rng = np.random.default_rng(109)
    x = np.linspace(-4.0, 4.0, 120)
    true_p = _sigmoid(-0.4 + 1.25 * x)
    y = rng.binomial(1, true_p)
    x_grid = np.linspace(-5.0, 5.0, 600)
    curves = [
        (-0.4, 0.8, "#34a853", "slope=0.8"),
        (-0.4, 1.25, "#1a73e8", "slope=1.25"),
        (-0.4, 2.0, "#d93025", "slope=2.0"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    jitter = rng.normal(0.0, 0.025, size=y.size)
    ax.scatter(x, y + jitter, s=24, color="#202124", alpha=0.45, label="binary data")
    for intercept, slope, color, label in curves:
        ax.plot(x_grid, _sigmoid(intercept + slope * x_grid), color=color, lw=2.4, label=label)
    ax.axhline(0.5, color="#5f6368", lw=1.0, ls="--")
    ax.set_title("Logistic regression models probability, not a continuous response")
    ax.set_xlabel("linear predictor x")
    ax.set_ylabel("P(Y=1 | x)")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(-4.7, 0.82, "logit(p)=beta0+beta1 x", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch10_logistic_curve.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 10 figures."""
    fig_linear_fit()
    fig_residual_diagnostics()
    fig_variance_decomposition()
    fig_logistic_curve()


if __name__ == "__main__":
    main()
