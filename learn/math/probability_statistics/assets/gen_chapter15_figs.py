"""Generate figures for chapter 15 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter15_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch15_*.png
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


def _true_function(x: np.ndarray) -> np.ndarray:
    """Return a smooth target function."""
    return np.sin(1.5 * x) + 0.25 * x


def _poly_predict(x_train: np.ndarray, y_train: np.ndarray, degree: int, x_grid: np.ndarray) -> np.ndarray:
    """Fit polynomial regression and predict on x_grid."""
    coefs = np.polyfit(x_train, y_train, deg=degree)
    return np.polyval(coefs, x_grid)


def fig_train_test_error_curve() -> None:
    """Plot train and test error as model complexity changes."""
    complexity = np.arange(1, 21)
    train_error = 0.78 * np.exp(-0.16 * complexity) + 0.035
    test_error = 0.18 + 0.62 * np.exp(-0.28 * complexity) + 0.012 * np.maximum(complexity - 8, 0) ** 1.45
    best_idx = int(np.argmin(test_error))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(complexity, train_error, "o-", color="#1a73e8", lw=2.3, label="training error")
    ax.plot(complexity, test_error, "s-", color="#d93025", lw=2.3, label="test error")
    ax.axvline(complexity[best_idx], color="#202124", ls="--", lw=1.5, label="best test complexity")
    ax.set_title("Training error keeps falling, but test error can rise")
    ax.set_xlabel("model complexity")
    ax.set_ylabel("error")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(2.0, 0.52, "underfit", fontsize=10)
    ax.text(14.0, 0.50, "overfit", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch15_train_test_error_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig_bias_variance_demo() -> None:
    """Show underfit, good fit and high-variance fits."""
    rng = np.random.default_rng(503)
    x_grid = np.linspace(-3.0, 3.0, 400)
    y_true = _true_function(x_grid)
    degrees = [1, 4, 13]
    titles = ["high bias\nunderfit", "balanced", "high variance\noverfit"]
    colors = ["#9aa0a6", "#1a73e8", "#d93025"]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), sharey=True)
    for ax, degree, title, color in zip(axes, degrees, titles, colors):
        for _ in range(28):
            x_train = np.sort(rng.uniform(-3.0, 3.0, size=22))
            y_train = _true_function(x_train) + rng.normal(0.0, 0.28, size=x_train.size)
            y_pred = _poly_predict(x_train, y_train, degree, x_grid)
            ax.plot(x_grid, y_pred, color=color, alpha=0.22, lw=1.0)
        ax.plot(x_grid, y_true, color="#202124", lw=2.4, label="true function")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.grid(alpha=0.22)
        ax.set_ylim(-2.2, 2.2)
        if ax is axes[0]:
            ax.set_ylabel("prediction")
            ax.legend(fontsize=8)
    fig.suptitle("Bias-variance tradeoff: repeated training samples lead to different fitted functions", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch15_bias_variance_demo.png", bbox_inches="tight")
    plt.close(fig)


def fig_learning_curve() -> None:
    """Plot a conceptual learning curve as sample size increases."""
    n = np.array([20, 35, 55, 80, 120, 180, 260, 380, 560, 800, 1200])
    train = 0.17 + 0.42 / np.sqrt(n / 20.0)
    validation = 0.22 + 0.70 / np.sqrt(n / 20.0) + 0.025 * np.exp(-n / 250.0)
    gap = validation - train

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(n, train, "o-", color="#1a73e8", lw=2.3, label="training error")
    ax.plot(n, validation, "s-", color="#d93025", lw=2.3, label="validation error")
    ax.fill_between(n, train, validation, color="#f9ab00", alpha=0.20, label="generalization gap")
    ax.set_xscale("log")
    ax.set_title("Learning curves show whether more data is likely to help")
    ax.set_xlabel("training set size, log scale")
    ax.set_ylabel("error")
    ax.grid(alpha=0.22, which="both")
    ax.legend(fontsize=9)
    ax.text(45, 0.55, "large gap:\nmore data may help", fontsize=10)
    ax.text(520, 0.29, "curves flatten:\nmodel or features may limit", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch15_learning_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig_complexity_generalization_risk() -> None:
    """Plot approximation, estimation and irreducible noise components."""
    complexity = np.linspace(0.5, 12.0, 300)
    approximation = 0.75 * np.exp(-0.35 * complexity) + 0.04
    estimation = 0.018 * complexity**1.55
    noise = np.full_like(complexity, 0.08)
    total = approximation + estimation + noise
    best_x = float(complexity[np.argmin(total)])
    best_y = float(total.min())

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(complexity, approximation, color="#1a73e8", lw=2.3, label="approximation error")
    ax.plot(complexity, estimation, color="#d93025", lw=2.3, label="estimation error")
    ax.plot(complexity, noise, color="#9aa0a6", lw=2.0, ls="--", label="irreducible noise")
    ax.plot(complexity, total, color="#202124", lw=2.6, label="generalization risk")
    ax.scatter([best_x], [best_y], color="#f9ab00", s=70, zorder=5, label="best tradeoff")
    ax.set_title("Generalization risk balances approximation and estimation errors")
    ax.set_xlabel("model complexity")
    ax.set_ylabel("risk")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch15_complexity_generalization_risk.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 15 figures."""
    fig_train_test_error_curve()
    fig_bias_variance_demo()
    fig_learning_curve()
    fig_complexity_generalization_risk()


if __name__ == "__main__":
    main()
