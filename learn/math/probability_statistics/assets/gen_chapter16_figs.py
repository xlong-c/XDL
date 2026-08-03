"""Generate figures for chapter 16 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter16_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch16_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def fig_ridge_lasso_geometry() -> None:
    """Draw Ridge and Lasso constraint geometry."""
    b1 = np.linspace(-2.2, 2.8, 420)
    b2 = np.linspace(-2.2, 2.8, 420)
    xx, yy = np.meshgrid(b1, b2)
    center = np.array([1.65, 1.15])
    loss = ((xx - center[0]) ** 2) / 1.8 + ((yy - center[1]) ** 2) / 0.75

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharex=True, sharey=True)
    for ax, title in zip(axes, ["Ridge: L2 ball", "Lasso: L1 diamond"]):
        ax.contour(xx, yy, loss, levels=[0.35, 0.8, 1.4, 2.2, 3.3], colors="#9aa0a6", linewidths=1.2)
        ax.scatter([center[0]], [center[1]], color="#202124", s=45, label="unregularized estimate")
        ax.axhline(0.0, color="#dadce0", lw=1.0)
        ax.axvline(0.0, color="#dadce0", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("beta1")
        ax.grid(alpha=0.18)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("beta2")

    axes[0].add_patch(Circle((0.0, 0.0), 1.55, fill=False, edgecolor="#1a73e8", lw=2.4, label="constraint"))
    ridge_point = np.array([1.20, 0.98])
    axes[0].scatter([ridge_point[0]], [ridge_point[1]], color="#d93025", s=55, zorder=5, label="regularized solution")
    axes[0].text(-1.9, 1.75, "L2 shrinks coefficients\nsmoothly", fontsize=10)

    diamond = np.array([[0.0, 1.55], [1.55, 0.0], [0.0, -1.55], [-1.55, 0.0]])
    axes[1].add_patch(Polygon(diamond, closed=True, fill=False, edgecolor="#1a73e8", lw=2.4, label="constraint"))
    lasso_point = np.array([1.55, 0.0])
    axes[1].scatter([lasso_point[0]], [lasso_point[1]], color="#d93025", s=55, zorder=5, label="regularized solution")
    axes[1].text(-1.9, 1.75, "L1 corners encourage\nexact zeros", fontsize=10)

    for ax in axes:
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Ridge and Lasso differ because their constraint geometry differs", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch16_ridge_lasso_geometry.png", bbox_inches="tight")
    plt.close(fig)


def fig_svm_margin() -> None:
    """Draw a separating hyperplane and SVM margins."""
    rng = np.random.default_rng(601)
    class_neg = rng.normal(loc=(-1.25, -0.70), scale=(0.38, 0.34), size=(26, 2))
    class_pos = rng.normal(loc=(1.15, 0.85), scale=(0.38, 0.34), size=(26, 2))
    w = np.array([0.85, -1.15])
    b = 0.05
    xs = np.linspace(-2.6, 2.6, 300)

    def line(level: float) -> np.ndarray:
        return -(w[0] * xs + b - level) / w[1]

    all_points = np.vstack([class_neg, class_pos])
    signed_distance = np.abs(all_points @ w + b) / np.linalg.norm(w)
    support_idx = np.argsort(signed_distance)[:6]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.scatter(class_neg[:, 0], class_neg[:, 1], color="#1a73e8", s=42, edgecolors="white", linewidth=0.5, label="class -1")
    ax.scatter(class_pos[:, 0], class_pos[:, 1], color="#d93025", s=42, edgecolors="white", linewidth=0.5, label="class +1")
    ax.scatter(all_points[support_idx, 0], all_points[support_idx, 1], facecolors="none", edgecolors="#202124", s=120, linewidth=1.8, label="support vectors")
    ax.plot(xs, line(0.0), color="#202124", lw=2.4, label="decision boundary")
    ax.plot(xs, line(1.0), color="#5f6368", lw=1.7, ls="--", label="margin")
    ax.plot(xs, line(-1.0), color="#5f6368", lw=1.7, ls="--")
    ax.set_title("SVM chooses a separating boundary with large margin")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-2.4, 2.4)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    ax.text(-2.45, 1.85, "only points near the margin\ncontrol the boundary", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch16_svm_margin.png", bbox_inches="tight")
    plt.close(fig)


def fig_decision_tree_splits() -> None:
    """Draw axis-aligned decision-tree splits."""
    rng = np.random.default_rng(607)
    x = rng.uniform(-2.5, 2.5, size=(180, 2))
    y = ((x[:, 0] > -0.25) & (x[:, 1] > -0.75) | (x[:, 0] > 1.15)).astype(int)

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.add_patch(Rectangle((-2.5, -2.5), 2.25, 5.0, facecolor="#e8f0fe", alpha=0.45, edgecolor="none"))
    ax.add_patch(Rectangle((-0.25, -2.5), 1.40, 1.75, facecolor="#e8f0fe", alpha=0.45, edgecolor="none"))
    ax.add_patch(Rectangle((-0.25, -0.75), 1.40, 3.25, facecolor="#fce8e6", alpha=0.45, edgecolor="none"))
    ax.add_patch(Rectangle((1.15, -2.5), 1.35, 5.0, facecolor="#fce8e6", alpha=0.45, edgecolor="none"))
    ax.axvline(-0.25, color="#202124", lw=2.0)
    ax.axhline(-0.75, xmin=( -0.25 + 2.5) / 5.0, xmax=(1.15 + 2.5) / 5.0, color="#202124", lw=2.0)
    ax.axvline(1.15, color="#202124", lw=2.0)
    ax.scatter(x[y == 0, 0], x[y == 0, 1], color="#1a73e8", s=26, alpha=0.82, edgecolors="white", linewidth=0.3, label="class 0")
    ax.scatter(x[y == 1, 0], x[y == 1, 1], color="#d93025", s=26, alpha=0.82, edgecolors="white", linewidth=0.3, label="class 1")
    ax.set_title("Decision trees split feature space with axis-aligned rules")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.grid(alpha=0.18)
    ax.legend(fontsize=9)
    ax.text(-2.35, 2.15, "if x1 <= -0.25\nthen class 0", fontsize=9)
    ax.text(1.25, -2.20, "if x1 > 1.15\nthen class 1", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch16_decision_tree_splits.png", bbox_inches="tight")
    plt.close(fig)


def fig_decision_boundaries_comparison() -> None:
    """Compare stylized decision boundaries for different model classes."""
    x1 = np.linspace(-3.0, 3.0, 260)
    x2 = np.linspace(-3.0, 3.0, 260)
    xx, yy = np.meshgrid(x1, x2)
    linear_score = yy - 0.55 * xx + 0.2
    kernel_score = 1.25 - ((xx - 0.2) ** 2 + (yy + 0.15) ** 2) + 0.18 * xx
    tree_score = np.where(xx < -0.45, -1.0, np.where(yy > 0.65, 1.0, np.where(xx > 1.15, 1.0, -1.0)))

    scores = [linear_score, kernel_score, tree_score]
    titles = ["Linear model", "Kernel-style nonlinear boundary", "Tree-style step boundary"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), sharex=True, sharey=True)
    for ax, score, title in zip(axes, scores, titles):
        ax.contourf(xx, yy, score > 0, levels=[-0.5, 0.5, 1.5], colors=["#e8f0fe", "#fce8e6"], alpha=0.95)
        ax.contour(xx, yy, score, levels=[0.0], colors="#202124", linewidths=2.0)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.grid(alpha=0.13)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("x2")
    fig.suptitle("Different hypothesis spaces imply different decision boundary shapes", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch16_decision_boundaries_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 16 figures."""
    fig_ridge_lasso_geometry()
    fig_svm_margin()
    fig_decision_tree_splits()
    fig_decision_boundaries_comparison()


if __name__ == "__main__":
    main()
