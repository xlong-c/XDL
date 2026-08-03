"""Generate figures for highmath/session1.html.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/assets/gen_highmath_session1_figs.py

Outputs:
    learn/math/assets/highmath_*.png
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


def fig_sequence_limit() -> None:
    """Plot a convergent sequence with an epsilon band."""
    n = np.arange(1, 16)
    a_n = 1.0 + 1.0 / n
    limit = 1.0
    eps = 0.2

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(limit - eps, limit + eps, color="#d2e3fc", alpha=0.8, label="epsilon band")
    ax.axhline(limit, color="#1a73e8", lw=2.0, label="limit L = 1")
    ax.plot(n, a_n, "o", color="#d93025", ms=6, label="a_n = 1 + 1/n")
    ax.vlines(n, limit, a_n, color="#f28b82", lw=1.2, alpha=0.7)
    ax.annotate(
        "after N, all terms stay in the band",
        xy=(6, a_n[5]),
        xytext=(8.2, 1.55),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0.9, 2.1)
    ax.set_xlabel("n")
    ax.set_ylabel("a_n")
    ax.set_title("Sequence limit: a_n approaches L")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "highmath_sequence_limit.png", bbox_inches="tight")
    plt.close(fig)


def fig_function_limit() -> None:
    """Plot a removable discontinuity to explain limit vs function value."""
    x_left = np.linspace(-1.6, 0.98, 200)
    x_right = np.linspace(1.02, 3.2, 200)
    y_left = x_left + 1.0
    y_right = x_right + 1.0

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x_left, y_left, color="#1a73e8", lw=2.2)
    ax.plot(x_right, y_right, color="#1a73e8", lw=2.2, label="y = (x^2 - 1)/(x - 1) = x + 1, x != 1")
    ax.scatter([1.0], [2.0], s=90, facecolors="white", edgecolors="#1a73e8", linewidths=2.0, zorder=5)
    ax.scatter([1.0], [3.0], s=80, color="#d93025", zorder=6, label="define f(1) = 3")
    ax.axvline(1.0, color="#9aa0a6", ls="--", lw=1.2)
    ax.axhline(2.0, color="#9aa0a6", ls=":", lw=1.2)
    ax.annotate(
        "limit is 2\n(near x = 1)",
        xy=(1.0, 2.0),
        xytext=(1.55, 1.45),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    ax.annotate(
        "function value is 3\n(at x = 1)",
        xy=(1.0, 3.0),
        xytext=(1.8, 3.25),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    ax.set_xlim(-1.6, 3.2)
    ax.set_ylim(-0.8, 4.2)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Function limit depends on nearby values, not only f(a)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "highmath_function_limit.png", bbox_inches="tight")
    plt.close(fig)


def fig_discontinuities() -> None:
    """Plot common types of discontinuities."""
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))

    x_left = np.linspace(-2.0, 0.98, 200)
    x_right = np.linspace(1.02, 2.0, 200)
    for x in (x_left, x_right):
        axes[0].plot(x, x + 1.0, color="#1a73e8", lw=2.0)
    axes[0].scatter([1.0], [2.0], s=70, facecolors="white", edgecolors="#1a73e8", linewidths=2.0)
    axes[0].scatter([1.0], [3.0], s=60, color="#d93025")
    axes[0].set_title("Removable")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].grid(alpha=0.2)

    x1 = np.linspace(-2.0, 0.0, 200, endpoint=False)
    x2 = np.linspace(0.0, 2.0, 200)
    axes[1].plot(x1, x1 + 1.0, color="#1a73e8", lw=2.0)
    axes[1].plot(x2, x2 + 2.0, color="#1a73e8", lw=2.0)
    axes[1].scatter([0.0], [1.0], s=70, facecolors="white", edgecolors="#1a73e8", linewidths=2.0)
    axes[1].scatter([0.0], [2.0], s=60, color="#1a73e8")
    axes[1].set_title("Jump")
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.2)

    x3_left = np.linspace(-2.0, 0.92, 300)
    x3_right = np.linspace(1.08, 2.0, 300)
    axes[2].plot(x3_left, 1.0 / (x3_left - 1.0), color="#1a73e8", lw=2.0)
    axes[2].plot(x3_right, 1.0 / (x3_right - 1.0), color="#1a73e8", lw=2.0)
    axes[2].axvline(1.0, color="#9aa0a6", ls="--", lw=1.2)
    axes[2].set_ylim(-8.0, 8.0)
    axes[2].set_title("Infinite")
    axes[2].set_xlabel("x")
    axes[2].grid(alpha=0.2)

    fig.suptitle("Typical discontinuities", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "highmath_discontinuities.png", bbox_inches="tight")
    plt.close(fig)


def fig_closed_interval() -> None:
    """Plot a continuous function on a closed interval with root/max/min."""
    x = np.linspace(-2.0, 2.0, 600)
    y = 0.3 * x**3 - x + 0.2
    root_idx = int(np.argmin(np.abs(y)))
    max_idx = int(np.argmax(y))
    min_idx = int(np.argmin(y))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(x, y, color="#1a73e8", lw=2.2, label="continuous on [a, b]")
    ax.axhline(0.0, color="#9aa0a6", lw=1.0)
    ax.axvline(-2.0, color="#9aa0a6", ls="--", lw=1.0)
    ax.axvline(2.0, color="#9aa0a6", ls="--", lw=1.0)
    ax.scatter([x[root_idx]], [y[root_idx]], color="#34a853", s=65, zorder=5, label="a zero")
    ax.scatter([x[max_idx]], [y[max_idx]], color="#d93025", s=65, zorder=5, label="maximum")
    ax.scatter([x[min_idx]], [y[min_idx]], color="#f9ab00", s=65, zorder=5, label="minimum")
    ax.annotate("root exists", xy=(x[root_idx], y[root_idx]), xytext=(-1.9, 0.55),
                arrowprops=dict(arrowstyle="->", color="#202124"), fontsize=10)
    ax.annotate("max", xy=(x[max_idx], y[max_idx]), xytext=(0.65, 0.95),
                arrowprops=dict(arrowstyle="->", color="#202124"), fontsize=10)
    ax.annotate("min", xy=(x[min_idx], y[min_idx]), xytext=(-0.45, -1.05),
                arrowprops=dict(arrowstyle="->", color="#202124"), fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Continuous function on a closed interval")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "highmath_closed_interval.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all figures."""
    fig_sequence_limit()
    fig_function_limit()
    fig_discontinuities()
    fig_closed_interval()


if __name__ == "__main__":
    main()
