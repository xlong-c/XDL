"""Generate figures for highmath/session2.md.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/assets/gen_highmath_session2_figs.py

Outputs:
    learn/math/assets/highmath_secant_tangent.png
    learn/math/assets/highmath_nondifferentiable.png
    learn/math/assets/highmath_linear_approx.png
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


def fig_secant_tangent() -> None:
    """Plot secant lines converging to a tangent line."""
    x = np.linspace(-0.2, 2.6, 500)
    y = x**2
    x0 = 1.0
    y0 = x0**2
    tangent_slope = 2.0 * x0
    tangent = y0 + tangent_slope * (x - x0)
    h_values = [0.8, 0.25]
    colors = ["#f28b82", "#f9ab00"]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(x, y, color="#1a73e8", lw=2.4, label=r"$y=x^2$")
    ax.plot(x, tangent, color="#34a853", lw=2.2, label="tangent at x0 = 1")
    ax.scatter([x0], [y0], color="#202124", s=60, zorder=5)
    ax.annotate(
        "P(x0, f(x0))",
        xy=(x0, y0),
        xytext=(0.4, 1.65),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )

    for idx, (h, color) in enumerate(zip(h_values, colors)):
        x1 = x0 + h
        y1 = x1**2
        secant_slope = (y1 - y0) / h
        secant = y0 + secant_slope * (x - x0)
        ax.plot(
            x,
            secant,
            color=color,
            lw=1.9,
            ls="--",
            label=f"secant, h = {h:.2f}",
        )
        ax.scatter([x1], [y1], color=color, s=52, zorder=5)
        if idx == 0:
            ax.annotate(
                "Q moves toward P\nand secant becomes tangent",
                xy=(x1, y1),
                xytext=(1.95, 4.2),
                arrowprops=dict(arrowstyle="->", color="#202124"),
                fontsize=10,
            )

    ax.set_xlim(-0.2, 2.6)
    ax.set_ylim(-0.2, 6.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Average rate approaches instantaneous rate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "highmath_secant_tangent.png", bbox_inches="tight")
    plt.close(fig)


def fig_nondifferentiable() -> None:
    """Plot several common continuous but nondifferentiable points."""
    x = np.linspace(-2.0, 2.0, 600)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))

    axes[0].plot(x, np.abs(x), color="#1a73e8", lw=2.2)
    axes[0].scatter([0.0], [0.0], color="#d93025", s=55, zorder=5)
    axes[0].set_title(r"Corner: $|x|$")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.2)

    axes[1].plot(x, np.abs(x) ** (2.0 / 3.0), color="#1a73e8", lw=2.2)
    axes[1].scatter([0.0], [0.0], color="#d93025", s=55, zorder=5)
    axes[1].set_title(r"Cusp: $x^{2/3}$")
    axes[1].set_xlabel("x")
    axes[1].grid(alpha=0.2)

    axes[2].plot(x, np.sign(x) * np.abs(x) ** (1.0 / 3.0), color="#1a73e8", lw=2.2)
    axes[2].scatter([0.0], [0.0], color="#d93025", s=55, zorder=5)
    axes[2].set_title(r"Vertical tangent: $x^{1/3}$")
    axes[2].set_xlabel("x")
    axes[2].grid(alpha=0.2)

    fig.suptitle("Continuous points where the derivative does not exist", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "highmath_nondifferentiable.png", bbox_inches="tight")
    plt.close(fig)


def fig_linear_approx() -> None:
    """Plot a tangent line as the linear approximation of a function."""
    x = np.linspace(-0.6, 0.9, 500)
    y = np.exp(x)
    tangent = 1.0 + x
    x0 = 0.0
    y0 = 1.0
    dx = 0.25
    x1 = x0 + dx
    y1 = float(np.exp(x1))
    dy = dx
    y_linear = y0 + dy

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(x, y, color="#1a73e8", lw=2.4, label=r"$y=e^x$")
    ax.plot(x, tangent, color="#34a853", lw=2.1, label=r"linear part $y=1+x$")
    ax.scatter([x0, x1], [y0, y1], color="#202124", s=52, zorder=5)
    ax.vlines(x1, y0, y1, color="#d93025", lw=2.0, label=r"actual increment $\Delta y$")
    ax.vlines(x1 + 0.04, y0, y_linear, color="#f9ab00", lw=2.0, label=r"differential $dy$")
    ax.annotate(
        r"$dx=\Delta x$",
        xy=(x1, y0 - 0.03),
        xytext=(0.07, 0.78),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    ax.annotate(
        r"$\Delta y$",
        xy=(x1, (y0 + y1) / 2.0),
        xytext=(0.39, 1.18),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    ax.annotate(
        r"$dy$",
        xy=(x1 + 0.04, (y0 + y_linear) / 2.0),
        xytext=(0.53, 1.12),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    ax.set_xlim(-0.6, 0.9)
    ax.set_ylim(0.45, 2.55)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Differential keeps the linear main part of the increment")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "highmath_linear_approx.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all figures for session2."""
    fig_secant_tangent()
    fig_nondifferentiable()
    fig_linear_approx()


if __name__ == "__main__":
    main()
