"""Generate figures for chapter 04 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter04_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch04_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def fig_joint_table_heatmap() -> None:
    """Joint PMF heatmap with marginal bars around it."""
    joint = np.array(
        [
            [0.06, 0.09, 0.05],
            [0.14, 0.21, 0.10],
            [0.08, 0.15, 0.12],
        ]
    )
    row_names = ["X=0", "X=1", "X=2"]
    col_names = ["Y=0", "Y=1", "Y=2"]
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    fig = plt.figure(figsize=(8.8, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4.0, 1.1], height_ratios=[1.1, 4.0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    im = ax_main.imshow(joint, cmap="Blues", vmin=0.0, vmax=joint.max())
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            ax_main.text(j, i, f"{joint[i, j]:.2f}", ha="center", va="center", fontsize=12)
    ax_main.set_xticks(range(3), labels=col_names)
    ax_main.set_yticks(range(3), labels=row_names)
    ax_main.set_title("Joint PMF p(X, Y)")

    ax_top.bar(np.arange(3), py, color="#f9ab00", edgecolor="#c26401")
    for i, v in enumerate(py):
        ax_top.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax_top.set_xlim(-0.5, 2.5)
    ax_top.set_xticks([])
    ax_top.set_ylabel("P(Y=y)")
    ax_top.set_title("Marginal of Y")
    ax_top.grid(alpha=0.2)

    ax_right.barh(np.arange(3), px, color="#34a853", edgecolor="#137333")
    for i, v in enumerate(px):
        ax_right.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)
    ax_right.set_ylim(-0.5, 2.5)
    ax_right.set_yticks([])
    ax_right.set_xlabel("P(X=x)")
    ax_right.set_title("Marginal of X")
    ax_right.grid(alpha=0.2)

    fig.colorbar(im, ax=[ax_main], shrink=0.78, label="joint probability")
    fig.suptitle("Joint distribution contains everything; marginals come from summing out one variable", fontsize=14)
    fig.savefig(OUT / "ps_ch04_joint_table_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def fig_marginal_projection() -> None:
    """2D density with projection to marginals."""
    x = np.linspace(-3.5, 3.5, 250)
    y = np.linspace(-3.5, 3.5, 250)
    X, Y = np.meshgrid(x, y)
    sigma_x, sigma_y = 1.0, 1.8
    rho = 0.55
    z = np.exp(
        -1.0
        / (2 * (1 - rho**2))
        * ((X / sigma_x) ** 2 - 2 * rho * (X / sigma_x) * (Y / sigma_y) + (Y / sigma_y) ** 2)
    )
    z = z / z.sum()
    px = z.sum(axis=0)
    py = z.sum(axis=1)

    fig = plt.figure(figsize=(10.4, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4.3, 1.2], height_ratios=[1.2, 4.3])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    cf = ax_main.contourf(X, Y, z, levels=18, cmap="viridis")
    ax_main.set_title("Joint density f(X, Y)")
    ax_main.set_xlabel("x")
    ax_main.set_ylabel("y")
    ax_main.annotate(
        "project to f_X(x)",
        xy=(1.7, -0.4),
        xytext=(1.2, -2.35),
        arrowprops=dict(arrowstyle="->", color="white"),
        color="white",
        fontsize=10,
    )
    ax_main.annotate(
        "project to f_Y(y)",
        xy=(-0.4, 2.0),
        xytext=(-2.75, 2.6),
        arrowprops=dict(arrowstyle="->", color="white"),
        color="white",
        fontsize=10,
    )

    ax_top.plot(x, px, color="#f9ab00", lw=2.3)
    ax_top.fill_between(x, 0, px, color="#fff1c6", alpha=0.9)
    ax_top.set_ylabel("f_X(x)")
    ax_top.set_xticks([])
    ax_top.set_title("Marginal density of X")
    ax_top.grid(alpha=0.2)

    ax_right.plot(py, y, color="#34a853", lw=2.3)
    ax_right.fill_betweenx(y, 0, py, color="#d7f2df", alpha=0.9)
    ax_right.set_xlabel("f_Y(y)")
    ax_right.set_yticks([])
    ax_right.set_title("Marginal density of Y")
    ax_right.grid(alpha=0.2)

    fig.colorbar(cf, ax=[ax_main], shrink=0.78, label="density")
    fig.suptitle("Marginalization means projecting a joint distribution onto one axis", fontsize=14)
    fig.savefig(OUT / "ps_ch04_marginal_projection.png", bbox_inches="tight")
    plt.close(fig)


def fig_conditional_slice() -> None:
    """Conditional slice through a joint density."""
    x = np.linspace(-3.0, 3.0, 240)
    y = np.linspace(-3.0, 3.0, 240)
    X, Y = np.meshgrid(x, y)
    sigma_x, sigma_y = 1.0, 1.5
    rho = 0.65
    z = np.exp(
        -1.0
        / (2 * (1 - rho**2))
        * ((X / sigma_x) ** 2 - 2 * rho * (X / sigma_x) * (Y / sigma_y) + (Y / sigma_y) ** 2)
    )
    y0 = 1.0
    idx = int(np.argmin(np.abs(y - y0)))
    cond = z[idx, :]
    cond = cond / np.trapezoid(cond, x)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    ax = axes[0]
    cf = ax.contourf(X, Y, z, levels=18, cmap="viridis")
    ax.axhline(y0, color="#ffffff", ls="--", lw=2.0)
    ax.text(-2.85, y0 + 0.16, "slice at Y = 1.0", color="white", fontsize=10)
    ax.set_title("Joint density with a horizontal slice")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, shrink=0.78)

    ax = axes[1]
    ax.plot(x, cond, color="#d93025", lw=2.4)
    ax.fill_between(x, 0, cond, color="#fce8e6", alpha=0.9)
    ax.set_title("Conditional density f(X | Y = 1.0)")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.2)
    ax.text(-2.5, cond.max() * 0.88, "renormalize the slice so its total area becomes 1", fontsize=10)

    fig.suptitle("Conditioning fixes one variable and then renormalizes along the remaining axis", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch04_conditional_slice.png", bbox_inches="tight")
    plt.close(fig)


def fig_bivariate_normal_contours() -> None:
    """Contours for a bivariate normal and transformed variables."""
    x = np.linspace(-4.0, 4.0, 320)
    y = np.linspace(-4.0, 4.0, 320)
    X, Y = np.meshgrid(x, y)
    sigma_x, sigma_y = 1.2, 0.8
    rho = 0.72
    z = np.exp(
        -1.0
        / (2 * (1 - rho**2))
        * ((X / sigma_x) ** 2 - 2 * rho * (X / sigma_x) * (Y / sigma_y) + (Y / sigma_y) ** 2)
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    cs = ax.contour(X, Y, z, levels=12, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bivariate Normal contours")
    ax.annotate(
        "positive covariance -> tilted ellipses",
        xy=(1.7, 1.3),
        xytext=(-3.1, 2.9),
        arrowprops=dict(arrowstyle="->", color="#202124"),
        fontsize=10,
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch04_bivariate_normal_contours.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 04 figures."""
    fig_joint_table_heatmap()
    fig_marginal_projection()
    fig_conditional_slice()
    fig_bivariate_normal_contours()


if __name__ == "__main__":
    main()
