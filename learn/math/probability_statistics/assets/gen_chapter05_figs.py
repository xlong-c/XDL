"""Generate figures for chapter 05 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter05_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch05_*.png
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


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return the Normal(mu, sigma^2) density."""
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def _entropy_bits(p: np.ndarray) -> float:
    """Compute discrete entropy in bits."""
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def fig_same_mean_different_variance() -> None:
    """Plot distributions with the same mean and different variances."""
    x = np.linspace(-6.0, 6.0, 700)
    configs = [
        (0.0, 0.7, "#1a73e8", "sigma = 0.7"),
        (0.0, 1.3, "#34a853", "sigma = 1.3"),
        (0.0, 2.2, "#d93025", "sigma = 2.2"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for mu, sigma, color, label in configs:
        ax.plot(x, _normal_pdf(x, mu, sigma), color=color, lw=2.4, label=label)
    ax.axvline(0.0, color="#202124", ls="--", lw=1.2, label="same mean")
    ax.set_title("Same mean, different variance")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    ax.text(1.6, 0.42, "larger variance -> wider spread", color="#d93025", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch05_same_mean_different_variance.png", bbox_inches="tight")
    plt.close(fig)


def fig_covariance_ellipse_axes() -> None:
    """Plot a covariance ellipse and principal axes."""
    cov = np.array([[3.0, 1.2], [1.2, 1.0]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = eigvecs @ np.diag(np.sqrt(eigvals)) @ circle

    rng = np.random.default_rng(12)
    points = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=650)

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.22, color="#1a73e8", edgecolors="none")
    ax.plot(ellipse[0], ellipse[1], color="#d93025", lw=2.5, label="1-sigma covariance ellipse")

    colors = ["#202124", "#34a853"]
    for i in range(2):
        vec = eigvecs[:, i]
        length = np.sqrt(eigvals[i])
        ax.annotate(
            "",
            xy=vec * length,
            xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color=colors[i], lw=2.6),
        )
        ax.text(*(vec * length * 1.16), f"lambda{i + 1}={eigvals[i]:.2f}", color=colors[i], fontsize=10)

    ax.set_title("Covariance matrix: eigenvectors are principal spread directions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch05_covariance_ellipse_axes.png", bbox_inches="tight")
    plt.close(fig)


def fig_entropy_comparison() -> None:
    """Compare entropy of peaked, medium and uniform distributions."""
    distributions = [
        ("peaked", np.array([0.86, 0.05, 0.04, 0.03, 0.02]), "#d93025"),
        ("medium", np.array([0.42, 0.25, 0.16, 0.10, 0.07]), "#f9ab00"),
        ("uniform", np.ones(5) / 5.0, "#1a73e8"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), sharey=True)
    for ax, (name, p, color) in zip(axes, distributions):
        h = _entropy_bits(p)
        ax.bar(np.arange(len(p)), p, color=color, alpha=0.82, edgecolor="#202124")
        ax.set_title(f"{name}\nH = {h:.2f} bits")
        ax.set_xlabel("outcome")
        ax.set_xticks(np.arange(len(p)))
        ax.grid(axis="y", alpha=0.22)
        if ax is axes[0]:
            ax.set_ylabel("probability")
    fig.suptitle("Entropy is larger when probability mass is more evenly spread", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch05_entropy_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_cross_entropy_kl_relation() -> None:
    """Visualize H(p, q) = H(p) + KL(p || q)."""
    labels = ["A", "B", "C", "D"]
    p = np.array([0.70, 0.20, 0.08, 0.02])
    q_good = np.array([0.62, 0.24, 0.10, 0.04])
    q_bad = np.array([0.20, 0.20, 0.30, 0.30])

    def cross_entropy_bits(pv: np.ndarray, qv: np.ndarray) -> float:
        return float(-(pv * np.log2(qv)).sum())

    h = _entropy_bits(p)
    ce_good = cross_entropy_bits(p, q_good)
    ce_bad = cross_entropy_bits(p, q_bad)
    kl_good = ce_good - h
    kl_bad = ce_bad - h

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)
    x = np.arange(len(labels))
    width = 0.34
    for ax, q, title, ce, kl in [
        (axes[0], q_good, "q close to p", ce_good, kl_good),
        (axes[1], q_bad, "q far from p", ce_bad, kl_bad),
    ]:
        ax.bar(x - width / 2, p, width=width, color="#1a73e8", label="true p")
        ax.bar(x + width / 2, q, width=width, color="#d93025", alpha=0.78, label="model q")
        ax.set_xticks(x, labels=labels)
        ax.set_title(f"{title}\nH(p,q)={ce:.2f}, KL={kl:.2f} bits")
        ax.set_xlabel("class")
        ax.grid(axis="y", alpha=0.22)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("probability")
    fig.suptitle(f"Cross entropy = entropy of p + KL penalty. H(p) = {h:.2f} bits", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch05_cross_entropy_kl_relation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 05 figures."""
    fig_same_mean_different_variance()
    fig_covariance_ellipse_axes()
    fig_entropy_comparison()
    fig_cross_entropy_kl_relation()


if __name__ == "__main__":
    main()
