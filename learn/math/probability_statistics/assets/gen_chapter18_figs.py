"""Generate figures for chapter 18 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter18_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch18_*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def fig_loss_probability_assumptions() -> None:
    """Draw common losses and their probability assumptions."""
    rows = [
        ("MSE", "Gaussian noise", "y = f(x) + eps,\neps ~ N(0,sigma^2)", "#e8f0fe", "#1a73e8"),
        ("MAE", "Laplace noise", "eps has sharper peak\nand heavier tails", "#e6f4ea", "#34a853"),
        ("BCE", "Bernoulli likelihood", "y in {0,1},\np = sigmoid(logit)", "#fce8e6", "#d93025"),
        ("Cross Entropy", "Categorical likelihood", "one class among K,\np = softmax(logits)", "#fff7d6", "#f9ab00"),
        ("NLL", "Density model", "maximize assigned\nprobability density", "#f3e8fd", "#9334e6"),
    ]

    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Many deep learning losses are negative log likelihoods under probability assumptions", fontsize=14)

    headers = [("Loss", 0.15), ("Probability view", 0.43), ("Modeling assumption", 0.74)]
    for text, x in headers:
        ax.text(x, 0.91, text, ha="center", va="center", fontsize=12, weight="bold", color="#202124")

    y_positions = np.linspace(0.78, 0.18, len(rows))
    for y, (loss, view, assumption, fc, ec) in zip(y_positions, rows):
        ax.add_patch(
            FancyBboxPatch(
                (0.04, y - 0.055),
                0.22,
                0.11,
                boxstyle="round,pad=0.014,rounding_size=0.018",
                facecolor=fc,
                edgecolor=ec,
                lw=1.9,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.31, y - 0.055),
                0.25,
                0.11,
                boxstyle="round,pad=0.014,rounding_size=0.018",
                facecolor="#ffffff",
                edgecolor=ec,
                lw=1.5,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.62, y - 0.055),
                0.32,
                0.11,
                boxstyle="round,pad=0.014,rounding_size=0.018",
                facecolor="#ffffff",
                edgecolor=ec,
                lw=1.5,
            )
        )
        ax.text(0.15, y, loss, ha="center", va="center", fontsize=11.5, weight="bold", color="#202124")
        ax.text(0.435, y, view, ha="center", va="center", fontsize=10.5, color="#202124")
        ax.text(0.78, y, assumption, ha="center", va="center", fontsize=10.0, color="#202124")
        ax.add_patch(FancyArrowPatch((0.265, y), (0.305, y), arrowstyle="->", mutation_scale=13, lw=1.6, color="#5f6368"))
        ax.add_patch(FancyArrowPatch((0.565, y), (0.615, y), arrowstyle="->", mutation_scale=13, lw=1.6, color="#5f6368"))

    ax.text(
        0.50,
        0.055,
        "Changing the loss often means changing the implied noise model or likelihood.",
        ha="center",
        fontsize=11,
        color="#3c4043",
    )
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch18_loss_probability_assumptions.png", bbox_inches="tight")
    plt.close(fig)


def _calibration_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Return deterministic confidence bins and calibration summaries."""
    rng = np.random.default_rng(801)
    n = 2600
    confidence = rng.beta(4.2, 1.7, size=n)
    true_accuracy = 0.08 + 0.86 * confidence - 0.10 * confidence * (1.0 - confidence)
    true_accuracy = np.clip(true_accuracy, 0.02, 0.98)
    labels = rng.binomial(1, true_accuracy)
    overconfident = np.clip(0.13 + 0.93 * confidence, 0.0, 0.995)
    calibrated = np.clip(0.05 + 0.90 * confidence, 0.0, 0.995)

    bins = np.linspace(0.0, 1.0, 11)
    centers = (bins[:-1] + bins[1:]) / 2.0

    def summarize(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        acc = np.zeros_like(centers)
        conf = np.zeros_like(centers)
        counts = np.zeros_like(centers)
        for i in range(len(centers)):
            mask = (scores >= bins[i]) & (scores < bins[i + 1])
            if i == len(centers) - 1:
                mask = (scores >= bins[i]) & (scores <= bins[i + 1])
            counts[i] = mask.sum()
            if counts[i] > 0:
                acc[i] = labels[mask].mean()
                conf[i] = scores[mask].mean()
            else:
                acc[i] = np.nan
                conf[i] = centers[i]
        ece = float(np.nansum((counts / counts.sum()) * np.abs(acc - conf)))
        brier = float(np.mean((scores - labels) ** 2))
        sparse_bins = counts < 25
        acc[sparse_bins] = np.nan
        conf[sparse_bins] = np.nan
        return acc, conf, ece, brier

    acc_over, conf_over, ece_over, brier_over = summarize(overconfident)
    acc_cal, conf_cal, ece_cal, brier_cal = summarize(calibrated)
    return conf_over, acc_over, conf_cal, acc_cal, ece_over, brier_over, ece_cal, brier_cal


def fig_calibration_confidence() -> None:
    """Draw calibration curves for overconfident and calibrated models."""
    conf_over, acc_over, conf_cal, acc_cal, ece_over, brier_over, ece_cal, brier_cal = _calibration_data()

    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    ax.plot([0, 1], [0, 1], color="#202124", lw=1.7, ls="--", label="perfect calibration")
    ax.plot(conf_over, acc_over, "o-", color="#d93025", lw=2.3, label=f"overconfident, ECE={ece_over:.3f}")
    ax.plot(conf_cal, acc_cal, "s-", color="#1a73e8", lw=2.3, label=f"better calibrated, ECE={ece_cal:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("High confidence is useful only when it is calibrated")
    ax.grid(alpha=0.23)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(
        0.58,
        0.16,
        f"Brier scores\nred: {brier_over:.3f}\nblue: {brier_cal:.3f}",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#ffffff", "ec": "#dadce0"},
    )
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch18_calibration_confidence.png", bbox_inches="tight")
    plt.close(fig)


def fig_vae_elbo_structure() -> None:
    """Draw VAE encoder, latent variable, decoder and ELBO terms."""
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2.0, y - h / 2.0),
                w,
                h,
                boxstyle="round,pad=0.018,rounding_size=0.02",
                facecolor=fc,
                edgecolor=ec,
                lw=2.0,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=10.5, color="#202124")

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=16, lw=2.0, color="#5f6368"))
        if label:
            ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0 + 0.035, label, ha="center", fontsize=9.5, color="#3c4043")

    box(0.10, 0.62, 0.15, 0.16, "data\nx", "#e8f0fe", "#1a73e8")
    box(0.30, 0.62, 0.20, 0.16, "encoder\nq_phi(z|x)", "#e6f4ea", "#34a853")
    box(0.52, 0.62, 0.16, 0.16, "latent\nz", "#fff7d6", "#f9ab00")
    box(0.73, 0.62, 0.20, 0.16, "decoder\np_theta(x|z)", "#fce8e6", "#d93025")
    box(0.91, 0.62, 0.14, 0.16, "reconstruct\nx_hat", "#f3e8fd", "#9334e6")
    arrow(0.18, 0.62, 0.20, 0.62)
    arrow(0.40, 0.62, 0.44, 0.62, "sample")
    arrow(0.60, 0.62, 0.63, 0.62)
    arrow(0.83, 0.62, 0.84, 0.62)

    xs = np.linspace(-3.0, 3.0, 240)
    ys = np.exp(-0.5 * xs**2)
    ys = 0.14 * ys / ys.max() + 0.20
    ax.plot(0.52 + xs * 0.035, ys, color="#f9ab00", lw=2.0)
    ax.text(0.52, 0.15, "prior p(z) usually pulls latent codes toward a simple distribution", ha="center", fontsize=10.2)

    ax.add_patch(Rectangle((0.18, 0.035), 0.64, 0.095, facecolor="#ffffff", edgecolor="#dadce0", lw=1.5))
    ax.text(
        0.50,
        0.082,
        "ELBO = reconstruction term - KL(q_phi(z|x) || p(z))",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#202124",
    )
    ax.set_title("VAE turns representation learning into probabilistic latent-variable modeling", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch18_vae_elbo_structure.png", bbox_inches="tight")
    plt.close(fig)


def _smooth_image(size: int = 40) -> np.ndarray:
    """Create a deterministic smooth image used as a clean sample."""
    xs = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(xs, xs)
    img = 0.35 + 0.45 * np.exp(-((xx + 0.25) ** 2 + (yy + 0.20) ** 2) / 0.16)
    img += 0.30 * np.exp(-((xx - 0.35) ** 2 + (yy - 0.30) ** 2) / 0.08)
    img += 0.18 * np.exp(-((xx + 0.15) ** 2 + (yy - 0.45) ** 2) / 0.05)
    return np.clip(img, 0.0, 1.0)


def fig_diffusion_denoising_process() -> None:
    """Draw forward noising and reverse denoising view of diffusion models."""
    rng = np.random.default_rng(809)
    clean = _smooth_image()
    noise = rng.normal(0.5, 0.28, size=clean.shape)
    levels = [0.0, 0.25, 0.50, 0.78, 1.0]
    forward = [np.clip((1.0 - level) * clean + level * noise, 0.0, 1.0) for level in levels]

    fig, axes = plt.subplots(2, 5, figsize=(13.0, 6.0))
    for i, ax in enumerate(axes[0]):
        ax.imshow(forward[i], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"t={i}")
        ax.axis("off")
        if i < 4:
            ax.annotate("", xy=(1.13, 0.5), xytext=(1.02, 0.5), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#5f6368"})

    reverse = list(reversed(forward))
    for i, ax in enumerate(axes[1]):
        ax.imshow(reverse[i], cmap="viridis", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("noise")
        elif i == 4:
            ax.set_title("sample")
        else:
            ax.set_title("denoise")
        if i < 4:
            ax.annotate("", xy=(1.13, 0.5), xytext=(1.02, 0.5), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#5f6368"})

    axes[0, 0].text(-0.24, 0.5, "forward\nadd noise", transform=axes[0, 0].transAxes, ha="center", va="center", fontsize=11)
    axes[1, 0].text(-0.24, 0.5, "reverse\nlearn denoise", transform=axes[1, 0].transAxes, ha="center", va="center", fontsize=11)
    fig.suptitle("Diffusion models learn to reverse a gradual noising process", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch18_diffusion_denoising_process.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 18 figures."""
    fig_loss_probability_assumptions()
    fig_calibration_confidence()
    fig_vae_elbo_structure()
    fig_diffusion_denoising_process()


if __name__ == "__main__":
    main()
