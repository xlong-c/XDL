"""Generate figures for chapter 19 of probability_statistics.

Run:
    MPLCONFIGDIR=/tmp/mpl python learn/math/probability_statistics/assets/gen_chapter19_figs.py

Outputs:
    learn/math/probability_statistics/assets/ps_ch19_*.png
"""

from math import exp, pi, sqrt
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


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Return Normal(mu, sigma^2) density."""
    return (1.0 / (sqrt(2.0 * pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def fig_real_ml_workflow() -> None:
    """Draw a practical machine learning workflow."""
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2.0, y - h / 2.0),
                w,
                h,
                boxstyle="round,pad=0.014,rounding_size=0.02",
                facecolor=fc,
                edgecolor=ec,
                lw=2.0,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=10.1, color="#202124")

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=15, lw=2.0, color="#5f6368"))

    colors = [
        ("#e8f0fe", "#1a73e8"),
        ("#e6f4ea", "#34a853"),
        ("#fff7d6", "#f9ab00"),
        ("#fce8e6", "#d93025"),
        ("#f3e8fd", "#9334e6"),
    ]
    top = [
        (0.11, "Data source\nsampling plan"),
        (0.30, "Cleaning\nlabel audit"),
        (0.50, "Train / val / test\nsplit design"),
        (0.70, "Model training\nand tuning"),
        (0.89, "Offline evaluation\nand model choice"),
    ]
    bottom = [
        (0.76, "Decision rule\nthreshold and cost"),
        (0.56, "Online experiment\nor shadow test"),
        (0.36, "Monitoring\ndrift and calibration"),
        (0.16, "Error analysis\nfeedback loop"),
    ]

    for i, (x, text) in enumerate(top):
        fc, ec = colors[i % len(colors)]
        box(x, 0.68, 0.16, 0.16, text, fc, ec)
        if i < len(top) - 1:
            arrow(x + 0.08, 0.68, top[i + 1][0] - 0.08, 0.68)

    arrow(0.89, 0.60, 0.76, 0.44)
    for i, (x, text) in enumerate(bottom):
        fc, ec = colors[(i + 2) % len(colors)]
        box(x, 0.34, 0.17, 0.16, text, fc, ec)
        if i < len(bottom) - 1:
            arrow(x - 0.085, 0.34, bottom[i + 1][0] + 0.085, 0.34)
    arrow(0.16, 0.42, 0.11, 0.60)

    ax.text(
        0.50,
        0.10,
        "A real workflow closes the loop: data, evaluation, decision, deployment, monitoring and error analysis.",
        ha="center",
        fontsize=11.5,
        color="#3c4043",
    )
    ax.set_title("Real machine learning work is an evaluation and decision loop", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch19_real_ml_workflow.png", bbox_inches="tight")
    plt.close(fig)


def _drift_data() -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    """Return train and serving feature samples plus drift summaries."""
    rng = np.random.default_rng(901)
    train = rng.normal(0.0, 1.0, size=4000)
    serving = np.concatenate(
        [
            rng.normal(0.55, 1.05, size=3200),
            rng.normal(2.10, 0.45, size=800),
        ]
    )
    bins = np.quantile(train, np.linspace(0.0, 1.0, 11))
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    train_hist, _ = np.histogram(train, bins=bins)
    serving_hist, _ = np.histogram(serving, bins=bins)
    train_prop = np.maximum(train_hist / train.size, 1e-6)
    serving_prop = np.maximum(serving_hist / serving.size, 1e-6)
    psi = float(np.sum((serving_prop - train_prop) * np.log(serving_prop / train_prop)))
    return train, serving, float(train.mean()), float(train.std()), float(serving.mean()), float(serving.std()), psi


def fig_data_drift() -> None:
    """Plot a feature distribution shift between training and serving data."""
    train, serving, train_mean, train_std, serving_mean, serving_std, psi = _drift_data()
    xs = np.linspace(-4.0, 4.2, 500)

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    ax.hist(train, bins=52, density=True, alpha=0.42, color="#1a73e8", edgecolor="white", label="training feature")
    ax.hist(serving, bins=52, density=True, alpha=0.40, color="#d93025", edgecolor="white", label="serving feature")
    ax.plot(xs, _normal_pdf(xs, train_mean, train_std), color="#1a73e8", lw=2.3)
    ax.plot(xs, _normal_pdf(xs, serving_mean, serving_std), color="#d93025", lw=2.3)
    ax.axvline(train_mean, color="#1a73e8", ls="--", lw=1.6)
    ax.axvline(serving_mean, color="#d93025", ls="--", lw=1.6)
    ax.text(
        1.45,
        0.12,
        f"train mean={train_mean:.2f}, sd={train_std:.2f}\nserving mean={serving_mean:.2f}, sd={serving_std:.2f}\nPSI={psi:.3f}",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#ffffff", "ec": "#dadce0"},
    )
    ax.set_title("Data drift: serving distribution can move away from training distribution")
    ax.set_xlabel("feature value")
    ax.set_ylabel("density")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch19_data_drift.png", bbox_inches="tight")
    plt.close(fig)


def _threshold_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float]:
    """Return threshold metrics and the cost-minimizing threshold."""
    rng = np.random.default_rng(907)
    n_pos = 520
    n_neg = 1080
    pos_scores = rng.beta(5.1, 2.1, size=n_pos)
    neg_scores = rng.beta(2.0, 5.4, size=n_neg)
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    thresholds = np.linspace(0.05, 0.95, 91)
    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)
    f1 = np.zeros_like(thresholds)
    cost = np.zeros_like(thresholds)
    fp_cost = 1.0
    fn_cost = 4.0

    for i, threshold in enumerate(thresholds):
        pred = scores >= threshold
        tp = np.sum(pred & (labels == 1))
        fp = np.sum(pred & (labels == 0))
        fn = np.sum(~pred & (labels == 1))
        precision[i] = tp / max(tp + fp, 1)
        recall[i] = tp / max(tp + fn, 1)
        f1[i] = 2.0 * precision[i] * recall[i] / max(precision[i] + recall[i], 1e-12)
        cost[i] = (fp_cost * fp + fn_cost * fn) / labels.size

    best_idx = int(np.argmin(cost))
    best_f1_idx = int(np.argmax(f1))
    prevalence = float(labels.mean())
    return (
        thresholds,
        precision,
        recall,
        f1,
        cost,
        float(thresholds[best_idx]),
        float(cost[best_idx]),
        float(thresholds[best_f1_idx]),
        float(f1[best_f1_idx]),
        prevalence,
    )


def fig_threshold_metrics() -> None:
    """Plot how metrics and expected cost change with threshold."""
    thresholds, precision, recall, f1, cost, best_cost_t, best_cost, best_f1_t, best_f1, prevalence = _threshold_data()

    fig, ax1 = plt.subplots(figsize=(9.4, 5.7))
    ax1.plot(thresholds, precision, color="#1a73e8", lw=2.2, label="precision")
    ax1.plot(thresholds, recall, color="#34a853", lw=2.2, label="recall")
    ax1.plot(thresholds, f1, color="#9334e6", lw=2.2, label="F1")
    ax1.axvline(best_f1_t, color="#9334e6", lw=1.5, ls="--", alpha=0.7)
    ax1.set_xlabel("decision threshold")
    ax1.set_ylabel("metric value")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, cost, color="#d93025", lw=2.4, label="expected cost")
    ax2.axvline(best_cost_t, color="#d93025", lw=1.5, ls="--", alpha=0.7)
    ax2.set_ylabel("cost per sample")
    ax2.set_ylim(0.0, max(cost) * 1.08)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax1.text(
        0.07,
        0.12,
        f"prevalence={prevalence:.3f}\nbest cost threshold={best_cost_t:.2f}\nbest cost={best_cost:.3f}\nbest F1 threshold={best_f1_t:.2f}",
        transform=ax1.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#ffffff", "ec": "#dadce0"},
    )
    ax1.set_title("Threshold choice changes precision, recall, F1 and decision cost")
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch19_threshold_metrics.png", bbox_inches="tight")
    plt.close(fig)


def fig_error_analysis_loop() -> None:
    """Draw an error analysis loop."""
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        (0.50, 0.82, "Collect errors\nand hard cases", "#e8f0fe", "#1a73e8"),
        (0.78, 0.58, "Slice by group\nsource and time", "#e6f4ea", "#34a853"),
        (0.66, 0.22, "Identify root cause\nlabel, data, model, decision", "#fff7d6", "#f9ab00"),
        (0.34, 0.22, "Prioritize fixes\nby impact and cost", "#fce8e6", "#d93025"),
        (0.22, 0.58, "Run controlled\nexperiment", "#f3e8fd", "#9334e6"),
    ]

    for i in range(len(nodes)):
        x0, y0 = nodes[i][0], nodes[i][1]
        x1, y1 = nodes[(i + 1) % len(nodes)][0], nodes[(i + 1) % len(nodes)][1]
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="->",
                connectionstyle="arc3,rad=-0.14",
                mutation_scale=16,
                lw=2.0,
                color="#5f6368",
                zorder=1,
            )
        )

    for x, y, text, fc, ec in nodes:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.135, y - 0.07),
                0.27,
                0.14,
                boxstyle="round,pad=0.018,rounding_size=0.02",
                facecolor=fc,
                edgecolor=ec,
                lw=2.0,
                zorder=2,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=10.0, color="#202124", zorder=3)

    ax.add_patch(
        FancyBboxPatch(
            (0.36, 0.46),
            0.28,
            0.12,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            facecolor="#ffffff",
            edgecolor="#dadce0",
            lw=1.8,
            zorder=4,
        )
    )
    ax.text(0.50, 0.52, "Track: count, severity,\nconfidence, user impact", ha="center", va="center", fontsize=10.2, color="#202124", zorder=5)
    ax.set_title("Error analysis is a repeated measurement and intervention loop", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "ps_ch19_error_analysis_loop.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate all chapter 19 figures."""
    fig_real_ml_workflow()
    fig_data_drift()
    fig_threshold_metrics()
    fig_error_analysis_loop()


if __name__ == "__main__":
    main()
