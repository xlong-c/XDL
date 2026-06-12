from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent

INK = "#182230"
MUTED = "#667085"
TEAL = "#0f766e"
BLUE = "#2563eb"
AMBER = "#b45309"
ROSE = "#be123c"
GREEN = "#047857"
VIOLET = "#7c3aed"
GRID = "#d7dee8"
SURFACE = "#f8fafc"


def save(fig: Figure, name: str) -> None:
    fig.savefig(ROOT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def setup_axis(ax: Axes, xlim: tuple[float, float], ylim: tuple[float, float], equal: bool = True) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.65)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def arrow(ax: Axes, start: tuple[float, float], vec: tuple[float, float], color: str, label: str) -> None:
    ax.arrow(
        start[0],
        start[1],
        vec[0],
        vec[1],
        color=color,
        linewidth=2,
        length_includes_head=True,
        head_width=0.10,
        head_length=0.14,
    )
    ax.text(start[0] + vec[0] * 1.04, start[1] + vec[1] * 1.04, label, color=color, fontsize=9, weight="bold")


def plot_line_intersection() -> None:
    x = np.linspace(-0.5, 3.5, 300)
    y1 = 2.4 - 0.7 * x
    y2 = 0.35 + 0.65 * x
    xi = (2.4 - 0.35) / (0.65 + 0.7)
    yi = 0.35 + 0.65 * xi

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    setup_axis(ax, (-0.5, 3.5), (-0.2, 2.8))
    ax.plot(x, y1, color=BLUE, linewidth=2.4, label=r"$0.7x+y=2.4$")
    ax.plot(x, y2, color=TEAL, linewidth=2.4, label=r"$-0.65x+y=0.35$")
    ax.scatter([xi], [yi], s=70, color=ROSE, zorder=4)
    ax.annotate(
        "solution",
        xy=(xi, yi),
        xytext=(xi + 0.35, yi + 0.35),
        arrowprops={"arrowstyle": "->", "color": ROSE},
        color=ROSE,
        fontsize=10,
        weight="bold",
    )
    ax.set_title("Two linear constraints meet at one point", color=INK, fontsize=12, weight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "la_ch01_line_intersection.png")


def plot_force_equilibrium() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    setup_axis(ax, (-2.4, 2.4), (-1.6, 2.5), equal=True)
    joint = (0.0, 0.0)
    forces = [
        ((0.0, 0.0), (1.45, 1.05), TEAL, r"$T_1$"),
        ((0.0, 0.0), (-1.35, 0.92), BLUE, r"$T_2$"),
        ((0.0, 0.0), (0.0, -1.35), ROSE, r"$mg$"),
    ]
    for start, vec, color, label in forces:
        arrow(ax, start, vec, color, label)
    ax.add_patch(Circle(joint, 0.09, color=INK, zorder=5))
    ax.plot([0, 1.8], [0, 1.3], color=MUTED, linewidth=1.2, linestyle="--")
    ax.plot([0, -1.8], [0, 1.22], color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(-2.1, 2.18, r"$\sum F_x=0,\quad \sum F_y=0$", fontsize=13, color=INK, weight="bold")
    ax.text(-2.1, 1.82, "static equilibrium becomes a linear system", fontsize=9.5, color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Forces at a joint", color=INK, fontsize=12, weight="bold")
    save(fig, "la_ch01_force_equilibrium.png")


def transformed_grid(ax: Axes, matrix: np.ndarray, color: str) -> None:
    values = np.linspace(-2, 2, 9)
    for value in values:
        line = np.vstack([np.linspace(-2, 2, 120), np.full(120, value)])
        mapped = matrix @ line
        ax.plot(mapped[0], mapped[1], color=color, linewidth=0.9, alpha=0.62)
        line = np.vstack([np.full(120, value), np.linspace(-2, 2, 120)])
        mapped = matrix @ line
        ax.plot(mapped[0], mapped[1], color=color, linewidth=0.9, alpha=0.62)


def plot_grid_transform() -> None:
    a = np.array([[1.25, 0.75], [0.35, 1.05]])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    setup_axis(axes[0], (-2.4, 2.4), (-2.4, 2.4))
    transformed_grid(axes[0], np.eye(2), BLUE)
    axes[0].set_title("Before: coordinate grid", color=INK, fontsize=11, weight="bold")
    setup_axis(axes[1], (-3.4, 3.4), (-2.7, 2.7))
    transformed_grid(axes[1], a, TEAL)
    arrow(axes[1], (0, 0), tuple(a[:, 0]), BLUE, r"$A e_1$")
    arrow(axes[1], (0, 0), tuple(a[:, 1]), ROSE, r"$A e_2$")
    axes[1].set_title("After: matrix as a transformation", color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch02_grid_transform.png")


def plot_circuit_linear_system() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.add_patch(Rectangle((0.25, 0.25), 7.5, 4.1, facecolor=SURFACE, edgecolor=GRID, linewidth=1.2))
    ax.plot([1, 3, 5, 7], [3.2, 3.2, 3.2, 3.2], color=INK, linewidth=2)
    ax.plot([1, 1, 7, 7], [1.1, 3.2, 3.2, 1.1], color=INK, linewidth=2)
    for cx, label in [(3, r"$R_1$"), (5, r"$R_2$")]:
        xs = np.linspace(cx - 0.55, cx + 0.55, 9)
        ys = 3.2 + 0.15 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0])
        ax.plot(xs, ys, color=AMBER, linewidth=2)
        ax.text(cx - 0.18, 3.55, label, fontsize=11, color=AMBER, weight="bold")
    ax.plot([4, 4], [3.2, 1.1], color=INK, linewidth=2)
    ax.text(3.75, 2.05, r"$R_3$", color=VIOLET, fontsize=11, weight="bold")
    ax.annotate("", xy=(2.1, 2.9), xytext=(1.2, 2.9), arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2})
    ax.annotate("", xy=(5.9, 2.9), xytext=(4.8, 2.9), arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 2})
    ax.text(1.55, 2.55, r"$I_1$", color=BLUE, fontsize=10, weight="bold")
    ax.text(5.25, 2.55, r"$I_2$", color=TEAL, fontsize=10, weight="bold")
    ax.text(0.8, 0.55, r"KVL/KCL:  $A\,i=b$  and  $i=A^{-1}b$ when the network matrix is invertible.", color=INK, fontsize=10)
    save(fig, "la_ch02_circuit_linear_system.png")


def plot_determinant_area() -> None:
    u = np.array([1.7, 0.55])
    v = np.array([0.45, 1.45])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    setup_axis(ax, (-0.4, 2.8), (-0.3, 2.5))
    poly = np.array([[0, 0], u, u + v, v])
    ax.add_patch(Polygon(poly, closed=True, facecolor="#dff5f0", edgecolor=TEAL, linewidth=2.2))
    arrow(ax, (0, 0), tuple(u), BLUE, r"$a_1$")
    arrow(ax, (0, 0), tuple(v), ROSE, r"$a_2$")
    det = abs(np.linalg.det(np.column_stack([u, v])))
    ax.text(0.28, 2.16, rf"area = $|\det A| \approx {det:.2f}$", color=INK, fontsize=12, weight="bold")
    ax.set_title("Determinant as oriented area scaling", color=INK, fontsize=12, weight="bold")
    save(fig, "la_ch03_determinant_area.png")


def plot_determinant_collapse() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    full = np.array([[1.4, 0.2], [0.3, 1.1]])
    flat = np.array([[1.4, 0.7], [0.6, 0.3]])
    for ax, matrix, title, color in [
        (axes[0], full, "nonzero determinant", TEAL),
        (axes[1], flat, "zero determinant", ROSE),
    ]:
        setup_axis(ax, (-0.3, 2.2), (-0.3, 1.8))
        u = matrix[:, 0]
        v = matrix[:, 1]
        poly = np.array([[0, 0], u, u + v, v])
        ax.add_patch(Polygon(poly, closed=True, facecolor=color, edgecolor=color, alpha=0.22, linewidth=2.2))
        arrow(ax, (0, 0), tuple(u), BLUE, r"$a_1$")
        arrow(ax, (0, 0), tuple(v), ROSE, r"$a_2$")
        ax.set_title(title, color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch03_determinant_collapse.png")


def plot_span_basis() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    setup_axis(axes[0], (-2.2, 2.2), (-1.6, 1.6))
    t = np.linspace(-2.2, 2.2, 120)
    axes[0].plot(t, 0.55 * t, color=TEAL, linewidth=2.2)
    arrow(axes[0], (0, 0), (1.45, 0.8), BLUE, r"$v$")
    arrow(axes[0], (0, 0), (-1.1, -0.6), ROSE, r"$-0.76v$")
    axes[0].set_title("One direction spans a line", color=INK, fontsize=11, weight="bold")

    setup_axis(axes[1], (-2.2, 2.2), (-1.6, 1.8))
    arrow(axes[1], (0, 0), (1.4, 0.35), BLUE, r"$b_1$")
    arrow(axes[1], (0, 0), (0.3, 1.25), ROSE, r"$b_2$")
    for a in np.linspace(-1.4, 1.4, 6):
        for b in np.linspace(-1.0, 1.0, 5):
            point = a * np.array([1.4, 0.35]) + b * np.array([0.3, 1.25])
            axes[1].scatter(point[0], point[1], s=8, color=TEAL, alpha=0.35)
    axes[1].set_title("Two independent directions span a plane", color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch04_span_basis.png")


def plot_incline_components() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    setup_axis(ax, (-0.5, 4.6), (-0.4, 3.2), equal=True)
    x = np.array([0.4, 4.2])
    y = 0.55 * x + 0.15
    ax.plot(x, y, color=INK, linewidth=2.2)
    theta = np.arctan(0.55)
    center = np.array([2.2, 1.36])
    tangent = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    ax.add_patch(Rectangle((center[0] - 0.25, center[1] - 0.18), 0.5, 0.36, angle=np.degrees(theta), facecolor="#eaf1ff", edgecolor=BLUE, linewidth=1.5))
    arrow(ax, tuple(center), tuple(-1.15 * np.array([0.0, 1.0])), ROSE, r"$mg$")
    arrow(ax, tuple(center), tuple(-0.62 * tangent), AMBER, r"$mg\sin\theta$")
    arrow(ax, tuple(center), tuple(-0.92 * normal), TEAL, r"$mg\cos\theta$")
    ax.text(0.7, 2.75, "change basis: gravity -> slope components", color=INK, fontsize=11, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "la_ch04_incline_components.png")


def plot_coordinate_change() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    setup_axis(ax, (-1.5, 3.2), (-1.3, 3.0))
    theta = np.deg2rad(32)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    b1 = np.array([np.cos(theta), np.sin(theta)])
    b2 = np.array([-np.sin(theta), np.cos(theta)])
    x = np.array([2.2, 1.15])
    arrow(ax, (0, 0), tuple(e1), MUTED, r"$e_1$")
    arrow(ax, (0, 0), tuple(e2), MUTED, r"$e_2$")
    arrow(ax, (0, 0), tuple(1.35 * b1), BLUE, r"$b_1$")
    arrow(ax, (0, 0), tuple(1.35 * b2), TEAL, r"$b_2$")
    arrow(ax, (0, 0), tuple(x), ROSE, r"$x$")
    ax.plot([x[0], 0], [x[1], x[1]], linestyle="--", color=GRID)
    ax.plot([x[0], x[0]], [x[1], 0], linestyle="--", color=GRID)
    ax.text(-1.24, 2.58, "same vector, different coordinates", color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch05_coordinate_change.png")


def plot_robot_frame() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    setup_axis(ax, (-1.2, 5.4), (-1.0, 3.8), equal=True)
    base = np.array([0.5, 0.5])
    sensor = np.array([3.2, 1.8])
    theta = np.deg2rad(38)
    bx = np.array([1.0, 0.0])
    by = np.array([0.0, 1.0])
    sx = np.array([np.cos(theta), np.sin(theta)])
    sy = np.array([-np.sin(theta), np.cos(theta)])
    arrow(ax, tuple(base), tuple(0.9 * bx), BLUE, "world x")
    arrow(ax, tuple(base), tuple(0.9 * by), BLUE, "world y")
    arrow(ax, tuple(sensor), tuple(0.75 * sx), TEAL, "cam x")
    arrow(ax, tuple(sensor), tuple(0.75 * sy), TEAL, "cam y")
    ax.plot([base[0], sensor[0]], [base[1], sensor[1]], color=ROSE, linestyle="--", linewidth=1.8)
    ax.add_patch(Circle(tuple(base), 0.08, color=BLUE))
    ax.add_patch(Circle(tuple(sensor), 0.10, color=TEAL))
    ax.text(1.2, 3.35, r"$p_{world}=R\,p_{camera}+t$", color=INK, fontsize=13, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "la_ch05_robot_frame.png")


def plot_eigenvectors() -> None:
    a = np.array([[1.8, 0.45], [0.0, 0.75]])
    vectors = [np.array([1.0, 0.0]), np.array([0.7, 1.1]), np.array([0.0, 1.0])]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    setup_axis(axes[0], (-0.5, 2.2), (-0.4, 1.6))
    for vec, color, label in [(vectors[0], BLUE, r"$v_1$"), (vectors[1], ROSE, r"$u$"), (vectors[2], TEAL, r"$v_2$")]:
        arrow(axes[0], (0, 0), tuple(vec), color, label)
    axes[0].set_title("Before transformation", color=INK, fontsize=11, weight="bold")
    setup_axis(axes[1], (-0.5, 2.6), (-0.4, 1.6))
    for vec, color, label in [(vectors[0], BLUE, r"$Av_1$"), (vectors[1], ROSE, r"$Au$"), (vectors[2], TEAL, r"$Av_2$")]:
        mapped = a @ vec
        arrow(axes[1], (0, 0), tuple(mapped), color, label)
    axes[1].set_title("Eigenvectors keep their direction", color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch06_eigenvectors.png")


def plot_vibration_modes() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, title, arrows in [
        (axes[0], "mode 1: in phase", [(0.35, BLUE), (0.35, BLUE)]),
        (axes[1], "mode 2: out of phase", [(0.35, ROSE), (-0.35, ROSE)]),
    ]:
        ax.set_xlim(-0.2, 4.2)
        ax.set_ylim(-0.6, 1.0)
        ax.axis("off")
        ax.plot([0, 4], [0, 0], color=GRID, linewidth=1.4)
        for x0 in [1.25, 2.75]:
            ax.add_patch(Rectangle((x0 - 0.22, -0.15), 0.44, 0.30, facecolor=SURFACE, edgecolor=INK, linewidth=1.4))
        for x0, color in [(1.25, arrows[0][1]), (2.75, arrows[1][1])]:
            dx = arrows[0][0] if x0 == 1.25 else arrows[1][0]
            ax.annotate("", xy=(x0 + dx, 0.36), xytext=(x0, 0.36), arrowprops={"arrowstyle": "->", "lw": 2.4, "color": color})
        ax.plot([0.25, 1.03], [0, 0], color=AMBER, linewidth=3)
        ax.plot([1.47, 2.53], [0, 0], color=AMBER, linewidth=3)
        ax.plot([2.97, 3.75], [0, 0], color=AMBER, linewidth=3)
        ax.set_title(title, color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch06_vibration_modes.png")


def plot_projection() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    setup_axis(ax, (-0.5, 3.4), (-0.5, 2.8))
    t = np.linspace(-0.2, 3.3, 120)
    ax.plot(t, 0.55 * t, color=TEAL, linewidth=2.2, label="subspace")
    y = np.array([2.55, 2.25])
    direction = np.array([1.0, 0.55])
    proj = direction * (np.dot(y, direction) / np.dot(direction, direction))
    arrow(ax, (0, 0), tuple(y), ROSE, r"$y$")
    arrow(ax, (0, 0), tuple(proj), BLUE, r"$\hat y$")
    ax.plot([y[0], proj[0]], [y[1], proj[1]], linestyle="--", color=AMBER, linewidth=2)
    ax.text((y[0] + proj[0]) / 2 + 0.05, (y[1] + proj[1]) / 2, "residual", color=AMBER, fontsize=9, weight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Least squares as orthogonal projection", color=INK, fontsize=12, weight="bold")
    save(fig, "la_ch07_projection.png")


def plot_sensor_fit() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(0, 10, 18)
    y = 1.7 + 0.78 * x + rng.normal(0, 0.55, size=x.shape)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(0, 10, 200)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(x, y, color=BLUE, s=38, label="sensor samples")
    ax.plot(xx, coef[0] * xx + coef[1], color=ROSE, linewidth=2.4, label="least-squares calibration")
    ax.set_xlabel("temperature")
    ax.set_ylabel("voltage")
    ax.grid(True, color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Calibrating a physical sensor", color=INK, fontsize=12, weight="bold")
    save(fig, "la_ch07_sensor_fit.png")


def plot_quadratic_contours() -> None:
    x = np.linspace(-2.2, 2.2, 220)
    y = np.linspace(-2.2, 2.2, 220)
    xx, yy = np.meshgrid(x, y)
    z_pos = 2.2 * xx**2 + 0.8 * yy**2
    z_sad = xx**2 - yy**2
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for ax, z, title in [(axes[0], z_pos, "positive definite"), (axes[1], z_sad, "indefinite saddle")]:
        setup_axis(ax, (-2.2, 2.2), (-2.2, 2.2))
        ax.contour(xx, yy, z, levels=12, cmap="viridis")
        ax.set_title(title, color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch08_quadratic_contours.png")


def plot_energy_bowl() -> None:
    x = np.linspace(-2.0, 2.0, 160)
    y = np.linspace(-2.0, 2.0, 160)
    xx, yy = np.meshgrid(x, y)
    k = np.array([[2.0, -0.7], [-0.7, 1.2]])
    z = 0.5 * (k[0, 0] * xx**2 + 2 * k[0, 1] * xx * yy + k[1, 1] * yy**2)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    setup_axis(ax, (-2, 2), (-2, 2))
    contours = ax.contourf(xx, yy, z, levels=18, cmap="YlGnBu")
    ax.contour(xx, yy, z, levels=10, colors="white", linewidths=0.7, alpha=0.8)
    fig.colorbar(contours, ax=ax, shrink=0.82, label="elastic energy")
    ax.set_title(r"Spring energy $E=\frac{1}{2}x^T Kx$", color=INK, fontsize=12, weight="bold")
    save(fig, "la_ch08_energy_bowl.png")


def plot_svd_ellipse() -> None:
    theta = np.linspace(0, 2 * np.pi, 240)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    a = np.array([[1.7, 0.9], [0.3, 0.8]])
    ellipse = a @ circle
    u, s, _ = np.linalg.svd(a)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    setup_axis(axes[0], (-1.4, 1.4), (-1.4, 1.4))
    axes[0].plot(circle[0], circle[1], color=BLUE, linewidth=2.2)
    axes[0].set_title("unit circle", color=INK, fontsize=11, weight="bold")
    setup_axis(axes[1], (-2.4, 2.4), (-1.6, 1.8))
    axes[1].plot(ellipse[0], ellipse[1], color=TEAL, linewidth=2.2)
    arrow(axes[1], (0, 0), tuple(s[0] * u[:, 0]), ROSE, r"$\sigma_1 u_1$")
    arrow(axes[1], (0, 0), tuple(s[1] * u[:, 1]), AMBER, r"$\sigma_2 u_2$")
    axes[1].set_title("matrix maps circle to ellipse", color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch09_svd_ellipse.png")


def plot_condition_number() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    x = np.linspace(-0.1, 2.2, 200)
    for ax, delta, title in [(axes[0], 0.75, "well separated"), (axes[1], 0.08, "nearly parallel")]:
        setup_axis(ax, (-0.1, 2.2), (-0.1, 2.2))
        ax.plot(x, 0.65 * x + 0.3, color=BLUE, linewidth=2.2)
        ax.plot(x, (0.65 + delta) * x + 0.06, color=ROSE, linewidth=2.2)
        ax.plot(x, (0.65 + delta) * x + 0.14, color=ROSE, linewidth=1.3, linestyle="--", alpha=0.65)
        ax.set_title(title, color=INK, fontsize=11, weight="bold")
    save(fig, "la_ch09_condition_number.png")


def plot_pca_scatter() -> None:
    rng = np.random.default_rng(13)
    base = rng.normal(size=(180, 2))
    transform = np.array([[2.0, 1.1], [0.45, 0.7]])
    data = base @ transform.T
    centered = data - data.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    setup_axis(ax, (-6, 6), (-3.6, 3.6), equal=False)
    ax.scatter(centered[:, 0], centered[:, 1], s=16, color=BLUE, alpha=0.52)
    arrow(ax, (0, 0), tuple(3.8 * pc1), ROSE, "PC1")
    arrow(ax, (0, 0), tuple(-3.8 * pc1), ROSE, "")
    ax.set_title("PCA finds the most energetic direction", color=INK, fontsize=12, weight="bold")
    save(fig, "la_ch10_pca_scatter.png")


def plot_mass_spring_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    labels = [
        ("physical system", 1.2, BLUE),
        (r"$M\ddot x+C\dot x+Kx=f$", 3.5, TEAL),
        ("eigenmodes", 5.9, AMBER),
        ("simulation / control", 8.2, ROSE),
    ]
    for text, x0, color in labels:
        ax.add_patch(Rectangle((x0 - 0.9, 1.35), 1.8, 1.1, facecolor=SURFACE, edgecolor=color, linewidth=2))
        ax.text(x0, 1.9, text, ha="center", va="center", color=INK, fontsize=10, weight="bold")
    for x0 in [2.1, 4.4, 6.8]:
        ax.annotate("", xy=(x0 + 0.55, 1.9), xytext=(x0, 1.9), arrowprops={"arrowstyle": "->", "lw": 2.2, "color": MUTED})
    ax.text(0.45, 3.32, "real engineering problem -> matrices -> decomposition -> decisions", color=INK, fontsize=12, weight="bold")
    save(fig, "la_ch10_mass_spring_pipeline.png")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    plot_line_intersection()
    plot_force_equilibrium()
    plot_grid_transform()
    plot_circuit_linear_system()
    plot_determinant_area()
    plot_determinant_collapse()
    plot_span_basis()
    plot_incline_components()
    plot_coordinate_change()
    plot_robot_frame()
    plot_eigenvectors()
    plot_vibration_modes()
    plot_projection()
    plot_sensor_fit()
    plot_quadratic_contours()
    plot_energy_bowl()
    plot_svd_ellipse()
    plot_condition_number()
    plot_pca_scatter()
    plot_mass_spring_pipeline()


if __name__ == "__main__":
    main()
