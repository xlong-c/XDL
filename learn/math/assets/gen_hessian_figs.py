"""生成 Hessian.md 配图(图内英文标注,保证跨环境不乱码)。

运行:python learn/math/assets/gen_hessian_figs.py
输出:learn/math/assets/*.png
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

OUT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 120, "font.size": 11, "axes.titlesize": 12})


def fig_critical_points() -> None:
    """三类临界点曲面:极小(正定)/极大(负定)/鞍点(不定)。"""
    g = np.linspace(-2, 2, 80)
    X, Y = np.meshgrid(g, g)
    cases = [
        ("Minimum  (H positive-definite)", X**2 + Y**2, "eig: (+, +)"),
        ("Maximum  (H negative-definite)", -(X**2 + Y**2), "eig: (-, -)"),
        ("Saddle  (H indefinite)", X**2 - Y**2, "eig: (+, -)"),
    ]
    fig = plt.figure(figsize=(13, 4.2))
    for i, (title, Z, tag) in enumerate(cases, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.9, linewidth=0)
        ax.scatter([0], [0], [0], color="red", s=40)
        ax.set_title(f"{title}\n{tag}")
        ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("f")
        ax.view_init(elev=28, azim=-60)
    fig.tight_layout()
    fig.savefig(OUT / "critical_points.png", bbox_inches="tight")
    plt.close(fig)


def fig_quadratic_contours() -> None:
    """二次型等高线 + 海森特征向量(主曲率方向)。"""
    A = np.array([[3.0, 1.0], [1.0, 2.0]])      # 对称正定
    eigval, eigvec = np.linalg.eigh(A)          # 升序
    g = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(g, g)
    Z = 0.5 * (A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    cs = ax.contour(X, Y, Z, levels=12, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7)
    # 特征向量:长轴对应小特征值(曲率小,等高线稀疏方向)
    colors = ["#d62728", "#1f77b4"]
    for k in range(2):
        vec = eigvec[:, k]
        scale = 2.4 / np.sqrt(eigval[k])
        ax.annotate("", xy=vec * scale, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=colors[k], lw=2.2))
        ax.text(*(vec * scale * 1.08),
                f"v{k+1}, lam={eigval[k]:.2f}", color=colors[k], fontsize=9)
    ax.scatter([0], [0], color="black", s=25, zorder=5)
    ax.set_title("Quadratic form  f = 0.5 xᵀA x\n"
                 "Hessian H = A;  eigenvectors = principal curvature axes")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "quadratic_contours.png", bbox_inches="tight")
    plt.close(fig)


def fig_rosenbrock() -> None:
    """Rosenbrock 等高线 + 全局极小点 (1,1)。"""
    g_x = np.linspace(-2, 2, 400)
    g_y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(g_x, g_y)
    Z = (1 - X) ** 2 + 100 * (Y - X**2) ** 2

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    cs = ax.contourf(X, Y, np.log10(Z + 1e-6), levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log10(f)")
    ax.scatter([1], [1], color="red", s=60, marker="*",
               zorder=5, label="global min (1,1)")
    ax.scatter([0.5], [0.5], color="white", s=35, marker="o",
               edgecolor="black", zorder=5, label="sample (0.5,0.5)")
    ax.set_title("Rosenbrock landscape (log scale)\n"
                 "curved valley → Hessian is ill-conditioned")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "rosenbrock.png", bbox_inches="tight")
    plt.close(fig)


def fig_sharpness() -> None:
    """平坦极小 vs 尖锐极小:曲率 = 海森(最大)特征值,关联泛化。"""
    x = np.linspace(-2, 2, 400)
    flat = 0.5 * x**2          # 小曲率
    sharp = 6.0 * x**2         # 大曲率

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(x, flat, lw=2.4, label="flat  min  (small H eig → good generalization)")
    ax.plot(x, sharp, lw=2.4, label="sharp min  (large H eig → sensitive)")
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.annotate("curvature = f''(0) = Hessian eigenvalue",
                xy=(0, 0), xytext=(0.2, 6),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    ax.set_ylim(-1, 12); ax.set_xlabel("parameter direction"); ax.set_ylabel("loss")
    ax.set_title("Loss curvature: flat vs sharp minima")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sharpness.png", bbox_inches="tight")
    plt.close(fig)


def fig_optimization() -> None:
    """病态二次型上:普通梯度下降(zigzag)vs 用海森预条件(牛顿,一步直达)。

    说明工程上"为什么需要曲率信息"——一阶法在病态曲面震荡,
    海森把各方向步长按曲率归一化,直达极小。
    """
    A = np.array([[20.0, 0.0], [0.0, 1.0]])     # 条件数 20,长条椭圆
    g = np.linspace(-1.2, 1.2, 300)
    gx, gy = np.meshgrid(g, g)
    Z = 0.5 * (A[0, 0] * gx**2 + A[1, 1] * gy**2)

    def grad(p):
        return A @ p

    x0 = np.array([1.0, 1.0])

    # 普通梯度下降:步长受最大特征值约束,小特征值方向极慢 -> zigzag
    eta = 1.8 / 20.0
    p = x0.copy(); gd = [p.copy()]
    for _ in range(40):
        p = p - eta * grad(p)
        gd.append(p.copy())
    gd = np.array(gd)

    # 牛顿步:Δ = -H⁻¹ g,二次型一步到极小
    Hinv = np.linalg.inv(A)
    newton = np.array([x0, x0 - Hinv @ grad(x0)])

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    cs = ax.contour(gx, gy, Z, levels=18, cmap="Greys", alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=6)
    ax.plot(gd[:, 0], gd[:, 1], "-o", color="#ff7f0e", ms=3, lw=1.4,
            label=f"Gradient descent (1st order): {len(gd)-1} steps, zigzag")
    ax.plot(newton[:, 0], newton[:, 1], "-o", color="#d62728", ms=7, lw=2.2,
            label="Newton step  Δ = -H⁻¹∇f : 1 step")
    ax.scatter([0], [0], marker="*", s=160, color="green", zorder=6, label="optimum")
    ax.set_title("Why curvature matters: ill-conditioned bowl (cond=20)\n"
                 "1st-order zigzags; Hessian rescales each axis → direct hit")
    ax.set_xlabel("x1 (steep, large eig)"); ax.set_ylabel("x2 (flat, small eig)")
    ax.legend(loc="upper right", fontsize=8); ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "optimization_trajectory.png", bbox_inches="tight")
    plt.close(fig)


def fig_engineering_paths() -> None:
    """7.3 三条工程化路径决策图(中文大字号 PNG,替代 Mermaid)。

    PNG 里文字已栅格化,查看时不依赖字体,故可放心用中文。
    生成时用 WenQuanYi 渲染中文。
    """
    from matplotlib.patches import FancyBboxPatch

    cfg = {
        "font.sans-serif": ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei"],
        "axes.unicode_minus": False,
    }
    with plt.rc_context(cfg):
        fig, ax = plt.subplots(figsize=(12, 7.4))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        def box(cx, cy, w, h, text, fc, ec, fs=14, bold=False):
            ax.add_patch(FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=fc, edgecolor=ec, linewidth=2.2))
            ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", color="#202124")

        def arrow(x0, y0, x1, y1):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#5f6368", lw=2.4))

        # 顶层:不可行的全海森
        box(0.5, 0.88, 0.58, 0.135,
            "完整海森 H  (N×N)\nO(N²) 存储 · O(N³) 求逆 —— 不可行",
            "#fce8e6", "#d62728", fs=15, bold=True)

        px = [0.17, 0.5, 0.83]
        paths = ["① 只要 H·v\n\nHVP:两次反向传播",
                 "② 只要结构\n\n对角 / 块对角 / Kronecker",
                 "③ 只要局部\n\n逐层小海森 $XX^{\\top}$"]
        for x, t in zip(px, paths):
            box(x, 0.54, 0.30, 0.17, t, "#e8f0fe", "#4285f4", fs=13.5)

        apps = ["二阶优化器\n学习率上界 2/λmax",
                "K-FAC / Shampoo / Adam\n预条件",
                "GPTQ / OBS\n量化 · 剪枝"]
        for x, t in zip(px, apps):
            box(x, 0.16, 0.30, 0.155, t, "#e6f4ea", "#34a853", fs=13.5)

        for x in px:
            arrow(0.5, 0.812, x, 0.628)     # H -> 路径
            arrow(x, 0.452, x, 0.242)       # 路径 -> 应用

        ax.set_title("海森的三条工程化路径:按需取曲率,从不构造全矩阵",
                     fontsize=16, fontweight="bold", pad=14)
        fig.tight_layout()
        fig.savefig(OUT / "engineering_paths.png", bbox_inches="tight")
    plt.close(fig)


def fig_taylor_approx() -> None:
    """一阶(切平面)vs 二阶(抛物面)泰勒逼近对比 —— 为什么需要二阶。

    左/中:同一曲面 f(x,y)=sin(x)cos(y) 上叠加切平面 / 抛物面;
    右:沿 y=y0 的截面,真实 vs 切线 vs 抛物线的贴合程度。
    """
    x0, y0 = 0.6, 0.4

    def f(x, y):
        return np.sin(x) * np.cos(y)

    f0 = np.sin(x0) * np.cos(y0)
    fx = np.cos(x0) * np.cos(y0)
    fy = -np.sin(x0) * np.sin(y0)
    fxx = -np.sin(x0) * np.cos(y0)
    fyy = -np.sin(x0) * np.cos(y0)
    fxy = -np.cos(x0) * np.sin(y0)

    r = 1.6
    g = np.linspace(-r, r, 60)
    DX, DY = np.meshgrid(g, g)
    X, Y = x0 + DX, y0 + DY
    Z = f(X, Y)
    Z_lin = f0 + fx * DX + fy * DY                                   # 一阶:切平面
    Z_quad = Z_lin + 0.5 * (fxx * DX**2 + 2 * fxy * DX * DY + fyy * DY**2)  # 二阶:抛物面

    fig = plt.figure(figsize=(15.5, 5.2))

    for idx, (Zap, col, name, sub) in enumerate([
        (Z_lin, "#d62728", "1st-order: tangent plane (flat)",
         "fits slope only"),
        (Z_quad, "#1f77b4", "2nd-order: paraboloid (curved)",
         r"adds Hessian term $\frac{1}{2}\delta^\top H\delta$"),
    ], 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.55, linewidth=0)
        ax.plot_wireframe(X, Y, Zap, color=col, rstride=8, cstride=8, lw=0.9)
        ax.scatter([x0], [y0], [f0], color="black", s=35)
        ax.set_title(f"{name}\n{sub}", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f")
        ax.view_init(elev=22, azim=-58)

    # 截面:沿 y=y0
    ax3 = fig.add_subplot(1, 3, 3)
    lx = np.linspace(-r, r, 240)
    ax3.plot(x0 + lx, f(x0 + lx, y0), "k", lw=2.6, label="true  f")
    ax3.plot(x0 + lx, f0 + fx * lx, "--", color="#d62728", lw=2,
             label="1st-order (tangent line)")
    ax3.plot(x0 + lx, f0 + fx * lx + 0.5 * fxx * lx**2, "--",
             color="#1f77b4", lw=2, label="2nd-order (parabola)")
    ax3.axvline(x0, color="gray", ls=":", lw=1)
    ax3.scatter([x0], [f0], color="black", zorder=5)
    ax3.set_title("Cross-section at y = y0\nparabola tracks the bend, line drifts off",
                  fontsize=11)
    ax3.set_xlabel("x"); ax3.set_ylabel("f"); ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "taylor_approx.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_critical_points()
    fig_quadratic_contours()
    fig_rosenbrock()
    fig_sharpness()
    fig_optimization()
    fig_engineering_paths()
    fig_taylor_approx()
    print("saved:", sorted(p.name for p in OUT.glob("*.png")))
