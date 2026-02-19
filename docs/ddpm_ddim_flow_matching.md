#IM 与 Flow Matching DDPM、DD 的区别

本文档整理了三种主流生成式模型的原理、特点与差异:Denoising Diffusion Probabilistic Models (DDPM)、Denoising Diffusion Implicit Models (DDIM) 以及 Flow Matching。

---

## 1. 概述

这三种方法都属于扩散概率模型 (Diffusion Probabilistic Models) 家族,核心思想都是通过逐步添加噪声来破坏数据,然后学习逆向去噪过程来生成新样本。它们在图像、音频、视频生成等领域取得了显著的成果。

| 特性 | DDPM | DDIM | Flow Matching |
|------|------|------|---------------|
| 发表时间 | 2020 | 2021 | 2022 |
| 采样方式 | 随机 (Stochastic) | 确定性 (Deterministic) | 确定性 (ODE) |
| 采样速度 | 慢 (~1000步) | 快 (~20-50步) | 快 (~10-20步) |
| 数学框架 | 马尔可夫链 | 非马尔可夫链 | 常微分方程 (ODE) |
| 轨迹直线性 | 弯曲 | 较直 | 直线 |

---

## 2. DDPM (Denoising Diffusion Probabilistic Models)

### 2.1 核心思想

DDPM 借鉴了非平衡热力学 (Nonequilibrium Thermodynamics) 的思想,通过两个关键过程来构建生成模型:

- **前向扩散过程 (Forward Process)**: 逐步向数据添加噪声,直到数据完全变成高斯噪声
- **逆向去噪过程 (Reverse Process)**: 学习从噪声恢复到原始数据的逆过程

### 2.2 数学框架

**前向过程** 是一个马尔可夫链,以 T 步将数据 $x_0$ 逐步转换为噪声 $x_T$:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

其中 $\beta_t$ 是方差调度 (Variance Schedule),通常随着 t 增大而增大。

通过重参数化技巧,可以一次性采样任意时间步 t 的 $x_t$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{i=1}^{t} (1 - \beta_i)$。

**逆向过程** 同样是一个马尔可夫链,但需要学习:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

模型需要预测在每一步添加的噪声 $\epsilon$。

**损失函数** 使用简化的去噪损失:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

### 2.3 特点

- **优点**:
  - 理论基础扎实,训练稳定
  - 生成质量高 FID 分数优秀
  - 可以进行 likelihood 计算
  - 支持条件生成 (Classifier/Classifier-free Guidance)

- **缺点**:
  - 采样速度极慢,需要 1000 步甚至更多
  - 每步都需要完整的噪声预测,计算开销大
  - 随机采样导致结果不确定

---

## 3. DDIM (Denoising Diffusion Implicit Models)

### 3.1 核心思想

DDIM 的核心创新在于放宽了逆向过程必须严格遵循马尔可夫链的限制,允许使用非马尔可夫 (Non-Markovian) 的逆向过程。这使得采样过程可以更高效,同时保持相同的训练目标。

### 3.2 数学框架

DDIM 保留了与 DDPM 相同的训练目标,但重新定义了采样过程。关键在于:不改变训练过程,只改变采样方式。

**采样公式**:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t)$$

当使用 $\eta = 0$ 时,采样过程变成**确定性的**,这就是 DDIM 与 DDPM 的主要区别。

**关键参数**:
- $\eta$: 控制噪声的随机性
  - $\eta = 1$: 等价于 DDPM (完全随机)
  - $\eta = 0$: 完全确定性采样
- $S$: 采样子步数 (可以远小于 T)

### 3.3 特点

- **优点**:
  - 采样速度大幅提升 (10-50 步 vs 1000 步)
  - 支持更少的采样步数 (Skip Steps)
  - 确定性生成便于重建和编辑
  - 与 DDPM 训练完全兼容

- **缺点**:
  - 步数过少时质量可能下降
  - 仍然是隐式模型,不能直接计算 likelihood

---

## 4. Flow Matching

### 4.1 核心思想

Flow Matching (流匹配) 是一种基于常微分方程 (ODE) 的生成建模框架,它直接学习从噪声分布到数据分布的向量场 (Vector Field)。与扩散模型不同,Flow Matching 不需要 Monte Carlo 采样,而是通过求解 ODE 来生成样本。

### 4.2 数学框架

Flow Matching 的目标是学习一个向量场 $v_t(x)$,使得:

$$\frac{dx}{dt} = v_t(x)$$

从时间 $t=0$ (噪声) 到 $t=1$ (数据) 的积分给出生成样本:

$$x_1 = x_0 + \int_0^1 v_t(x_t) dt$$

**条件流匹配 (Conditional Flow Matching)**:

对于高斯路径,定义从噪声到数据的线性插值:

$$x_t = (1 - t) \cdot z + t \cdot x, \quad z \sim \mathcal{N}(0, \mathbf{I})$$

目标是让学习的向量场 $u_t(x|z, x_0)$ 匹配:

$$u_t(x_t | z, x_0) = x_0 - z$$

**损失函数**:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x, z} \left[ \| u_\theta(x_t, t) - (x - z) \|^2 \right]$$

### 4.3 典型实现: Rectified Flow

**Rectified Flow** (也称 Reflow) 是 Flow Matching 的一种重要变体,其核心思想是让轨迹尽可能直:

$$x_t = (1 - t) \cdot x_0 + t \cdot x_1$$

这意味着直接从数据点 $x_0$ 和噪声点 $x_1$ 构建直线轨迹,然后学习沿这些直线移动的速度场。

**优势**:
- 轨迹直,ODE 求解效率高
- 10 步以内即可获得高质量样本
- 已被 Flux 等前沿模型采用

### 4.4 特点

- **优点**:
  - 采样极快 (5-20 步)
  - 轨迹直,计算效率高
  - 确定性生成,便于编辑
  - 训练目标简单直接
  - 易于扩展到多模态

- **缺点**:
  - 训练可能比 DDPM 复杂
  - 理论框架相对较新
  - 与现有扩散生态系统兼容但不完全等价

---

## 5. 核心区别对比

### 5.1 采样机制

| 方面 | DDPM | DDIM | Flow Matching |
|------|------|------|---------------|
| 采样类型 | 随机 (SDE-like) | 确定性或混合 (ODE-like) | 确定性 (ODE) |
| 步数 | 1000+ | 20-50 | 5-20 |
| 轨迹 | 随机游走 | 较直 | 最直 |
| 可逆性 | 部分可逆 | 可逆 | 完全可逆 |

### 5.2 训练目标

**DDPM/DDIM**: 预测噪声 $\epsilon$
$$\mathcal{L} = \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

**Flow Matching**: 预测速度场 $v$
$$\mathcal{L} = \| v - v_\theta(x_t, t) \|^2$$

虽然形式不同,但两者在数学上被证明是等价的 (当使用特定的概率路径时)。

### 5.3 生成质量与速度

根据实际测试:

| 步数 | DDPM | DDIM | Flow Matching |
|------|------|------|---------------|
| 1000 步 | 最优 | - | - |
| 50 步 | 差 | 接近最优 | 最优 |
| 20 步 | 很差 | 中等 | 接近最优 |
| 10 步 | 不可用 | 差 | 接近最优 |

---

## 6. 适用场景

### DDPM 适用场景:
- 离线生成任务,对速度不敏感
- 需要精确 likelihood 计算
- 研究扩散模型理论基础
- 作为其他方法的基准

### DDIM 适用场景:
- 需要快速采样但保持高质量
- 需要确定性生成用于编辑/重建
- 生产环境部署
- 与现有扩散模型生态兼容

### Flow Matching 适用场景:
- 追求极致生成速度
- 大规模生成任务
- 需要直线轨迹进行编辑/控制
- 新一代生成模型 (如 Flux, Stable Diffusion 3)

---

## 7. 总结

DDPM、DDIM 和 Flow Matching 代表了扩散概率模型家族的三代演进:

1. **DDPM** 建立了理论基础,证明了扩散模型的可行性,但采样速度极慢
2. **DDIM** 通过非马尔可夫逆向过程大幅加速采样,是工业应用的主流选择
3. **Flow Matching** 用 ODE 框架重新诠释扩散,是新一代高速、高质量模型的基础

需要注意的是,最新研究证明 Flow Matching 和扩散模型在数学上是等价的,区别主要在于实现细节和采样策略。选择哪种方法取决于具体应用场景的需求。

---

## 8. 参考资料

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (Song et al., 2021)](https://arxiv.org/abs/2010.02502)
- [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [Rectified Flow (Liu et al., 2022)](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)
- [Diffusion Meets Flow Matching](https://diffusionflow.github.io/)

---

最后更新: 2026-02-19
