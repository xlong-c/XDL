# probability_statistics todo

最后更新: 2026-06-10

## 约定

- 章节文件命名: `chapterNN_*.md`
- 配图目录: `learn/math/probability_statistics/assets/`
- 每章完成后执行 3 步:
  1. 生成图片
  2. 检查图片路径和文件存在性
  3. 复核公式, 定义, 例题结论是否自洽

## 当前进度

- [x] `outline.md` 已整理为总纲
- [x] Chapter 01: `chapter01_random_phenomena_probability.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 02: `chapter02_conditional_probability_independence_bayes.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 03: `chapter03_random_variables_common_distributions.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 04: `chapter04_multivariate_random_variables_transformations.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 05: `chapter05_expectation_variance_entropy_information.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 06: `chapter06_limit_theorems_concentration_processes.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 07: `chapter07_population_sample_statistic_sampling_distribution.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 08: `chapter08_point_interval_likelihood_estimation.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 09: `chapter09_hypothesis_testing_p_value_power_experiment.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 10: `chapter10_regression_anova_glm_intro.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 11: `chapter11_nonparametric_resampling_robust_methods.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 12: `chapter12_bayesian_prior_posterior_conjugacy.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 13: `chapter13_bayesian_inference_map_mcmc_vi.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 14: `chapter14_bayesian_regression_hierarchical_uncertainty.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 15: `chapter15_statistical_learning_risk_generalization.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 16: `chapter16_supervised_learning_regularization_kernel_methods.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 17: `chapter17_probabilistic_latent_variable_models_em.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 18: `chapter18_probability_statistics_view_of_deep_learning.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成
- [x] Chapter 19: `chapter19_evaluation_selection_decision_workflow.md`
  - [x] 正文完成
  - [x] 配图脚本完成
  - [x] 生成与自检完成

## 第一章自检记录

- 图片已生成:
  - `assets/ps_ch01_event_operations.png`
  - `assets/ps_ch01_frequency_stabilization.png`
  - `assets/ps_ch01_classical_tree.png`
  - `assets/ps_ch01_geometric_probability.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 古典概型适用条件: 有限且等可能
  - 几何概型示例: 单位正方形内四分之一圆概率为 `pi / 4`
  - 加法公式: `P(A ∪ B) = P(A) + P(B) - P(A ∩ B)`

## 第二章自检记录

- 图片已生成:
  - `assets/ps_ch02_conditional_probability.png`
  - `assets/ps_ch02_bayes_update.png`
  - `assets/ps_ch02_independence_heatmaps.png`
  - `assets/ps_ch02_conditional_independence.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 医学筛查后验概率 `0.161016949...`, 约为 `16.1%`
  - 两两独立反例中 `P(A∩B∩C)=1/4`, 而 `P(A)P(B)P(C)=1/8`, 因此不相互独立
  - 条件独立数值例子中 `P(F∩C)=0.074`, 而 `P(F)P(C)=0.0425`, 因此整体上不独立

## 第三章自检记录

- 图片已生成:
  - `assets/ps_ch03_random_variable_mapping.png`
  - `assets/ps_ch03_cdf_pmf_pdf_alignment.png`
  - `assets/ps_ch03_binomial_poisson_approx.png`
  - `assets/ps_ch03_normal_family.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 两次抛硬币示例的 PMF 为 `(0.25, 0.5, 0.25)`, 累积 CDF 为 `(0.25, 0.75, 1.0)`
  - `Binomial(4, 0.5)` 概率和为 `1.0`
  - `Binomial(50, 0.08)` 与 `Poisson(4)` 前几项概率接近, 近似图无异常
  - Negative Binomial 在本章采用"第 r 次成功出现时的总试验次数"定义, `r=1` 时与 Geometric 前几项一致

## 第四章自检记录

- 图片已生成:
  - `assets/ps_ch04_joint_table_heatmap.png`
  - `assets/ps_ch04_marginal_projection.png`
  - `assets/ps_ch04_conditional_slice.png`
  - `assets/ps_ch04_bivariate_normal_contours.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 联合 PMF 示例总和为 `1.0`
  - 示例边缘分布为 `P_X=(0.20, 0.45, 0.35)`, `P_Y=(0.28, 0.45, 0.27)`
  - 变换 `U=X+Y, V=X-Y` 的反变换 Jacobian 绝对行列式为 `0.5`
  - 边缘化投影图标注已复查, 无明显裁切

## 第五章自检记录

- 图片已生成:
  - `assets/ps_ch05_same_mean_different_variance.png`
  - `assets/ps_ch05_covariance_ellipse_axes.png`
  - `assets/ps_ch05_entropy_comparison.png`
  - `assets/ps_ch05_cross_entropy_kl_relation.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 交叉熵与 KL 示例满足 `H(p,q)=H(p)+KL(p||q)`, 其中 `H(p)=1.228972... bits`
  - good q 示例中 `CE-H=0.024199...`, 与 `KL=0.024199...` 一致
  - bad q 示例中 `CE-H=1.034459...`, 与 `KL=1.034459...` 一致
  - 协方差矩阵示例特征值为正, 约 `(0.43795, 3.56205)`, 符合半正定直觉

## 第六章自检记录

- 图片已生成:
  - `assets/ps_ch06_sample_mean_convergence.png`
  - `assets/ps_ch06_clt_standardized_histogram.png`
  - `assets/ps_ch06_concentration_bounds.png`
  - `assets/ps_ch06_markov_chain_states.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - Bernoulli 样本均值图围绕真实均值 `p=0.6` 收敛
  - CLT 图使用 Bernoulli(`p=0.35`) 和的标准化, `n=5,30,120` 的目标方差分别为 `1.1375, 6.825, 27.3`
  - Hoeffding 原始上界已在绘图中截断到不超过 `1.0`, 截断后单调下降
  - Markov 链示例转移矩阵每行和为 `1`

## 第七章自检记录

- 图片已生成:
  - `assets/ps_ch07_sampling_flow.png`
  - `assets/ps_ch07_empirical_cdf.png`
  - `assets/ps_ch07_sample_mean_distribution.png`
  - `assets/ps_ch07_classical_sampling_distributions.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 样本均值满足 `E[bar X]=mu`, `Var(bar X)=sigma^2/n`
  - 样本方差估计总体方差时分母用 `n-1`, 自由度来源已核对
  - 正态总体下 `((n-1)S^2)/sigma^2 ~ Chi-square(n-1)` 和 `(bar X-mu)/(S/sqrt(n)) ~ t(n-1)` 已核对
  - Chi-square/t/F helper 代表点已校验: `0.18393972058572117`, `0.3956321848940978`, `0.5448781251821823`

## 第八章自检记录

- 图片已生成:
  - `assets/ps_ch08_estimator_variability.png`
  - `assets/ps_ch08_likelihood_curve.png`
  - `assets/ps_ch08_confidence_interval_coverage.png`
  - `assets/ps_ch08_sample_size_interval_width.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 正态均值已知方差情形 MLE 为样本均值, 图中样本均值为 `5.08`
  - `n=25, sigma=1` 的 95% 均值区间半宽为 `1.96/sqrt(25)=0.392`
  - 置信区间覆盖模拟使用固定随机种子, 覆盖计数为 `77/80`
  - 样本量与区间宽度图使用总宽度 `2*1.96*sigma/sqrt(n)`, 符合 `1/sqrt(n)` 规律
  - 正文已区分频率学置信区间的长期覆盖率解释和贝叶斯后验概率解释

## 第九章自检记录

- 图片已生成:
  - `assets/ps_ch09_rejection_regions.png`
  - `assets/ps_ch09_type_errors_power.png`
  - `assets/ps_ch09_p_value_tail.png`
  - `assets/ps_ch09_ab_test_flow.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 双侧检验 `z=2.2` 的 p 值约为 `0.02780689502699718`
  - 单侧拒绝域阈值 `z=1.645` 的右尾概率约为 `0.04998490553912138`
  - 第一类错误, 第二类错误和功效图中 `beta≈0.5967717843205245`, `power≈0.4032282156794755`
  - 20 次独立检验且单次 `alpha=0.05` 时至少一次误报概率约为 `0.6415140775914581`
  - 正文已核对 p 值不是 `P(H0|data)`, 不拒绝 `H0` 不等于证明 `H0` 成立

## 第十章自检记录

- 图片已生成:
  - `assets/ps_ch10_linear_fit.png`
  - `assets/ps_ch10_residual_diagnostics.png`
  - `assets/ps_ch10_variance_decomposition.png`
  - `assets/ps_ch10_logistic_curve.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - OLS 模拟数据截距约为 `1.5263293517548762`, 斜率约为 `0.7870255176406068`
  - 方差分解满足 `SST≈SSR+SSE`, 浮点残差约为 `-4.831690603168681e-13`
  - 回归示例 `R^2≈0.8204068947750732`, 残差均值约为 `-4.1871268357291617e-16`
  - 正文已核对线性回归预测/解释边界, 残差诊断, ANOVA 平方和分解和 Logistic 回归的 Bernoulli 似然/交叉熵关系

## 第十一章自检记录

- 图片已生成:
  - `assets/ps_ch11_bootstrap_flow.png`
  - `assets/ps_ch11_rank_vs_parametric.png`
  - `assets/ps_ch11_outlier_mean_median.png`
  - `assets/ps_ch11_heavy_tail_distribution.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 异常值敏感性示例中 outlier 从 `0` 到 `18` 时均值从约 `0.13975188415421774` 增至约 `0.7022518841542178`, 中位数稳定在约 `0.2755950577705338`
  - 秩示例中 A/B 组原始均值约为 `0.9886808275164723` 和 `1.529449181905301`, 平均秩约为 `20.083333333333332` 和 `28.916666666666668`
  - Bootstrap 流程已核对为有放回重采样, 置换检验已强调可交换性前提
  - 正文已核对非参数方法不是无假设, Bootstrap 不能修复原始样本偏差, 异常值不应自动删除

## 第十二章自检记录

- 图片已生成:
  - `assets/ps_ch12_prior_to_posterior_update.png`
  - `assets/ps_ch12_prior_influence.png`
  - `assets/ps_ch12_conjugate_update_flow.png`
  - `assets/ps_ch12_posterior_predictive.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - Beta-Binomial 示例先验 `Beta(2,2)`, 数据 `13/18`, 后验为 `Beta(15,7)`
  - 后验均值约为 `0.6818181818181818`, 后验众数约为 `0.7`
  - 未来 20 次试验的 Beta-Binomial 后验预测分布概率和约为 `1.0000000000000044`, 预测均值约为 `13.636363636363685`
  - Gamma-Poisson 示例若先验为 `Gamma(3,2)`, 总计数 `27`, 曝光量 `10`, 后验为 `Gamma(30,12)`
  - 正文已核对可信区间和置信区间解释差异, 共轭先验只是计算便利而非唯一合理选择

## 第十三章自检记录

- 图片已生成:
  - `assets/ps_ch13_mle_map_comparison.png`
  - `assets/ps_ch13_mcmc_trace_histogram.png`
  - `assets/ps_ch13_vi_approximation.png`
  - `assets/ps_ch13_elbo_decomposition.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - MLE/MAP 图中 MLE 网格值约为 `1.1012875536480684`, MAP 网格值约为 `0.7948497854077252`
  - 双峰目标后验密度数值积分为 `1.0`
  - Metropolis 随机游走示例接受率约为 `0.6895379075815163`, burn-in 后样本均值约为 `-0.0013468839447031396`
  - 已将 `np.trapz` 兼容性问题修为 `np.trapezoid`
  - 正文已核对 MAP 只是点估计, MCMC 需要混合/收敛诊断, VI 是优化得到的近似分布

## 第十四章自检记录

- 图片已生成:
  - `assets/ps_ch14_bayesian_regression_band.png`
  - `assets/ps_ch14_hierarchical_partial_pooling.png`
  - `assets/ps_ch14_calibration_curve.png`
  - `assets/ps_ch14_uncertainty_decomposition.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 贝叶斯线性回归示例后验均值约为 `[0.7657445861739438, 1.2362575108859224]`
  - 后验协方差对角线约为 `[0.013729025100540842, 0.004182837690204435]`
  - `x=0` 的 epistemic sd 约为 `0.11717092258978266`, `x=±5` 的 epistemic sd 约为 `0.3439476229829939`, 数据范围外不确定性更大
  - 校准示例中 calibrated model 的 Brier score 约为 `0.20541724724467741`, overconfident model 约为 `0.21326139507132888`
  - 正文已核对 epistemic uncertainty 和 aleatoric uncertainty 的来源差异, 层次模型部分池化和校准曲线解释

## 第十五章自检记录

- 图片已生成:
  - `assets/ps_ch15_train_test_error_curve.png`
  - `assets/ps_ch15_bias_variance_demo.png`
  - `assets/ps_ch15_learning_curve.png`
  - `assets/ps_ch15_complexity_generalization_risk.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 训练/测试误差曲线中最佳测试复杂度为 `9`, 最小测试误差约为 `0.2418849561847101`
  - 学习曲线 generalization gap 从约 `0.3530779086596658` 降到约 `0.08635358824082803`
  - 泛化风险分解图中最佳复杂度约为 `4.153846153846154`, 最小风险约为 `0.45888527415125585`
  - 正文已核对经验风险与期望风险差异, 训练/验证/测试集角色, 偏差-方差权衡和正则化与 MAP 的连接

## 第十六章自检记录

- 图片已生成:
  - `assets/ps_ch16_ridge_lasso_geometry.png`
  - `assets/ps_ch16_svm_margin.png`
  - `assets/ps_ch16_decision_tree_splits.png`
  - `assets/ps_ch16_decision_boundaries_comparison.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - Ridge/Lasso 几何图中 Ridge 示例点 L2 范数约为 `1.549322432549145`, Lasso 示例点 L1 范数为 `1.55`
  - SVM 间隔图中最近样本到边界距离约为 `0.005478174480300924`, 6 个近边界点平均距离约为 `0.027088566441571663`
  - 决策树切分示例类别数为 class 0: `106`, class 1: `74`
  - 正文已核对 L1/L2 正则化几何差异, SVM hinge loss 和最大间隔, 核技巧, 树模型与集成方法的偏差/方差角色

## 第十七章自检记录

- 图片已生成:
  - `assets/ps_ch17_generative_discriminative_flow.png`
  - `assets/ps_ch17_gmm_mixture_density.png`
  - `assets/ps_ch17_em_iteration.png`
  - `assets/ps_ch17_hmm_state_transition.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - GMM 样本均值约为 `-0.3032487325043373`, 样本标准差约为 `1.4674045042729469`
  - EM 最终估计约为 `pi1=0.6432176342447352`, `mu1=-1.320290417260681`, `sigma1=0.46607982205150117`, `mu2=1.5303036209379013`, `sigma2=0.6325798539667804`
  - 混合密度数值积分约为 `0.9999999999609429`, 符合概率密度归一化要求
  - 四张图已视觉抽查, 标注清晰, 无明显裁切或遮挡
  - 正文已核对生成式/判别式模型差异, 朴素贝叶斯平滑, 潜变量建模, EM 的 E-step/M-step 分工, HMM 与概率图模型初步

## 第十八章自检记录

- 图片已生成:
  - `assets/ps_ch18_loss_probability_assumptions.png`
  - `assets/ps_ch18_calibration_confidence.png`
  - `assets/ps_ch18_vae_elbo_structure.png`
  - `assets/ps_ch18_diffusion_denoising_process.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - MSE 对应高斯噪声负对数似然, MAE 对应 Laplace 噪声负对数似然
  - BCE 对应 Bernoulli 似然, Cross Entropy 对应 Categorical 似然
  - 校准图中 overconfident 模型 `ECE≈0.11782968622716511`, `Brier≈0.2105883273005794`
  - 校准图中 better calibrated 模型 `ECE≈0.019409433038975365`, `Brier≈0.19706525356545695`
  - 图片已视觉抽查, 损失对照, 校准曲线, VAE 结构和 Diffusion 去噪流程均无明显裁切或遮挡
  - 正文已核对 Softmax 概率解释与校准差异, 正则化/MAP 直觉, BatchNorm 批统计量, SGD 噪声, 不确定性估计, VAE ELBO, Diffusion 去噪概率过程和互信息视角

## 第十九章自检记录

- 图片已生成:
  - `assets/ps_ch19_real_ml_workflow.png`
  - `assets/ps_ch19_data_drift.png`
  - `assets/ps_ch19_threshold_metrics.png`
  - `assets/ps_ch19_error_analysis_loop.png`
- Markdown 图片相对路径已检查, 无缺失文件
- 关键结论已复核:
  - 数据漂移示例中 training feature 均值约为 `0.023037710077359826`, 标准差约为 `1.0088356966731202`
  - 数据漂移示例中 serving feature 均值约为 `0.8461625813071874`, 标准差约为 `1.1343476123970657`, `PSI≈0.5793746232629574`
  - 阈值示例中正类比例为 `0.325`, 最小成本阈值约为 `0.41999999999999993`, 最小成本约为 `0.193125`
  - 阈值示例中最优 F1 阈值约为 `0.57`, 最优 F1 约为 `0.8415741675075682`
  - 图片已视觉抽查, 工作流图, 漂移图, 阈值图和错误分析闭环图均无明显裁切或遮挡
  - 正文已核对样本偏差, 数据泄漏, train/val/test 分工, 交叉验证, 指标选择, 成本敏感阈值, 校准, 模型比较, 线上实验, 漂移监控, 因果意识, 可重复性和错误分析流程

## 全局检查记录

- Chapter 文件数量已检查: `19`
- 配图脚本数量已检查: `19`
- 章节图片数量已检查: `76`
- Markdown 图片引用已检查: `76`, 缺失文件数 `0`
- `todo.md` 完成状态已检查: Chapter 01 到 Chapter 19 均为 `[x]`, 未完成项 `0`
- `assets/gen_chapter??_figs.py` 已全部通过 `python -m py_compile`
- 章节正文, 配图脚本和 `todo.md` 已检查常见全角标点, 无残留匹配项

## 下一步

- 第一轮 Chapter 01 到 Chapter 19 主线教程已完成. 后续可按 `outline.md` 的高级专题和附录建议继续扩展
