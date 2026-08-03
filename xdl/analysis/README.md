# xdl.analysis

`xdl.analysis/` 提供模型中间表征解析与可解释性分析的轻量工具,默认面向离线分析和研究脚本.

## 当前内容

- `activation.py`: 模块输出抓取,用于缓存中间激活
- `probe.py`: 线性 probe 拟合与评分
- `concept.py`: concept probe 与 TCAV 风格概念方向分析
- `grad_cam.py`: CNN 风格特征图的 Grad-CAM
- `attention.py`: 纯张量级 rollout 聚合,以及 ViT / CLIP 风格模型的 attention 抽取适配
- `report.py`: 结果写出 helpers,用于把 analysis 结果落成 JSON / CSV / Markdown

## 设计边界

- 这里放的是分析 API,不接管训练主逻辑
- 训练期如果需要抓特征,优先通过 callback 或 forward hook 采样后再调用这里的分析函数
- 首批能力以纯 PyTorch,无额外依赖为主,方便在 XDL 研究脚本中直接复用
- `concept.py` 当前提供的是二分类概念方向与 TCAV 风格方向导数分析,不是完整数据集或统计显著性工作流
- 如果你已经有模型,目标类别和模块名, 可以直接用 `compute_module_tcav(...)` 对单个 batch 计算概念方向导数和 TCAV 分数

## 什么时候用哪种

- 看"这一层已经能不能分出类别或属性": 用 `fit_linear_probe(...)`
- 看"某个概念是否已经出现在中间层": 用 `fit_concept_probe(...)` 或 `compute_module_tcav(...)`
- 看 CNN 路线的空间关注区域: 用 `compute_grad_cam(...)`
- 看 ViT / CLIP 路线 token 到 patch 的聚合: 用 `attention_rollout_for_model(...)`
- 需要把结果交给别的脚本或人工阅读: 用 `write_analysis_bundle(...)`

这几个能力是并列的,不要把它们当成同一种"透镜". `probe` 更偏读出, `Grad-CAM` 和 `attention rollout` 更偏空间归因, `TCAV` 更偏概念方向分析.

## 使用示例

```python
import torch
from xdl.analysis import (
    capture_activations,
    compute_grad_cam,
    fit_concept_probe,
    fit_linear_probe,
    tcav_from_probe,
)

records = capture_activations(model, ["backbone.layer4"], images)
features = records["backbone.layer4"].value

probe = fit_linear_probe(features.flatten(1), labels)
concept_probe = fit_concept_probe(features.flatten(1), concept_labels)
cam = compute_grad_cam(model, "backbone.layer4", images)
score = tcav_from_probe(feature_gradients, concept_probe)
```

如果你需要把结果交给别的脚本或人工阅读, 直接用 `write_analysis_bundle(...)` 即可, 不需要引入额外工作流层.

```python
from xdl.analysis import write_analysis_bundle

outputs = write_analysis_bundle(
    "runs/analysis",
    report_name="vit_probe",
    summary={
        "probe_accuracy": 0.91,
        "tcav_score": 0.74,
    },
    records=[
        {"layer": "blocks.0", "probe_accuracy": 0.68},
        {"layer": "blocks.11", "probe_accuracy": 0.91},
    ],
    markdown_sections={
        "summary": {
            "best_layer": "blocks.11",
            "note": "cls token probe is stable after middle layers",
        }
    },
)
```
