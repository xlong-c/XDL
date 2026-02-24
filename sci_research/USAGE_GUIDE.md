# 科研工具 Skill 使用指南

> 完整的研究文献分析与管理工作流

**版本**: v0.3.0  
**更新日期**: 2026-02-24

---

## 📦 已实现功能

| Phase | Skill | 状态 | 功能 |
|-------|-------|------|------|
| **Phase 1** | `dblp_analyzer` | ✅ 完成 | DBLP 论文分析 + CSV 导出 |
| **Phase 2** | `abstract_fetcher` | ✅ 完成 | Crossref/arXiv 批量摘要获取 |
| **Phase 2** | `precision_classifier` | ✅ 完成 | 基于摘要的精确分类 |
| **Phase 3** | `paper_extractor` | ⏳ 待开发 | 关键信息提取 |
| **Phase 3** | `deep_analysis_report` | ⏳ 待开发 | 深入分析报告 |
| **Phase 4** | `zotero_dual_sync` | ⏳ 待开发 | Zotero 双目录管理 |

---

## 🚀 快速开始

### 完整工作流 (TIP 专刊分析示例)

```bash
# 步骤 1: DBLP 分析 + CSV 导出
python -m skills.dblp_analyzer \
  --url "https://dblp.org/db/journals/tip/tip33.html" \
  --keywords "segmentation" "efficient" "mobile" \
  --exclude-keywords "survey" "review" \
  --format csv \
  --output tip33_raw.csv

# 步骤 2: 批量获取摘要
python -m skills.abstract_fetcher \
  --input tip33_raw.csv \
  --output tip33_with_abstracts.csv \
  --id-column doi

# 步骤 3: 精确分类
python -m skills.precision_classifier \
  --input tip33_with_abstracts.csv \
  --output tip33_classified.csv \
  --taxonomy config/tip_segmentation.yaml \
  --output-dir tip33_analysis/ \
  --min-score 1.0
```

**输出**:
- `tip33_raw.csv` - 255 篇论文基本信息
- `tip33_with_abstracts.csv` - 含摘要 (成功率>90%)
- `tip33_classified.csv` - 精确分类结果
- `tip33_analysis/` - 按类别导出的 Markdown 文件

---

## 📋 详细使用说明

### 1. DBLP 论文分析器 (`dblp_analyzer`)

**功能**: 解析 DBLP 页面，提取论文元数据，按规则分类

**命令行**:
```bash
python -m skills.dblp_analyzer [选项]
```

**常用选项**:
| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--url` | DBLP 页面 URL (必需) | - |
| `--output` | 输出文件路径 | `dblp_analysis.md` |
| `--format` | 输出格式：markdown/csv/json | `markdown` |
| `--keywords` | 标题必须包含的关键词 | - |
| `--exclude-keywords` | 标题必须排除的关键词 | - |
| `--min-year` | 最小年份 | - |
| `--max-year` | 最大年份 | - |
| `--top-n` | 每个领域显示前 N 篇 | 全部 |

**示例**:
```bash
# 分析 CVPR 2024，只看 Backbone 相关，排除综述
python -m skills.dblp_analyzer \
  --url "https://dblp.org/db/conf/cvpr/cvpr2024.html" \
  --keywords "backbone" "resnet" "transformer" \
  --exclude-keywords "survey" "review" \
  --format csv \
  --output cvpr2024_backbone.csv
```

**CSV 输出格式**:
```csv
category,title,authors,year,venue,doi,arxiv_id
AI/ML,"MobileSeg: Real-time Segmentation","John Doe",2024,TIP,10.1109/TIP.2024.1234567,
```

---

### 2. 摘要获取器 (`abstract_fetcher`)

**功能**: 通过 Crossref/arXiv API 批量获取摘要

**命令行**:
```bash
python -m skills.abstract_fetcher [选项]
```

**常用选项**:
| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入 CSV (含 doi 列) | - |
| `--output` | 输出 CSV 路径 | `papers_with_abstracts.csv` |
| `--id-column` | ID 列名：doi/arxiv_id | `doi` |
| `--max-workers` | 并发数 | 5 |
| `--cache-dir` | 缓存目录 | `./cache/abstracts` |

**示例**:
```bash
# 从 DOI 获取摘要
python -m skills.abstract_fetcher \
  --input papers.csv \
  --output papers_with_abstracts.csv

# 从 arXiv ID 获取摘要
python -m skills.abstract_fetcher \
  --input papers.csv \
  --output papers_with_abstracts.csv \
  --id-column arxiv_id
```

**缓存机制**:
- 自动保存获取的摘要到 `./cache/abstracts/abstract_cache.json`
- 重复请求自动使用缓存
- 缓存永久保存，手动可删除

---

### 3. 精确分类器 (`precision_classifier`)

**功能**: 基于标题 + 摘要进行多标签分类

**命令行**:
```bash
python -m skills.precision_classifier [选项]
```

**常用选项**:
| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入 CSV (含 title/abstract) | - |
| `--output` | 输出 CSV 路径 | `classified_papers.csv` |
| `--taxonomy` | 分类体系 YAML 文件 | 内置默认 |
| `--output-dir` | 按类别导出目录 | - |
| `--min-score` | 最小分数阈值 | 1.0 |

**示例**:
```bash
# 使用自定义分类体系
python -m skills.precision_classifier \
  --input papers_with_abstracts.csv \
  --output classified_papers.csv \
  --taxonomy config/tip_segmentation.yaml \
  --output-dir analysis_by_category/
```

**分类体系配置** (`config/tip_segmentation.yaml`):
```yaml
taxonomy:
  Semantic Segmentation:
    keywords: ["semantic segmentation", "panoptic"]
    subcategories:
      Medical: ["medical", "mri", "ct"]
      Remote Sensing: ["satellite", "aerial"]
  
  Efficient Segmentation:
    keywords: ["efficient", "mobile", "real-time"]
    subcategories:
      Quantization: ["quantization", "low-bit"]
      Pruning: ["pruning", "sparse"]
```

**输出格式**:
- **CSV**: 包含 `primary_category`, `confidence`, `all_categories` 列
- **Markdown**: 按类别导出到单独文件

---

## 🎯 典型使用场景

### 场景 1: TIP 专刊分析 (分割 + 小模型)

```bash
# 完整流程
./analyze_tip.sh tip33

# 脚本内容:
#!/bin/bash
URL="https://dblp.org/db/journals/tip/tip$1.html"

# 1. DBLP 分析
python -m skills.dblp_analyzer \
  --url "$URL" \
  --keywords "segmentation" "efficient" "mobile" \
  --format csv \
  --output tip_${1}_raw.csv

# 2. 获取摘要
python -m skills.abstract_fetcher \
  --input tip_${1}_raw.csv \
  --output tip_${1}_with_abstracts.csv

# 3. 精确分类
python -m skills.precision_classifier \
  --input tip_${1}_with_abstracts.csv \
  --taxonomy config/tip_segmentation.yaml \
  --output tip_${1}_classified.csv \
  --output-dir tip${1}_analysis/
```

**输出**:
```
tip33_analysis/
├── Semantic_Segmentation.md (18 篇)
├── Efficient_Segmentation.md (15 篇)
├── Backbone.md (12 篇)
└── Instance_Segmentation.md (8 篇)
```

---

### 场景 2: CVPR/ICCV/ECCV 会议追踪

```bash
# 分析 CVPR 2024 Backbone 方向
python -m skills.dblp_analyzer \
  --url "https://dblp.org/db/conf/cvpr/cvpr2024.html" \
  --keywords "backbone" "resnet" "vit" \
  --format csv \
  --output cvpr2024_backbone.csv

python -m skills.abstract_fetcher \
  --input cvpr2024_backbone.csv \
  --output cvpr2024_backbone_abstracts.csv

python -m skills.precision_classifier \
  --input cvpr2024_backbone_abstracts.csv \
  --taxonomy config/backbone.yaml \
  --output cvpr2024_backbone_classified.csv \
  --output-dir cvpr2024_backbone_analysis/
```

---

### 场景 3: 自定义研究方向

```python
# Python API 方式
from skills.dblp_analyzer import DBLPAnalyzer
from skills.abstract_fetcher import AbstractFetcher
from skills.precision_classifier import PrecisionClassifier

# 1. DBLP 分析
analyzer = DBLPAnalyzer()
result = analyzer.analyze_url(
    "https://dblp.org/db/journals/tip/tip33.html",
    keywords=["rag", "retrieval augmented"],
    exclude_keywords=["survey"]
)
analyzer.export_csv(result, "tip33_rag.csv")

# 2. 获取摘要
fetcher = AbstractFetcher()
import csv

with open("tip33_rag.csv", 'r') as f:
    reader = csv.DictReader(f)
    dois = [row['doi'] for row in reader if row.get('doi')]

abstracts = fetcher.batch_fetch(dois=dois)
fetcher.merge_with_csv("tip33_rag.csv", abstracts, "tip33_rag_abstracts.csv")

# 3. 自定义分类
custom_taxonomy = {
    "RAG": {
        "keywords": ["rag", "retrieval augmented", "retrieval-language"],
        "subcategories": {
            "Vision": ["image", "vision", "visual"],
            "NLP": ["text", "language", "nlp"]
        }
    },
    "Efficient LLM": {
        "keywords": ["efficient", "compression", "quantization"]
    }
}

classifier = PrecisionClassifier(custom_taxonomy=custom_taxonomy)
by_category = classifier.classify_csv(
    "tip33_rag_abstracts.csv",
    "tip33_rag_classified.csv",
    min_score=1.0
)

classifier.export_by_category(by_category, "tip33_rag_analysis/", format='markdown')
```

---

## 📊 输出示例

### DBLP 分析输出 (CSV)

```csv
category,title,authors,year,venue,doi,arxiv_id
AI/ML,"MobileSeg: Real-time Segmentation on Mobile Devices","J. Smith, K. Lee",2024,TIP,10.1109/TIP.2024.1234567,
Efficient ML,"EdgeSeg: Fast Semantic Segmentation","A. Wang et al.",2024,TIP,10.1109/TIP.2024.7654321,
```

### 精确分类输出 (CSV)

```csv
title,abstract,primary_category,confidence,all_categories
"MobileSeg: Real-time...","We propose MobileSeg...",Efficient Segmentation,high,Efficient Segmentation|AI/ML
"EdgeSeg: Fast...","EdgeSeg enables...","Efficient Segmentation",high,Efficient Segmentation|Computer Vision
```

### Markdown 分类报告

```markdown
# Efficient Segmentation

**论文数**: 15

---

### 1. MobileSeg: Real-time Segmentation on Mobile Devices

**作者**: J. Smith, K. Lee  
**年份**: 2024  
**DOI**: [10.1109/TIP.2024.1234567](https://doi.org/10.1109/TIP.2024.1234567)  
**置信度**: ✅ high  
**其他类别**: Efficient Segmentation|AI/ML

---
```

---

## ⚙️ 配置说明

### 分类体系配置

**文件位置**: `config/tip_segmentation.yaml`, `config/backbone.yaml`

**结构**:
```yaml
keywords:
  must_include:      # 必须包含的关键词
    - "segmentation"
  efficiency:        # 效率相关关键词
    - "efficient"
  exclude:          # 排除的关键词
    - "survey"

taxonomy:           # 分类体系
  Category Name:
    keywords:       # 类别关键词
      - "keyword1"
    subcategories:  # 子类别 (可选)
      Sub Category:
        - "sub_keyword"

zotero:            # Zotero 配置 (Phase 4 使用)
  venue_path: "Journal/TIP/2024/Vol.33"
  topic_paths:
    - "Segmentation/Semantic"

analysis:          # 深入分析配置 (Phase 3 使用)
  include_comparison_table: true
  top_n_recommendations: 10
```

---

## 🔧 故障排除

### 问题 1: DBLP 解析失败

**症状**: 提取到 0 篇论文  
**原因**: DBLP 页面结构变化或网络问题

**解决方案**:
```bash
# 检查网络连接
curl -I https://dblp.org/db/journals/tip/tip33.html

# 增加超时
python -m skills.dblp_analyzer --url "..." --timeout 30
```

### 问题 2: 摘要获取成功率低

**症状**: 大量论文无摘要  
**原因**: DOI 无法在 Crossref 中找到

**解决方案**:
```bash
# 尝试 arXiv ID
python -m skills.abstract_fetcher \
  --input papers.csv \
  --id-column arxiv_id

# 增加并发数
python -m skills.abstract_fetcher \
  --max-workers 10
```

### 问题 3: 分类不准确

**症状**: 大量论文分类为 "Unclassified"  
**原因**: 关键词不匹配或阈值过高

**解决方案**:
```bash
# 降低阈值
python -m skills.precision_classifier \
  --min-score 0.5

# 自定义分类体系，添加更多关键词
python -m skills.precision_classifier \
  --taxonomy my_custom.yaml
```

---

## 📈 开发计划

### Phase 3: 深入分析 (待开发)
- [ ] `paper_extractor` - 方法/数据集/指标提取
- [ ] `deep_analysis_report` - 对比表格/趋势分析

### Phase 4: Zotero 集成 (待开发)
- [ ] `zotero_dual_sync` - 双目录同步
- [ ] `zotero_tagger` - 自动标签

### Phase 5: 工作流自动化 (待开发)
- [ ] `workflow_orchestrator` - 一键执行完整流程
- [ ] 断点续传
- [ ] 进度可视化

---

## 📝 最佳实践

1. **先用 DBLP 快速筛选** - 零成本获得论文列表
2. **结合关键词过滤** - 缩小范围到特定方向
3. **批量获取摘要** - 提高分类准确率
4. **自定义分类体系** - 适应具体研究需求
5. **人工确认重要论文** - 自动分类仅供参考

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

**添加新的分类体系**:
```yaml
# config/my_research_area.yaml
taxonomy:
  My Research Area:
    keywords: ["keyword1", "keyword2"]
    subcategories:
      Sub Area: ["sub_keyword"]
```

---

## 📝 License

MIT License

---

## 🙏 致谢

感谢以下开源项目:
- [DBLP](https://dblp.org/)
- [Crossref](https://www.crossref.org/)
- [arXiv](https://arxiv.org/)
