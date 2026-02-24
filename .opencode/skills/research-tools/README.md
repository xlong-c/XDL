# Research Tools MCP Skill

计算机科学研究工具集 - 零 token 消耗的论文分析、分类、摘要获取和报告生成工具。

## 安装与配置

### 1. 依赖安装

```bash
cd /root/workspace/xdl/sci_research
pip install -r requirements.txt
```

**requirements.txt**:
```txt
requests>=2.28.0
beautifulsoup4>=4.11.0
pyyaml>=6.0
pyzotero>=1.5.0  # 可选，Zotero 同步需要
```

### 2. OpenCode 配置

技能已配置在 `.opencode/skills/research-tools/opencode.json`。

MCP 服务器通过 stdio 协议与 OpenCode 通信。

## 可用工具

### 1. `dblp_analyze` - DBLP 论文分析

分析 DBLP 期刊/会议页面，自动分类到 10+ 计算机科学领域。

**特点**:
- 零 token 消耗 - 基于规则匹配
- 快速 - ~50 篇/秒
- 支持关键词筛选和年份筛选

**示例**:
```json
{
  "url": "https://dblp.org/db/conf/icml/icml2024.html",
  "keywords": ["deep learning", "neural network"],
  "exclude_keywords": ["survey", "review"],
  "min_year": 2024,
  "output_format": "markdown"
}
```

**返回**:
```json
{
  "success": true,
  "total_papers": 156,
  "distribution": {
    "AI/ML": 89,
    "Data Mining": 34,
    "Unclassified": 33
  },
  "output_path": "./output/dblp_analysis.md",
  "content": "..."
}
```

---

### 2. `fetch_abstracts` - 批量摘要获取

通过 Crossref/arXiv API 批量获取论文摘要。

**特点**:
- 支持缓存，避免重复请求
- 并发请求，提高效率
- 失败重试机制

**示例**:
```json
{
  "dois": ["10.1109/TIP.2024.1234567", "10.1109/TIP.2024.7654321"],
  "arxiv_ids": ["2401.12345", "2402.67890"],
  "max_workers": 5
}
```

**返回**:
```json
{
  "success": true,
  "fetched_count": 4,
  "abstracts": {
    "10.1109/TIP.2024.1234567": {
      "doi": "10.1109/TIP.2024.1234567",
      "title": "Paper Title",
      "abstract": "Abstract text...",
      "year": 2024,
      "source": "crossref"
    }
  }
}
```

---

### 3. `classify_papers` - 论文精确分类

基于标题和摘要进行多标签分类。

**特点**:
- 支持自定义分类体系
- 多标签分类，支持子类别
- 置信度评分

**示例**:
```json
{
  "papers": [
    {
      "title": "MobileSeg: Real-time Segmentation",
      "abstract": "We propose MobileSeg...",
      "doi": "10.1109/TIP.2024.1234567"
    }
  ],
  "min_score": 1.0,
  "output_dir": "./analysis/by_category"
}
```

**返回**:
```json
{
  "success": true,
  "classified_count": 1,
  "categories": ["AI/ML", "Computer Vision"],
  "distribution": {
    "AI/ML": 1,
    "Computer Vision": 1
  },
  "papers": [...]
}
```

---

### 4. `extract_paper_info` - 关键信息提取

从论文中提取方法、数据集、指标、代码链接等。

**提取内容**:
- 研究方法类型
- 使用的数据集
- 评估指标及数值
- 代码链接 (GitHub)
- 关键贡献点

**示例**:
```json
{
  "papers": [
    {
      "title": "MobileSeg: Real-time Segmentation",
      "abstract": "We propose MobileSeg for efficient segmentation..."
    }
  ],
  "group_by": "method"
}
```

**返回**:
```json
{
  "success": true,
  "extracted_count": 1,
  "papers": [
    {
      "title": "...",
      "methods": "Semantic Segmentation|Efficient ML",
      "datasets": "Cityscapes|COCO",
      "metrics": "mIoU|FPS",
      "metric_values": {"mIoU": 78.5, "FPS": 30},
      "code_url": "https://github.com/...",
      "contributions": "We propose..."
    }
  ],
  "summary": "..."
}
```

---

### 5. `generate_analysis_report` - 深入分析报告

生成完整的文献分析报告。

**报告内容**:
- 领域分布统计
- 时间线分析
- 方法对比表格
- 数据集/指标统计
- Top N 推荐论文
- 发展趋势与开放问题

**示例**:
```json
{
  "papers": [...],
  "top_n": 10,
  "include_timeline": true,
  "include_comparison": true
}
```

**返回**:
```json
{
  "success": true,
  "paper_count": 156,
  "report": "# 深入分析报告\n\n..."
}
```

---

### 6. `run_full_workflow` - 完整工作流

一键执行完整分析流程。

**流程**:
1. DBLP 分析
2. 摘要获取
3. 精确分类
4. 信息提取
5. 报告生成
6. (可选) Zotero 同步

**示例**:
```json
{
  "dblp_url": "https://dblp.org/db/journals/tip/tip33.html",
  "keywords": ["segmentation", "efficient"],
  "exclude_keywords": ["survey"],
  "output_dir": "./tip33_analysis",
  "zotero_sync": false
}
```

**返回**:
```json
{
  "success": true,
  "duration": 45.3,
  "files": {
    "raw_csv": "./tip33_analysis/01_raw_papers.csv",
    "abstracts_csv": "./tip33_analysis/02_with_abstracts.csv",
    "classified_csv": "./tip33_analysis/03_classified.csv",
    "extracted_csv": "./tip33_analysis/04_extracted.csv",
    "deep_analysis_md": "./tip33_analysis/05_deep_analysis.md"
  },
  "errors": []
}
```

---

## 使用场景

### 场景 1: 快速浏览期刊/会议

```
用户：分析 TIP 2024 年第 33 卷的分割相关论文

工具调用：
{
  "tool": "dblp_analyze",
  "args": {
    "url": "https://dblp.org/db/journals/tip/tip33.html",
    "keywords": ["segmentation"],
    "output_format": "markdown"
  }
}
```

### 场景 2: 深入分析特定方向

```
用户：深入分析 RAG 领域的最新进展

工具调用：
1. dblp_analyze - 筛选相关论文
2. fetch_abstracts - 获取摘要
3. classify_papers - 精确分类
4. extract_paper_info - 提取关键信息
5. generate_analysis_report - 生成报告
```

### 场景 3: 一键完整分析

```
用户：完整分析 CVPR 2024 的 Backbone 方向论文

工具调用：
{
  "tool": "run_full_workflow",
  "args": {
    "dblp_url": "https://dblp.org/db/conf/cvpr/cvpr2024.html",
    "keywords": ["backbone", "resnet", "vit"],
    "output_dir": "./cvpr2024_backbone"
  }
}
```

---

## 分类体系

默认支持 10+ 计算机科学领域：

| 领域 | 关键词示例 |
|------|-----------|
| AI/ML | neural network, deep learning, transformer, llm |
| Data Mining | clustering, classification, recommendation |
| Information Retrieval | search, ranking, retrieval, question answering |
| Database Systems | sql, nosql, query optimization, vector database |
| Computer Networks | routing, protocol, iot, 5g, sdn |
| Security & Privacy | cryptography, authentication, blockchain, adversarial |
| Software Engineering | testing, devops, microservice, llm4se |
| HCI | user interface, visualization, vr/ar |
| Theory | complexity, algorithm, optimization, proof |
| HPC | parallel, distributed, gpu, cuda, supercomputing |

---

## Zotero 集成 (可选)

配置 Zotero 同步需要：

1. 安装 `pyzotero`: `pip install pyzotero`
2. 获取 Zotero Library ID 和 API Key
3. 配置双目录路径：
   - Venue 目录：`Journal/TIP/2024/Vol.33`
   - Topic 目录：`Segmentation/Semantic|Efficient ML`

---

## 故障排除

### 问题 1: DBLP 解析失败

**症状**: 提取到 0 篇论文

**解决**:
```bash
# 检查网络连接
curl -I https://dblp.org/db/journals/tip/tip33.html

# 增加超时
python -m skills.dblp_analyzer --url "..." --timeout 30
```

### 问题 2: 摘要获取成功率低

**症状**: 大量论文无摘要

**解决**:
- 尝试使用 arXiv ID 而非 DOI
- 增加并发数：`max_workers: 10`
- 检查缓存目录权限

### 问题 3: 分类不准确

**解决**:
- 降低 `min_score` 阈值 (默认 1.0 → 0.5)
- 自定义分类体系，添加更多关键词
- 确保摘要内容完整

---

## 许可证

MIT License

---

## 贡献

欢迎提交 Issue 和 Pull Request！

**添加新的分类领域**:
在 `dblp_analyzer.py` 中修改 `DEFAULT_TAXONOMY`。

**改进分类算法**:
- 添加更多关键词
- 优化匹配策略
- 引入机器学习分类器
