# OpenCode Research Tools Skill - 使用指南

## 快速开始

### 在 OpenCode 中使用

当你在 OpenCode 中安装了此技能后，可以通过以下方式调用：

#### 方式 1: 直接调用工具

```
分析 ICML 2024 的论文分布
```

OpenCode 会自动调用 `dblp_analyze` 工具。

#### 方式 2: 明确指定工具

```
使用 dblp_analyze 工具分析这个 DBLP 页面：
https://dblp.org/db/journals/tip/tip33.html
关键词：segmentation, efficient
```

#### 方式 3: 完整工作流

```
完整分析 CVPR 2024 的 Backbone 方向论文：
- DBLP URL: https://dblp.org/db/conf/cvpr/cvpr2024.html
- 关键词：backbone, resnet, vit
- 输出目录：./cvpr2024_backbone
```

---

## 工具调用示例

### 1. DBLP 分析

**用户输入**:
```
帮我分析 TIP 2024 年第 33 卷的论文，关注分割和小模型方向
```

**OpenCode 工具调用**:
```json
{
  "name": "dblp_analyze",
  "arguments": {
    "url": "https://dblp.org/db/journals/tip/tip33.html",
    "keywords": ["segmentation", "efficient", "mobile"],
    "exclude_keywords": ["survey", "review"],
    "output_format": "markdown"
  }
}
```

**预期输出**:
- 领域分布表格
- 每个领域的论文列表
- Markdown 格式报告

---

### 2. 批量获取摘要

**用户输入**:
```
为这些 DOI 获取摘要：
- 10.1109/TIP.2024.1234567
- 10.1109/TIP.2024.7654321
```

**OpenCode 工具调用**:
```json
{
  "name": "fetch_abstracts",
  "arguments": {
    "dois": [
      "10.1109/TIP.2024.1234567",
      "10.1109/TIP.2024.7654321"
    ],
    "max_workers": 5
  }
}
```

**预期输出**:
- 成功获取的摘要数量
- 每篇论文的标题、摘要、年份

---

### 3. 论文分类

**用户输入**:
```
对这批论文进行精确分类，使用自定义分类体系
```

**OpenCode 工具调用**:
```json
{
  "name": "classify_papers",
  "arguments": {
    "papers": [
      {
        "title": "MobileSeg: Real-time Segmentation",
        "abstract": "We propose MobileSeg for efficient...",
        "doi": "10.1109/TIP.2024.1234567"
      }
    ],
    "min_score": 1.0,
    "output_dir": "./analysis/by_category"
  }
}
```

**预期输出**:
- 分类结果分布
- 每篇论文的主要类别和置信度
- 按类别导出的 Markdown 文件

---

### 4. 信息提取

**用户输入**:
```
从这些论文中提取方法、数据集和指标信息
```

**OpenCode 工具调用**:
```json
{
  "name": "extract_paper_info",
  "arguments": {
    "papers": [...],
    "group_by": "method"
  }
}
```

**预期输出**:
- 每篇论文的方法类型
- 使用的数据集
- 评估指标及数值
- 代码链接 (如有)
- 方法汇总报告

---

### 5. 生成分析报告

**用户输入**:
```
基于这些分析结果生成深入报告
```

**OpenCode 工具调用**:
```json
{
  "name": "generate_analysis_report",
  "arguments": {
    "papers": [...],
    "top_n": 10,
    "include_timeline": true,
    "include_comparison": true
  }
}
```

**预期输出**:
- 完整的 Markdown 分析报告
- 包含：领域分布、时间线、对比表格、Top 推荐、趋势分析

---

### 6. 完整工作流

**用户输入**:
```
一键分析这个会议的所有论文
https://dblp.org/db/conf/icml/icml2024.html
```

**OpenCode 工具调用**:
```json
{
  "name": "run_full_workflow",
  "arguments": {
    "dblp_url": "https://dblp.org/db/conf/icml/icml2024.html",
    "keywords": ["deep learning"],
    "output_dir": "./icml2024_analysis"
  }
}
```

**预期输出**:
- 完整的工作流执行结果
- 所有中间文件和最终报告的路径
- 执行时间和统计信息

---

## 典型使用场景

### 场景 1: 开始新研究方向

**目标**: 快速了解某个领域的研究现状

**步骤**:
1. 使用 `dblp_analyze` 分析顶会/期刊
2. 查看领域分布，确定热点方向
3. 人工阅读高相关度论文

**对话示例**:
```
用户：我想研究高效语义分割，帮我看看 TIP 最新一期有哪些相关论文

OpenCode: 正在分析 TIP 2024 年第 33 卷...
[调用 dblp_analyze]
找到 255 篇论文，其中：
- Semantic Segmentation: 18 篇
- Efficient Segmentation: 15 篇
- Backbone: 12 篇

需要我获取这些论文的摘要吗？
```

---

### 场景 2: 文献调研

**目标**: 深入分析特定方向的最新进展

**步骤**:
1. DBLP 筛选相关论文
2. 批量获取摘要
3. 精确分类
4. 提取关键信息
5. 生成分析报告

**对话示例**:
```
用户：帮我深入分析 RAG 领域的最新进展

OpenCode: 好的，我将执行以下步骤：
1. 分析 ACL/EMNLP 2024 的 RAG 相关论文
2. 获取摘要
3. 分类和提取信息
4. 生成分析报告

[逐步执行各个工具]

完成！分析报告已生成，包括：
- 34 篇 RAG 相关论文
- 热门方法：Retrieval Augmentation, Context Compression
- 常用数据集：Natural Questions, MS MARCO
- Top 10 推荐论文列表
```

---

### 场景 3: 追踪最新论文

**目标**: 定期追踪特定期刊/会议的最新论文

**步骤**:
1. 定期运行 DBLP 分析
2. 筛选关键词
3. 导出 Markdown 报告
4. (可选) 同步到 Zotero

**对话示例**:
```
用户：帮我追踪 TIP 最新一期的分割论文

OpenCode: 正在分析 https://dblp.org/db/journals/tip/tip33.html...
找到 18 篇分割相关论文：
- Semantic Segmentation: 12 篇
- Instance Segmentation: 6 篇

已导出 Markdown 报告：./tip33_segmentation.md

需要同步到 Zotero 吗？
```

---

## 自定义分类体系

如需自定义分类体系，可以创建 YAML 配置文件：

**config/my_research.yaml**:
```yaml
taxonomy:
  My Research Area:
    keywords: ["keyword1", "keyword2"]
    subcategories:
      Sub Area 1: ["sub_keyword1"]
      Sub Area 2: ["sub_keyword2"]
  
  Another Area:
    keywords: ["keyword3", "keyword4"]
```

然后在调用时指定：
```
使用这个分类体系对论文进行分类：config/my_research.yaml
```

---

## 故障排除

### 问题 1: 工具无法调用

**症状**: OpenCode 提示找不到工具

**解决**:
1. 检查技能是否正确安装在 `.opencode/skills/research-tools/`
2. 确认 `opencode.json` 配置正确
3. 重启 OpenCode

### 问题 2: DBLP 解析失败

**症状**: 提取到 0 篇论文

**解决**:
- 检查网络连接
- 验证 URL 是否正确
- 增加超时时间

### 问题 3: 摘要获取失败

**症状**: 大量论文无摘要

**解决**:
- 检查 DOI 是否正确
- 尝试使用 arXiv ID
- 增加并发数

---

## 最佳实践

1. **先用 DBLP 快速筛选** - 零成本获得论文列表
2. **结合关键词过滤** - 缩小范围到特定方向
3. **批量获取摘要** - 提高分类准确率
4. **自定义分类体系** - 适应具体研究需求
5. **人工确认重要论文** - 自动分类仅供参考

---

## 性能提示

- **DBLP 分析**: ~50 篇/秒，适合批量处理
- **摘要获取**: 并发 5-10 个请求，成功率>90%
- **分类**: 基于规则匹配，毫秒级响应
- **报告生成**: 百篇论文约 1-2 秒

---

## 支持与贡献

遇到问题或想添加新功能？

1. 提交 Issue 描述问题
2. 提交 Pull Request 改进代码
3. 添加新的分类领域
4. 改进分类算法

---

## 许可证

MIT License
