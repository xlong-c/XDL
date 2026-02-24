# 快速开始 - Research Tools MCP Skill

## 1 分钟快速开始

### 前提条件

- Python 3.8+
- OpenCode 已安装并配置

### 步骤 1: 安装依赖 (30 秒)

```bash
cd /root/workspace/xdl/sci_research
pip install requests beautifulsoup4 pyyaml
```

### 步骤 2: 验证安装 (30 秒)

```bash
python -c "from skills.mcp_server import MCPServer; print('✅ MCP Server 就绪')"
```

### 步骤 3: 在 OpenCode 中使用

现在可以在 OpenCode 中直接使用自然语言调用工具：

```
分析这个 DBLP 页面：
https://dblp.org/db/conf/icml/icml2024.html
关键词：deep learning, neural network
```

---

## 常用命令

### 分析 DBLP 页面

```bash
python -m skills.dblp_analyzer \
  --url "https://dblp.org/db/conf/icml/icml2024.html" \
  --keywords "deep learning" \
  --exclude-keywords "survey" \
  --output icml2024_dl.md
```

### 获取摘要

```bash
python -m skills.abstract_fetcher \
  --input papers.csv \
  --output papers_with_abstracts.csv
```

### 分类论文

```bash
python -m skills.precision_classifier \
  --input papers_with_abstracts.csv \
  --output classified_papers.csv \
  --output-dir analysis_by_category/
```

### 完整工作流

```bash
python -m skills.workflow_orchestrator \
  --url "https://dblp.org/db/journals/tip/tip33.html" \
  --keywords "segmentation" "efficient" \
  --output-dir tip33_analysis/
```

---

## 在 OpenCode 中的使用示例

### 示例 1: 快速浏览

```
用户：帮我看看 ICML 2024 有哪些深度学习相关论文

OpenCode: [调用 dblp_analyze 工具]
找到 156 篇深度学习相关论文：
- AI/ML: 89 篇
- Data Mining: 34 篇
- 其他：33 篇

需要我获取这些论文的摘要吗？
```

### 示例 2: 深入分析

```
用户：深入分析 RAG 领域的最新进展

OpenCode: [执行完整工作流]
1. 分析 ACL/EMNLP 2024
2. 获取摘要
3. 分类和提取信息
4. 生成分析报告

完成！找到 34 篇 RAG 相关论文，
报告已生成：./analysis/rag_analysis.md
```

### 示例 3: 获取摘要

```
用户：为这些 DOI 获取摘要：
- 10.1109/TIP.2024.1234567
- 10.1109/TIP.2024.7654321

OpenCode: [调用 fetch_abstracts 工具]
成功获取 2 个摘要：
1. "MobileSeg: Real-time..."
   Abstract: We propose...
2. "EdgeSeg: Fast..."
   Abstract: EdgeSeg enables...
```

---

## 工具速查表

| 工具 | 用途 | 速度 |
|------|------|------|
| `dblp_analyze` | DBLP 分析 | ~50 篇/秒 |
| `fetch_abstracts` | 摘要获取 | 并发 5-10 |
| `classify_papers` | 论文分类 | 毫秒级 |
| `extract_paper_info` | 信息提取 | 毫秒级 |
| `generate_analysis_report` | 报告生成 | 1-2 秒/百篇 |
| `run_full_workflow` | 完整流程 | 30-60 秒 |

---

## 常见问题

**Q: 如何自定义分类体系？**

A: 创建 YAML 配置文件，然后在调用时指定 `taxonomy_file` 参数。

**Q: 摘要获取成功率低？**

A: 尝试使用 arXiv ID 而非 DOI，或增加并发数。

**Q: 如何同步到 Zotero？**

A: 安装 `pyzotero`，然后在完整工作流中启用 `zotero_sync`。

---

## 下一步

- 📖 阅读完整文档：`.opencode/skills/research-tools/README.md`
- 📚 使用指南：`.opencode/skills/research-tools/USAGE_GUIDE.md`
- 🔧 测试脚本：`sci_research/test_mcp_server.py`

---

## 支持

遇到问题？提交 Issue 或查看文档。

**License**: MIT
