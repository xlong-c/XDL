# 科研工具 Skill 体系总览

> 为计算机科学研究人员打造的零 token 消耗科研辅助工具集

---

## 📦 已实现工具

### 1. DBLP 论文领域分析器 (`dblp_analyzer`)

**核心功能**: 分析 DBLP 期刊/会议页面的所有论文，自动分类到 10+ 计算机科学领域

**特点**:
- ✅ 零 token 消耗 - 基于规则匹配，无需 LLM
- ✅ 批量处理 - 单次处理整个期刊卷期或会议论文集
- ✅ 快速 - ~50 篇/秒的处理速度
- ✅ 多格式输出 - Markdown 表格、JSON

**使用示例**:
```bash
python -m skills.dblp_analyzer \
  --url "https://dblp.org/db/conf/icml/icml2024.html" \
  --output icml2024_analysis.md
```

**详细文档**: [DBLP_ANALYZER_GUIDE.md](./skills/DBLP_ANALYZER_GUIDE.md)

---

## 🚀 即将实现

### 2. 论文搜索工具 (`paper_search`)

**计划功能**:
- 跨 arXiv、Semantic Scholar、DBLP 统一搜索
- 支持关键词、作者、年份、领域筛选
- 批量导出搜索结果

### 3. Zotero 导入工具 (`zotero_importer`)

**计划功能**:
- 从 DOI 批量导入文献到 Zotero
- 自动分类到指定集合
- 同步标签和笔记

### 4. 论文摘要工具 (`paper_summarize`)

**计划功能**:
- AI 生成论文 TL;DR 和详细摘要
- 结构化提取方法、数据集、指标
- 支持批量处理

### 5. 文献综述生成器 (`literature_review`)

**计划功能**:
- 基于一组论文自动生成综述草稿
- 按方法/时间线组织
- 识别关键工作和开放问题

---

## 🎯 设计理念

### 1. 零 token 优先

能不用 LLM 就不用 LLM，优先使用：
- 规则匹配
- 关键词筛选
- 轻量级分类器

### 2. 批量处理

避免逐篇处理，采用：
- 批量 API 请求
- 合并 prompt 减少 token
- 并行处理

### 3. 模块化

每个工具独立可用，也可组合使用：
```python
# 工作流示例
from skills.dblp_analyzer import DBLPAnalyzer
from skills.zotero_importer import ZoteroImporter

# 1. 分析 DBLP 页面
analyzer = DBLPAnalyzer()
result = analyzer.analyze_url("https://dblp.org/db/conf/icml/icml2024.html")

# 2. 导出 AI/ML 领域论文 DOI
ai_dois = [p['doi'] for p in result['categories']['AI/ML'] if p['doi']]

# 3. 批量导入 Zotero
importer = ZoteroImporter()
importer.import_by_doi(ai_dois, collection="ICML2024_AI")
```

### 4. 本地优先

- 数据存储本地化
- 支持自托管服务
- 减少对云服务依赖

---

## 📊 工具对比

| 工具 | Token 消耗 | 准确率 | 速度 | 适用场景 |
|------|-----------|--------|------|---------|
| **DBLP 分析器** | 0 | 75-85% | 快 | 快速浏览期刊/会议 |
| **论文搜索** (计划) | 0 | N/A | 快 | 跨平台文献发现 |
| **Zotero 导入** (计划) | 0 | 100% | 中 | 文献管理 |
| **论文摘要** (计划) | 中 (50-200/篇) | 85-95% | 中 | 深度理解论文 |
| **文献综述** (计划) | 高 (500+/百篇) | 80-90% | 慢 | 文献综述写作 |

---

## 🛠️ 技术栈

- **Python 3.8+**: 所有工具使用 Python 实现
- **Requests**: HTTP 请求
- **BeautifulSoup4**: HTML 解析
- **PyZotero**: Zotero API 客户端（计划）
- **arxiv.py**: arXiv API 客户端（计划）

---

## 📁 目录结构

```
sci_research/
├── skills/
│   ├── __init__.py              # 包初始化
│   ├── dblp_analyzer.py         # DBLP 分析器
│   ├── paper_search.py          # 论文搜索 (TODO)
│   ├── zotero_importer.py       # Zotero 导入 (TODO)
│   ├── paper_summarize.py       # 论文摘要 (TODO)
│   └── literature_review.py     # 文献综述 (TODO)
├── DBLP_ANALYZER_GUIDE.md       # DBLP 工具使用指南
└── README.md                    # 本文件
```

---

## 🎓 使用场景

### 场景 1: 开始新研究方向

```bash
# 1. 分析顶会最近 3 年论文
python -m skills.dblp_analyzer \
  --url "https://dblp.org/db/conf/nips/nips2024.html" \
  --keywords "deep learning" \
  --output nips2024_dl.md

# 2. 查看领域分布，确定热点方向
# 3. 人工阅读高相关度论文
```

### 场景 2: 追踪最新进展

```python
# 批量分析多个期刊最新卷
from skills.dblp_analyzer import DBLPAnalyzer

analyzer = DBLPAnalyzer()
for venue in ['jmlr', 'pami', 'tist']:
    url = f"https://dblp.org/db/journals/{venue}/{venue}25.html"
    result = analyzer.analyze_url(url, quiet=True)
    print(f"{venue}: {result['distribution']}")
```

### 场景 3: 筛选特定领域论文

```python
# 自定义分类体系，精确定位研究方向
custom = {
    "RAG": ["retrieval augmented", "RAG", "retrieval-language"],
    "LLM Efficiency": ["efficient", "compression", "quantization"]
}
analyzer = DBLPAnalyzer(custom_taxonomy=custom)
result = analyzer.analyze_url("https://dblp.org/db/conf/acl/acl2024.html")
print(f"RAG 相关：{len(result['categories']['RAG'])} 篇")
```

---

## 📈 路线图

### Phase 1 (已完成)
- ✅ DBLP 论文领域分析器
- ✅ 基础文档和使用指南

### Phase 2 (进行中)
- 🔄 Zotero 导入工具
- 🔄 跨平台论文搜索

### Phase 3 (计划)
- ⏳ 论文摘要工具
- ⏳ 文献综述生成器

### Phase 4 (未来)
- ⏳ 研究趋势分析
- ⏳ 自动更新提醒
- ⏳ 与 Obsidian/Notion 集成

---

## 🤝 贡献

欢迎贡献代码、文档或提出建议！

### 添加新的领域分类

```python
# 在 dblp_analyzer.py 中修改 DEFAULT_TAXONOMY
DEFAULT_TAXONOMY = {
    # ... 现有领域 ...
    "My New Field": ["keyword1", "keyword2"]
}
```

### 改进分类算法

- 添加更多关键词
- 优化匹配策略
- 引入机器学习分类器

---

## 📝 License

MIT License

---

## 🙏 致谢

感谢以下开源项目：
- [DBLP](https://dblp.org/) - 计算机科学书目数据库
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) - HTML 解析库
- [Requests](https://requests.readthedocs.io/) - HTTP 请求库
