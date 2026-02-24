"""
DBLP 论文领域分析器 - 零 token 消耗版本

专为快速筛选期刊/会议论文设计的轻量级分类工具。
基于规则匹配，无需 LLM，无 token 消耗。
"""

import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Any
from datetime import datetime
import json
import re


class DBLPAnalyzer:
    """DBLP 论文领域分析器"""
    
    # 默认计算机科学分类体系
    DEFAULT_TAXONOMY = {
        "AI/ML": [
            "neural network", "deep learning", "machine learning", "reinforcement learning",
            "natural language processing", "computer vision", "knowledge graph",
            "reasoning", "planning", "multi-agent", "generative", "llm", "transformer",
            "attention mechanism", "representation learning", "self-supervised",
            "contrastive learning", "meta-learning", "few-shot learning", "zero-shot"
        ],
        "Data Mining": [
            "clustering", "classification", "association rule", "pattern mining",
            "anomaly detection", "feature selection", "recommendation system",
            "frequent itemset", "sequential pattern", "graph mining", "stream mining"
        ],
        "Information Retrieval": [
            "search engine", "query", "ranking", "relevance", "text retrieval",
            "question answering", "document retrieval", "web search", "retrieval",
            "semantic search", "hybrid search", "reranking"
        ],
        "Database Systems": [
            "sql", "nosql", "query optimization", "transaction", "indexing",
            "distributed database", "data warehouse", "data lake", "streaming database",
            "vector database", "graph database", "relational database"
        ],
        "Computer Networks": [
            "routing", "protocol", "tcp/ip", "wireless", "iot", "5g", "sdn",
            "network security", "edge computing", "fog computing", "network slicing",
            "mac protocol", "sensor network"
        ],
        "Security & Privacy": [
            "cryptography", "authentication", "encryption", "attack", "vulnerability",
            "privacy-preserving", "blockchain", "zero-knowledge", "secure computation",
            "federated learning", "differential privacy", "adversarial", "backdoor"
        ],
        "Software Engineering": [
            "testing", "debugging", "refactoring", "code analysis", "devops",
            "continuous integration", "microservice", "architecture", "code review",
            "technical debt", "software maintenance", "agile", "llm4se"
        ],
        "Human-Computer Interaction": [
            "user interface", "usability", "accessibility", "visualization",
            "user experience", "interaction design", "hci", "human-computer",
            "eye tracking", "gesture", "vr/ar", "mixed reality"
        ],
        "Theory & Algorithms": [
            "complexity", "approximation algorithm", "optimization", "graph theory",
            "cryptography", "logic", "proof", "computational complexity",
            "online algorithm", "randomized algorithm", "lower bound"
        ],
        "High Performance Computing": [
            "parallel computing", "distributed system", "gpu", "cuda", "mapreduce",
            "spark", "load balancing", "mpi", "high performance", "supercomputing",
            "accelerator", "tensor core"
        ]
    }
    
    def __init__(
        self, 
        custom_taxonomy: Optional[Dict[str, List[str]]] = None,
        timeout: int = 15,
        verbose: bool = True
    ):
        """
        初始化分析器
        
        Args:
            custom_taxonomy: 自定义分类体系，覆盖默认值
            timeout: HTTP 请求超时时间（秒）
            verbose: 是否打印详细日志
        """
        self.taxonomy = custom_taxonomy or self.DEFAULT_TAXONOMY
        self.timeout = timeout
        self.verbose = verbose
    
    def analyze_url(
        self, 
        dblp_url: str,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        分析 DBLP 页面所有论文的领域分布
        
        Args:
            dblp_url: DBLP 期刊/会议页面 URL
            min_year: 最小年份筛选
            max_year: 最大年份筛选
            keywords: 标题必须包含的关键词
            exclude_keywords: 标题必须排除的关键词
        
        Returns:
            分类结果字典
        """
        if self.verbose:
            print(f"📊 开始分析 DBLP 页面：{dblp_url}")
        
        # 步骤 1: 解析 DBLP HTML 页面
        papers = self._parse_dblp_page(dblp_url)
        if self.verbose:
            print(f"✓ 从 DBLP 页面提取 {len(papers)} 篇论文")
        
        # 步骤 2: 应用筛选条件
        if min_year or max_year:
            papers = self._filter_by_year(
                papers, 
                min_year=min_year, 
                max_year=max_year
            )
            if self.verbose:
                print(f"✓ 年份筛选后剩余 {len(papers)} 篇")
        
        if keywords or exclude_keywords:
            papers = self._filter_by_keywords(
                papers,
                must_include=keywords,
                must_exclude=exclude_keywords
            )
            if self.verbose:
                print(f"✓ 关键词筛选后剩余 {len(papers)} 篇")
        
        # 步骤 3: 领域分类
        classifications = self._classify_by_rules(papers)
        
        # 步骤 4: 组织结果
        result = self._organize_results(papers, classifications, dblp_url)
        
        if self.verbose:
            print(f"✓ 分析完成！领域分布：{result['distribution']}")
        
        return result
    
    def _parse_dblp_page(self, url: str) -> List[Dict[str, Any]]:
        """
        解析 DBLP HTML 页面，提取论文元数据
        
        DBLP 页面结构:
        <li class="entry ipyr" data-dblpkey="...">
          <div class="data">
            <span class="title">论文标题</span>
            <span class="authors">作者列表</span>
            <span class="venue">期刊/会议名</span>
            <span class="year">年份</span>
            <span class="doi">DOI 链接</span>
            <span class="ee">电子版链接（可能是 arXiv）</span>
          </div>
        </li>
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; DBLPAnalyzer/1.0; +https://github.com/your-repo)'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f"无法访问 DBLP 页面：{e}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        papers = []
        
        # 查找所有论文条目
        for entry in soup.find_all('li', class_='entry'):
            paper = self._parse_entry(entry)
            if paper and paper.get('title'):
                papers.append(paper)
        
        return papers
    
    def _parse_entry(self, entry) -> Optional[Dict[str, Any]]:
        """解析单个论文条目"""
        # 提取标题
        title_elem = entry.find('span', class_='title')
        if not title_elem:
            return None
        
        title = title_elem.get_text(strip=True)
        
        # 提取作者
        authors_elem = entry.find('span', class_='authors')
        authors = authors_elem.get_text(strip=True) if authors_elem else ""
        
        # 提取 DOI
        doi_elem = entry.find('span', class_='doi')
        doi = None
        if doi_elem:
            doi_link = doi_elem.find('a')
            if doi_link and 'doi.org' in doi_link.get('href', ''):
                doi = doi_link['href'].split('doi.org/')[-1]
        
        # 提取 arXiv ID（如果有）
        ee_elem = entry.find('span', class_='ee')
        arxiv_id = None
        if ee_elem:
            ee_link = ee_elem.find('a')
            if ee_link:
                ee_url = ee_link.get('href', '')
                if 'arxiv.org' in ee_url:
                    arxiv_id = ee_url.split('arxiv.org/abs/')[-1]
        
        # 提取年份
        year_elem = entry.find('span', class_='year')
        year = year_elem.get_text(strip=True) if year_elem else ""
        
        # 提取 venue（期刊/会议名）
        venue_elem = entry.find('span', class_='venue')
        venue = venue_elem.get_text(strip=True) if venue_elem else ""
        
        return {
            'title': title,
            'authors': authors,
            'year': year,
            'doi': doi,
            'arxiv_id': arxiv_id,
            'venue': venue,
            'dblp_key': entry.get('data-dblpkey', '')
        }
    
    def _filter_by_year(
        self, 
        papers: List[Dict], 
        min_year: Optional[int] = None,
        max_year: Optional[int] = None
    ) -> List[Dict]:
        """按年份筛选论文"""
        filtered = []
        for paper in papers:
            try:
                year = int(paper['year'])
                if min_year and year < min_year:
                    continue
                if max_year and year > max_year:
                    continue
                filtered.append(paper)
            except (ValueError, TypeError):
                # 年份无法解析时保留
                filtered.append(paper)
        return filtered
    
    def _filter_by_keywords(
        self,
        papers: List[Dict],
        must_include: Optional[List[str]] = None,
        must_exclude: Optional[List[str]] = None
    ) -> List[Dict]:
        """按关键词筛选论文"""
        filtered = []
        for paper in papers:
            title_lower = paper['title'].lower()
            
            # 必须包含的关键词
            if must_include:
                if not any(kw.lower() in title_lower for kw in must_include):
                    continue
            
            # 必须排除的关键词
            if must_exclude:
                if any(kw.lower() in title_lower for kw in must_exclude):
                    continue
            
            filtered.append(paper)
        
        return filtered
    
    def _classify_by_rules(self, papers: List[Dict]) -> List[Dict]:
        """
        基于规则的零成本分类
        
        Returns:
            每篇论文的分类结果列表
        """
        classifications = []
        
        for paper in papers:
            title_lower = paper['title'].lower()
            scores = {}
            
            # 对每个领域计算匹配分数
            for category, keywords in self.taxonomy.items():
                match_count = sum(1 for kw in keywords if kw in title_lower)
                if match_count > 0:
                    # 计算加权分数（避免长领域名占优）
                    scores[category] = match_count
            
            # 返回 top-3 匹配领域
            sorted_cats = sorted(
                scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            classifications.append({
                'title': paper['title'],
                'categories': [
                    {'name': cat, 'score': score} 
                    for cat, score in sorted_cats
                ],
                'primary_category': sorted_cats[0][0] if sorted_cats else "Unclassified",
                'confidence': 'high' if sorted_cats and sorted_cats[0][1] >= 2 else 'medium'
            })
        
        return classifications
    
    def _organize_results(
        self, 
        papers: List[Dict], 
        classifications: List[Dict],
        url: str
    ) -> Dict[str, Any]:
        """按领域组织结果"""
        by_category: Dict[str, List[Dict]] = {}
        
        for paper, cls in zip(papers, classifications):
            primary_cat = cls['primary_category']
            
            if primary_cat not in by_category:
                by_category[primary_cat] = []
            
            by_category[primary_cat].append({
                'title': paper['title'],
                'authors': paper['authors'],
                'year': paper['year'],
                'venue': paper['venue'],
                'doi': paper['doi'],
                'arxiv_id': paper['arxiv_id'],
                'classification': cls['categories'],
                'confidence': cls['confidence']
            })
        
        # 计算分布统计
        distribution = {
            cat: len(papers_list) 
            for cat, papers_list in by_category.items()
        }
        
        # 添加总计数
        unclassified_count = sum(
            1 for cls in classifications 
            if cls['primary_category'] == 'Unclassified'
        )
        if unclassified_count > 0:
            distribution['Unclassified'] = unclassified_count
        
        return {
            'total_papers': len(papers),
            'url': url,
            'analyzed_at': datetime.now().isoformat(),
            'categories': by_category,
            'distribution': distribution,
            'taxonomy_used': list(self.taxonomy.keys())
        }
    
    def export_markdown(
        self, 
        result: Dict[str, Any], 
        output_path: str,
        top_n: Optional[int] = None
    ) -> str:
        """
        导出为 Markdown 格式
        
        Args:
            result: analyze_url 的返回结果
            output_path: 输出文件路径
            top_n: 每个领域只显示前 N 篇（None 则显示全部）
        
        Returns:
            生成的 Markdown 内容
        """
        lines = [
            f"# DBLP 论文领域分析",
            f"",
            f"**URL**: {result['url']}",
            f"**总论文数**: {result['total_papers']}",
            f"**分析时间**: {result['analyzed_at'].split('T')[0]}",
            f"",
            f"## 领域分布",
            f"",
            f"| 领域 | 论文数 | 占比 |",
            f"|------|--------|------|"
        ]
        
        # 领域分布表格（按数量降序）
        sorted_dist = sorted(
            result['distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for cat, count in sorted_dist:
            percentage = (count / result['total_papers'] * 100) if result['total_papers'] > 0 else 0
            lines.append(f"| {cat} | {count} | {percentage:.1f}% |")
        
        lines.append(f"")
        lines.append(f"---")
        
        # 详细论文列表
        for cat, papers in sorted(
            result['categories'].items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        ):
            lines.append(f"")
            lines.append(f"## {cat} ({len(papers)}篇)")
            lines.append(f"")
            
            display_papers = papers[:top_n] if top_n else papers
            for i, paper in enumerate(display_papers, 1):
                lines.append(f"{i}. **{paper['title']}**")
                lines.append(f"   - 作者：{paper['authors']}")
                if paper['year']:
                    lines.append(f"   - 年份：{paper['year']}")
                if paper['venue']:
                    lines.append(f"   - 期刊/会议：{paper['venue']}")
                if paper['doi']:
                    lines.append(
                        f"   - DOI: [{paper['doi']}](https://doi.org/{paper['doi']})"
                    )
                if paper['arxiv_id']:
                    lines.append(
                        f"   - arXiv: [{paper['arxiv_id']}](https://arxiv.org/abs/{paper['arxiv_id']})"
                    )
                # 显示分类置信度
                confidence_emoji = "✅" if paper['confidence'] == 'high' else "⚠️"
                lines.append(f"   - 置信度：{confidence_emoji} {paper['confidence']}")
                lines.append(f"")
        
        content = "\n".join(lines)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if self.verbose:
            print(f"✓ Markdown 报告已保存至：{output_path}")
        
        return content
    
    def export_json(
        self, 
        result: Dict[str, Any], 
        output_path: str,
        pretty: bool = True
    ) -> str:
        """导出为 JSON 格式"""
        indent = 2 if pretty else None
        content = json.dumps(result, ensure_ascii=False, indent=indent)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if self.verbose:
            print(f"✓ JSON 数据已保存至：{output_path}")
        
        return content


# 命令行入口
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DBLP 论文领域分析器 - 零 token 消耗的快速分类工具'
    )
    parser.add_argument(
        '--url', 
        type=str, 
        required=True,
        help='DBLP 期刊/会议页面 URL'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='dblp_analysis.md',
        help='输出文件路径（默认：dblp_analysis.md）'
    )
    parser.add_argument(
        '--format', 
        type=str, 
        choices=['markdown', 'json'],
        default='markdown',
        help='输出格式（默认：markdown）'
    )
    parser.add_argument(
        '--min-year',
        type=int,
        help='最小年份筛选'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        help='最大年份筛选'
    )
    parser.add_argument(
        '--keywords',
        type=str,
        nargs='+',
        help='标题必须包含的关键词'
    )
    parser.add_argument(
        '--exclude-keywords',
        type=str,
        nargs='+',
        help='标题必须排除的关键词'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        help='每个领域只显示前 N 篇论文'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='禁用详细日志输出'
    )
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = DBLPAnalyzer(verbose=not args.quiet)
    
    # 执行分析
    result = analyzer.analyze_url(
        args.url,
        min_year=args.min_year,
        max_year=args.max_year,
        keywords=args.keywords,
        exclude_keywords=args.exclude_keywords
    )
    
    # 导出结果
    if args.format == 'markdown':
        analyzer.export_markdown(result, args.output, top_n=args.top_n)
    else:
        analyzer.export_json(result, args.output)
    
    print(f"\n✅ 分析完成！")
    if args.top_n and len(result['categories']) > 0:
        first_cat = list(result['categories'].values())[0]
        if len(first_cat) > args.top_n:
            print(f"⚠️  每个领域只显示前 {args.top_n} 篇，完整数据见输出文件")


if __name__ == "__main__":
    main()
