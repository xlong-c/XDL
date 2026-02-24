"""
精确分类器 - 基于摘要的论文精确分类

使用自定义分类体系对论文进行多标签分类，
支持细分领域、置信度评分。
"""

import csv
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import yaml


class PrecisionClassifier:
    """论文精确分类器"""
    
    def __init__(
        self,
        taxonomy_file: Optional[str] = None,
        custom_taxonomy: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        初始化分类器
        
        Args:
            taxonomy_file: 分类体系 YAML 文件路径
            custom_taxonomy: 自定义分类体系（覆盖文件）
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        
        # 加载分类体系
        if custom_taxonomy:
            self.taxonomy = custom_taxonomy
        elif taxonomy_file:
            self.taxonomy = self._load_taxonomy(taxonomy_file)
        else:
            self.taxonomy = self._get_default_taxonomy()
    
    def _load_taxonomy(self, filepath: str) -> Dict[str, Any]:
        """加载 YAML 分类体系"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"分类体系文件不存在：{filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config.get('taxonomy', config)
    
    def _get_default_taxonomy(self) -> Dict[str, Any]:
        """默认分类体系"""
        return {
            "AI/ML": {
                "keywords": [
                    "neural network", "deep learning", "machine learning",
                    "reinforcement learning", "transformer", "attention",
                    "representation learning", "generative", "llm"
                ],
                "subcategories": {
                    "Computer Vision": ["vision", "image", "detection", "segmentation"],
                    "NLP": ["language", "text", "nlp", "translation"],
                    "RL": ["reinforcement", "policy", "reward", "mdp"]
                }
            },
            "Data Mining": {
                "keywords": [
                    "clustering", "classification", "pattern mining",
                    "anomaly detection", "recommendation"
                ]
            },
            "Information Retrieval": {
                "keywords": [
                    "search", "retrieval", "ranking", "query",
                    "question answering"
                ]
            },
            "Database Systems": {
                "keywords": [
                    "database", "sql", "nosql", "query optimization",
                    "transaction", "indexing"
                ]
            },
            "Computer Networks": {
                "keywords": [
                    "network", "routing", "protocol", "wireless",
                    "iot", "5g", "sdn"
                ]
            },
            "Security & Privacy": {
                "keywords": [
                    "security", "cryptograph", "authentication",
                    "encryption", "attack", "privacy", "blockchain"
                ]
            },
            "Software Engineering": {
                "keywords": [
                    "software", "testing", "debugging", "refactoring",
                    "code", "devops", "microservice"
                ]
            },
            "HCI": {
                "keywords": [
                    "user interface", "usability", "accessibility",
                    "visualization", "hci", "interaction"
                ]
            },
            "Theory": {
                "keywords": [
                    "complexity", "algorithm", "optimization",
                    "graph theory", "proof", "approximation"
                ]
            },
            "HPC": {
                "keywords": [
                    "parallel", "distributed", "gpu", "cuda",
                    "high performance", "supercomputing"
                ]
            }
        }
    
    def classify_csv(
        self,
        input_csv: str,
        output_csv: str,
        title_col: str = 'title',
        abstract_col: str = 'abstract',
        min_score: float = 1.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        分类 CSV 文件中的论文
        
        Args:
            input_csv: 输入 CSV (含 title 和 abstract 列)
            output_csv: 输出 CSV (添加分类结果)
            title_col: 标题列名
            abstract_col: 摘要列名
            min_score: 最小分数阈值
        
        Returns:
            按类别组织的论文字典
        """
        # 读取 CSV
        papers = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                papers.append(row)
        
        if self.verbose:
            print(f"📊 开始分类 {len(papers)} 篇论文...")
        
        # 分类
        results = []
        by_category: Dict[str, List[Dict]] = {}
        
        for i, paper in enumerate(papers):
            title = paper.get(title_col, '')
            abstract = paper.get(abstract_col, '')
            
            if not title and not abstract:
                if self.verbose:
                    print(f"  ⚠️  第{i+1}篇：缺少标题和摘要，跳过")
                continue
            
            # 分类
            cats = self.classify_paper(title, abstract, min_score)
            
            # 合并结果
            result = {
                **paper,
                'primary_category': cats['primary']['name'] if cats['primary'] else 'Unclassified',
                'confidence': cats['primary']['confidence'] if cats['primary'] else 'low',
                'all_categories': '|'.join([c['name'] for c in cats['all']])
            }
            results.append(result)
            
            # 按类别组织
            if cats['primary']:
                cat_name = cats['primary']['name']
                if cat_name not in by_category:
                    by_category[cat_name] = []
                by_category[cat_name].append(result)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(papers)} 篇")
        
        # 写入输出 CSV
        if results:
            fieldnames = list(results[0].keys())
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            if self.verbose:
                print(f"\n✅ 完成！分类结果已保存至：{output_csv}")
                print(f"\n📊 类别分布:")
                for cat, papers_list in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True):
                    print(f"  {cat}: {len(papers_list)} 篇")
        
        return by_category
    
    def classify_paper(
        self,
        title: str,
        abstract: str,
        min_score: float = 1.0
    ) -> Dict[str, Any]:
        """
        分类单篇论文
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            min_score: 最小分数阈值
        
        Returns:
            {
                'primary': {'name': str, 'score': float, 'confidence': str},
                'all': [{'name': str, 'score': float, 'confidence': str}]
            }
        """
        text = f"{title} {abstract}".lower()
        
        scores = {}
        for category, config in self.taxonomy.items():
            # 获取关键词
            keywords = config.get('keywords', [])
            
            # 计算匹配分数
            match_count = sum(1 for kw in keywords if kw in text)
            
            # 子类别匹配 (加分)
            sub_matches = []
            if 'subcategories' in config:
                for sub_name, sub_keywords in config['subcategories'].items():
                    if any(kw in text for kw in sub_keywords):
                        sub_matches.append(sub_name)
                        match_count += 0.5
            
            if match_count >= min_score:
                scores[category] = {
                    'score': match_count,
                    'subcategories': sub_matches
                }
        
        # 排序
        sorted_cats = sorted(
            scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # 构建结果
        all_cats = []
        for cat_name, data in sorted_cats:
            confidence = self._calc_confidence(data['score'])
            all_cats.append({
                'name': cat_name,
                'score': data['score'],
                'confidence': confidence,
                'subcategories': data['subcategories']
            })
        
        primary = all_cats[0] if all_cats else None
        
        return {
            'primary': primary,
            'all': all_cats
        }
    
    def _calc_confidence(self, score: float) -> str:
        """计算置信度"""
        if score >= 3:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"
    
    def export_by_category(
        self,
        by_category: Dict[str, List[Dict]],
        output_dir: str,
        format: str = 'markdown'
    ):
        """
        按类别导出论文
        
        Args:
            by_category: classify_csv 返回的按类别组织的字典
            output_dir: 输出目录
            format: 输出格式 (markdown/csv)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for category, papers in by_category.items():
            if format == 'markdown':
                self._export_category_md(papers, category, output_path)
            elif format == 'csv':
                self._export_category_csv(papers, category, output_path)
    
    def _export_category_md(
        self,
        papers: List[Dict],
        category: str,
        output_dir: Path
    ):
        """导出为 Markdown"""
        lines = [
            f"# {category}",
            f"",
            f"**论文数**: {len(papers)}",
            f"",
            f"---",
            f""
        ]
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'No Title')
            authors = paper.get('authors', '')
            year = paper.get('year', '')
            doi = paper.get('doi', '')
            
            lines.append(f"### {i}. {title}")
            lines.append(f"")
            if authors:
                lines.append(f"**作者**: {authors}")
            if year:
                lines.append(f"**年份**: {year}")
            if doi:
                lines.append(f"**DOI**: [{doi}](https://doi.org/{doi})")
            
            # 显示置信度
            confidence = paper.get('confidence', '')
            emoji = "✅" if confidence == 'high' else "⚠️"
            lines.append(f"**置信度**: {emoji} {confidence}")
            
            # 显示子类别
            all_cats = paper.get('all_categories', '')
            if all_cats and all_cats != category:
                lines.append(f"**其他类别**: {all_cats}")
            
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")
        
        # 写入文件
        filename = self._sanitize_filename(category)
        output_file = output_dir / f"{filename}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        if self.verbose:
            print(f"  ✓ 导出 {category}: {output_file}")
    
    def _export_category_csv(
        self,
        papers: List[Dict],
        category: str,
        output_dir: Path
    ):
        """导出为 CSV"""
        filename = self._sanitize_filename(category)
        output_file = output_dir / f"{filename}.csv"
        
        if papers:
            fieldnames = list(papers[0].keys())
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(papers)
            
            if self.verbose:
                print(f"  ✓ 导出 {category}: {output_file}")
    
    def _sanitize_filename(self, name: str) -> str:
        """清理文件名"""
        return name.replace('/', '_').replace('\\', '_').replace(' ', '_')


# 命令行入口
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="基于摘要的论文精确分类"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 CSV (含 title 和 abstract 列)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="classified_papers.csv",
        help="输出 CSV 路径"
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        help="分类体系 YAML 文件"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="按类别导出目录 (可选)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="最小分数阈值"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="禁用详细日志"
    )
    
    args = parser.parse_args()
    
    # 创建分类器
    classifier = PrecisionClassifier(
        taxonomy_file=args.taxonomy,
        verbose=not args.quiet
    )
    
    # 分类
    by_category = classifier.classify_csv(
        input_csv=args.input,
        output_csv=args.output,
        min_score=args.min_score
    )
    
    # 按类别导出
    if args.output_dir:
        classifier.export_by_category(
            by_category=by_category,
            output_dir=args.output_dir,
            format='markdown'
        )
        
        if args.verbose:
            print(f"\n✅ 按类别导出完成：{args.output_dir}/")


if __name__ == "__main__":
    main()
