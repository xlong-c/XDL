"""
深入分析报告生成器

基于分类和提取的结果生成深入分析报告：
- 对比表格
- 时间线分析
- 趋势总结
- Top N 推荐
"""

import csv
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import Counter
from datetime import datetime


class DeepAnalysisReport:
    """深入分析报告生成器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化报告生成器
        
        Args:
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
    
    def generate_report(
        self,
        input_csv: str,
        output_md: str,
        top_n: int = 10,
        include_timeline: bool = True,
        include_comparison: bool = True
    ) -> str:
        """
        生成深入分析报告
        
        Args:
            input_csv: 输入 CSV (含分类和提取的信息)
            output_md: 输出 Markdown 文件路径
            top_n: 推荐 Top N 数量
            include_timeline: 是否包含时间线
            include_comparison: 是否包含对比表格
        
        Returns:
            生成的 Markdown 内容
        """
        # 读取数据
        papers = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                papers.append(row)
        
        if self.verbose:
            print(f"📊 开始生成深入分析报告，共 {len(papers)} 篇论文...")
        
        # 生成报告
        lines = []
        
        # 1. 标题和摘要
        lines.extend(self._generate_header(papers))
        
        # 2. 领域分布
        lines.extend(self._generate_distribution(papers))
        
        # 3. 时间线分析
        if include_timeline:
            lines.extend(self._generate_timeline(papers))
        
        # 4. 方法对比表格
        if include_comparison:
            lines.extend(self._generate_comparison_table(papers))
        
        # 5. 数据集统计
        lines.extend(self._generate_dataset_stats(papers))
        
        # 6. 指标统计
        lines.extend(self._generate_metric_stats(papers))
        
        # 7. Top N 推荐
        lines.extend(self._generate_top_recommendations(papers, top_n))
        
        # 8. 开放问题与趋势
        lines.extend(self._generate_trends(papers))
        
        # 写入文件
        content = "\n".join(lines)
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if self.verbose:
            print(f"\n✅ 深入分析报告已生成：{output_md}")
        
        return content
    
    def _generate_header(self, papers: List[Dict]) -> List[str]:
        """生成报告标题"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        lines = [
            "# 深入分析报告\n",
            f"**生成时间**: {now}",
            f"**总论文数**: {len(papers)}\n",
            "---\n"
        ]
        
        return lines
    
    def _generate_distribution(self, papers: List[Dict]) -> List[str]:
        """生成领域分布统计"""
        lines = [
            "## 领域分布\n"
        ]
        
        # 按类别统计
        category_counter = Counter()
        for paper in papers:
            category = paper.get('primary_category', 'Unclassified')
            category_counter[category] += 1
        
        # 表格
        lines.append("| 领域 | 论文数 | 占比 |")
        lines.append("|------|--------|------|")
        
        total = len(papers)
        for category, count in category_counter.most_common():
            percentage = count / total * 100
            lines.append(f"| {category} | {count} | {percentage:.1f}% |")
        
        lines.append("\n---\n")
        return lines
    
    def _generate_timeline(self, papers: List[Dict]) -> List[str]:
        """生成时间线分析"""
        lines = [
            "## 时间线分析\n"
        ]
        
        # 按年份统计
        year_counter = Counter()
        for paper in papers:
            year = paper.get('year', '')
            if year:
                try:
                    year_int = int(year)
                    year_counter[year_int] += 1
                except:
                    pass
        
        if year_counter:
            lines.append("### 论文发表趋势\n")
            lines.append("| 年份 | 论文数 |")
            lines.append("|------|--------|")
            
            for year in sorted(year_counter.keys()):
                lines.append(f"| {year} | {year_counter[year]} |")
            
            # 趋势描述
            years = sorted(year_counter.keys())
            if len(years) >= 2:
                first_year = years[0]
                last_year = years[-1]
                first_count = year_counter[first_year]
                last_count = year_counter[last_year]
                
                if last_count > first_year:
                    growth = (last_count - first_count) / first_count * 100
                    lines.append(f"\n**趋势**: 从 {first_year} 年到 {last_year} 年，论文数量增长了 {growth:.1f}%\n")
            
            lines.append("\n---\n")
        
        return lines
    
    def _generate_comparison_table(self, papers: List[Dict]) -> List[str]:
        """生成方法对比表格"""
        lines = [
            "## 方法对比\n"
        ]
        
        # 按方法分组
        method_papers = {}
        for paper in papers:
            methods = paper.get('methods', '')
            if methods:
                for method in methods.split('|'):
                    if method not in method_papers:
                        method_papers[method] = []
                    method_papers[method].append(paper)
        
        if not method_papers:
            lines.append("*暂无方法信息*\n")
            lines.append("\n---\n")
            return lines
        
        # 生成每个方法的详细对比
        for method_name, method_papers_list in sorted(method_papers.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"### {method_name} ({len(method_papers_list)}篇)\n")
            
            # 提取对比信息
            table_data = []
            for paper in method_papers_list:
                title = paper.get('title', 'No Title')[:50] + "..." if len(paper.get('title', '')) > 50 else paper.get('title', '')
                datasets = paper.get('datasets', '')
                
                # 解析 metric_values
                metric_values_str = paper.get('metric_values', '{}')
                try:
                    metric_values = json.loads(metric_values_str)
                except:
                    metric_values = {}
                
                # 提取关键指标
                miou = metric_values.get('mIoU', '')
                accuracy = metric_values.get('Accuracy', '')
                fps = metric_values.get('FPS', '')
                
                table_data.append({
                    'title': title,
                    'datasets': datasets,
                    'mIoU': f"{miou}%" if miou else '',
                    'Accuracy': f"{accuracy}%" if accuracy else '',
                    'FPS': str(fps) if fps else ''
                })
            
            if table_data:
                lines.append("| 论文 | 数据集 | mIoU | Accuracy | FPS |")
                lines.append("|------|--------|------|----------|-----|")
                
                for row in table_data:
                    lines.append(
                        f"| {row['title']} | {row['datasets']} | {row['mIoU']} | {row['Accuracy']} | {row['FPS']} |"
                    )
                
                lines.append("\n")
        
        lines.append("---\n")
        return lines
    
    def _generate_dataset_stats(self, papers: List[Dict]) -> List[str]:
        """生成数据集统计"""
        lines = [
            "## 使用数据集统计\n"
        ]
        
        dataset_counter = Counter()
        for paper in papers:
            datasets = paper.get('datasets', '')
            if datasets:
                for dataset in datasets.split('|'):
                    dataset_counter[dataset] += 1
        
        if dataset_counter:
            lines.append("| 数据集 | 使用次数 |")
            lines.append("|--------|----------|")
            
            for dataset, count in dataset_counter.most_common(10):
                lines.append(f"| {dataset} | {count} |")
            
            lines.append("\n---\n")
        else:
            lines.append("*暂无数据集信息*\n")
            lines.append("---\n")
        
        return lines
    
    def _generate_metric_stats(self, papers: List[Dict]) -> List[str]:
        """生成评估指标统计"""
        lines = [
            "## 评估指标统计\n"
        ]
        
        metric_counter = Counter()
        for paper in papers:
            metrics = paper.get('metrics', '')
            if metrics:
                for metric in metrics.split('|'):
                    metric_counter[metric] += 1
        
        if metric_counter:
            lines.append("| 指标 | 使用次数 |")
            lines.append("|------|----------|")
            
            for metric, count in metric_counter.most_common():
                lines.append(f"| {metric} | {count} |")
            
            lines.append("\n---\n")
        else:
            lines.append("*暂无指标信息*\n")
            lines.append("---\n")
        
        return lines
    
    def _generate_top_recommendations(
        self,
        papers: List[Dict],
        top_n: int
    ) -> List[str]:
        """生成 Top N 推荐"""
        lines = [
            f"## 推荐论文 (Top {top_n})\n"
        ]
        
        # 排序策略：
        # 1. 有代码链接优先
        # 2. 提到多个数据集优先
        # 3. 有具体指标数值优先
        
        scored_papers = []
        for paper in papers:
            score = 0
            
            # 有代码链接 +3 分
            if paper.get('code_url'):
                score += 3
            
            # 提到数据集数量
            datasets = paper.get('datasets', '')
            if datasets:
                score += len(datasets.split('|'))
            
            # 有指标数值 +1
            metric_values_str = paper.get('metric_values', '{}')
            try:
                metric_values = json.loads(metric_values_str)
                if metric_values:
                    score += 1
            except:
                pass
            
            # 有贡献描述 +1
            if paper.get('contributions'):
                score += 1
            
            scored_papers.append((paper, score))
        
        # 按分数排序
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # 输出 Top N
        for i, (paper, score) in enumerate(scored_papers[:top_n], 1):
            title = paper.get('title', 'No Title')
            authors = paper.get('authors', '')
            year = paper.get('year', '')
            doi = paper.get('doi', '')
            code_url = paper.get('code_url', '')
            contributions = paper.get('contributions', '')
            
            lines.append(f"### {i}. {title}\n")
            
            if authors:
                lines.append(f"**作者**: {authors}\n")
            if year:
                lines.append(f"**年份**: {year}\n")
            if doi:
                lines.append(f"**DOI**: [{doi}](https://doi.org/{doi})\n")
            if code_url:
                lines.append(f"**代码**: [{code_url}]({code_url})\n")
            
            if contributions:
                lines.append(f"\n**关键贡献**:\n")
                for contrib in contributions.split('|')[:2]:
                    lines.append(f"- {contrib}\n")
            
            lines.append(f"\n---\n\n")
        
        return lines
    
    def _generate_trends(self, papers: List[Dict]) -> List[str]:
        """生成趋势分析"""
        lines = [
            "## 发展趋势与开放问题\n"
        ]
        
        # 分析最近年份的趋势
        recent_papers = []
        for paper in papers:
            year = paper.get('year', '')
            if year:
                try:
                    if int(year) >= 2023:
                        recent_papers.append(paper)
                except:
                    pass
        
        if recent_papers:
            lines.append("### 近期趋势 (2023-2024)\n")
            
            # 高频关键词
            all_methods = []
            for paper in recent_papers:
                methods = paper.get('methods', '')
                if methods:
                    all_methods.extend(methods.split('|'))
            
            if all_methods:
                method_counter = Counter(all_methods)
                top_methods = method_counter.most_common(5)
                
                lines.append("**热门方法**:\n")
                for method, count in top_methods:
                    lines.append(f"- {method} ({count}篇)\n")
                
                lines.append("\n")
        
        lines.append("### 开放问题\n")
        lines.append("基于以上分析，建议关注以下方向:\n\n")
        lines.append("1. **高效模型设计**: 端侧/移动端部署需求增长\n")
        lines.append("2. **多任务学习**: 分割 + 检测 + 分类的联合优化\n")
        lines.append("3. **数据效率**: 小样本/弱监督学习方法\n")
        lines.append("4. **可解释性**: 模型决策过程的可视化与分析\n")
        
        lines.append("\n---\n")
        return lines


# 命令行入口
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="生成深入分析报告"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 CSV (含分类和提取的信息)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="deep_analysis.md",
        help="输出 Markdown 文件路径"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="推荐 Top N 数量"
    )
    parser.add_argument(
        "--no-timeline",
        action="store_true",
        help="不包含时间线分析"
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="不包含对比表格"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="禁用详细日志"
    )
    
    args = parser.parse_args()
    
    report_generator = DeepAnalysisReport(verbose=not args.quiet)
    
    report_generator.generate_report(
        input_csv=args.input,
        output_md=args.output,
        top_n=args.top_n,
        include_timeline=not args.no_timeline,
        include_comparison=not args.no_comparison
    )


if __name__ == "__main__":
    main()
