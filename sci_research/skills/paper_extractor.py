"""
论文关键信息提取器

从论文标题和摘要中提取关键信息：
- 研究方法
- 使用的数据集
- 评估指标
- 代码链接
"""

import csv
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path


class PaperExtractor:
    """论文关键信息提取器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化提取器
        
        Args:
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        
        # 数据集关键词
        self.dataset_patterns = {
            "Cityscapes": ["cityscapes"],
            "COCO": ["coco", "ms-coco"],
            "ImageNet": ["imagenet"],
            "PASCAL VOC": ["pascal voc", "voc2007", "voc2012"],
            "ADE20K": ["ade20k", "ade-20k"],
            "Mapillary": ["mapillary"],
            "BDD100K": ["bdd100k", "bdd-100k"],
            "CIFAR": ["cifar10", "cifar100", "cifar"],
            "MNIST": ["mnist"]
        }
        
        # 指标关键词
        self.metric_patterns = {
            "mIoU": ["miou", "mean iou"],
            "Accuracy": ["accuracy", "acc"],
            "mAP": ["map", "mean average precision"],
            "F1": ["f1 score", "f1-measure"],
            "FPS": ["fps", "frames per second", "inference time"],
            "Params": ["params", "parameters", "model size"],
            "FLOPs": ["flops", "gflops", "computational cost"],
            "AUC": ["auc", "area under curve"]
        }
        
        # 方法类型关键词
        self.method_patterns = {
            "Semantic Segmentation": ["semantic segmentation"],
            "Instance Segmentation": ["instance segmentation"],
            "Panoptic Segmentation": ["panoptic segmentation"],
            "Object Detection": ["object detection", "object detector"],
            "Image Classification": ["image classification", "classifier"],
            "Knowledge Distillation": ["knowledge distillation", "distillation", "teacher-student"],
            "Quantization": ["quantization", "low-bit", "8-bit", "4-bit"],
            "Pruning": ["pruning", "sparse", "sparsity"]
        }
    
    def extract_csv(
        self,
        input_csv: str,
        output_csv: str,
        title_col: str = 'title',
        abstract_col: str = 'abstract'
    ) -> List[Dict[str, Any]]:
        """
        从 CSV 批量提取论文信息
        
        Args:
            input_csv: 输入 CSV (含 title 和 abstract 列)
            output_csv: 输出 CSV (添加提取的信息)
            title_col: 标题列名
            abstract_col: 摘要列名
        
        Returns:
            提取结果列表
        """
        papers = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                papers.append(row)
        
        if self.verbose:
            print(f"📊 开始从 {len(papers)} 篇论文中提取关键信息...")
        
        results = []
        for i, paper in enumerate(papers):
            title = paper.get(title_col, '')
            abstract = paper.get(abstract_col, '')
            
            if not title and not abstract:
                continue
            
            extracted = self.extract_paper(title, abstract)
            result = {**paper, **extracted}
            results.append(result)
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(papers)} 篇")
        
        if results:
            fieldnames = list(results[0].keys())
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            if self.verbose:
                print(f"\n✅ 提取完成！结果已保存至：{output_csv}")
        
        return results
    
    def extract_paper(
        self,
        title: str,
        abstract: str
    ) -> Dict[str, Any]:
        """
        提取单篇论文的关键信息
        
        Args:
            title: 论文标题
            abstract: 论文摘要
        
        Returns:
            提取的信息字典
        """
        text = f"{title} {abstract}".lower()
        
        # 提取方法类型
        methods = []
        for method_name, keywords in self.method_patterns.items():
            if any(kw in text for kw in keywords):
                methods.append(method_name)
        
        # 提取数据集
        datasets = []
        for dataset_name, keywords in self.dataset_patterns.items():
            if any(kw in text for kw in keywords):
                datasets.append(dataset_name)
        
        # 提取指标
        metrics = {}
        for metric_name, keywords in self.metric_patterns.items():
            if any(kw in text for kw in keywords):
                metrics[metric_name] = True
        
        # 提取数值型指标
        metric_values = self._extract_metric_values(text)
        
        # 提取代码链接 (从摘要中找 GitHub 等)
        code_url = self._extract_code_url(abstract)
        
        # 提取关键贡献点 (基于摘要句子分析)
        contributions = self._extract_contributions(abstract)
        
        return {
            'methods': '|'.join(methods),
            'datasets': '|'.join(datasets),
            'metrics': '|'.join(metrics.keys()),
            'metric_values': json.dumps(metric_values, ensure_ascii=False),
            'code_url': code_url,
            'contributions': '|'.join(contributions[:3])
        }
    
    def _extract_metric_values(self, text: str) -> Dict[str, float]:
        """提取具体指标数值"""
        values = {}
        
        # mIoU: "achieves 78.5% mIoU" 或 "mIoU of 78.5%"
        patterns = [
            (r"miou[:\s]+(\d+\.?\d*)\s*%", "mIoU"),
            (r"(\d+\.?\d*)\s*%\s*miou", "mIoU"),
            (r"accuracy[:\s]+(\d+\.?\d*)\s*%", "Accuracy"),
            (r"(\d+\.?\d*)\s*%\s*accuracy", "Accuracy"),
            (r"map[:\s]+(\d+\.?\d*)\s*%", "mAP"),
            (r"(\d+\.?\d*)\s*%\s*map", "mAP"),
            (r"flops[:\s]+(\d+\.?\d*)\s*(g)?flops", "FLOPs"),
            (r"params[:\s]+(\d+\.?\d*)\s*(m)?", "Params"),
            (r"fps[:\s]+(\d+\.?\d*)", "FPS")
        ]
        
        for pattern, metric_name in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    values[metric_name] = value
                except:
                    pass
        
        return values
    
    def _extract_code_url(self, abstract: str) -> str:
        """从摘要中提取代码链接"""
        # GitHub URL
        github_pattern = r"https?://github\.com/[^\s]+"
        matches = re.findall(github_pattern, abstract)
        if matches:
            return matches[0]
        
        # arXiv abstract (可选)
        arxiv_pattern = r"https?://arxiv\.org/[^\s]+"
        matches = re.findall(arxiv_pattern, abstract)
        if matches:
            return matches[0]
        
        return ""
    
    def _extract_contributions(self, abstract: str) -> List[str]:
        """从摘要中提取关键贡献点"""
        contributions = []
        
        # 贡献指示词
        indicators = [
            "we propose",
            "we present",
            "we introduce",
            "our method",
            "our approach",
            "the key contribution",
            "to address",
            "in this paper"
        ]
        
        # 分割句子
        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in indicators):
                # 清理句子
                cleaned = sentence.strip()
                if len(cleaned) > 20 and len(cleaned) < 300:
                    contributions.append(cleaned)
        
        return contributions[:3]
    
    def export_summary(
        self,
        extracted_papers: List[Dict[str, Any]],
        output_file: str,
        group_by: str = 'method'
    ):
        """
        导出汇总统计
        
        Args:
            extracted_papers: extract_csv 返回的结果
            output_file: 输出文件路径
            group_by: 分组方式 (method/dataset)
        """
        stats = {}
        
        for paper in extracted_papers:
            if group_by == 'method':
                key = paper.get('methods', 'Unknown')
            elif group_by == 'dataset':
                key = paper.get('datasets', 'Unknown')
            else:
                key = 'All'
            
            if key not in stats:
                stats[key] = []
            stats[key].append(paper)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 论文关键信息汇总\n\n")
            f.write(f"**分组方式**: {group_by}\n")
            f.write(f"**总论文数**: {len(extracted_papers)}\n\n")
            
            for category, papers in sorted(stats.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"## {category} ({len(papers)}篇)\n\n")
                
                for i, paper in enumerate(papers, 1):
                    title = paper.get('title', 'No Title')
                    datasets = paper.get('datasets', '')
                    metrics = paper.get('metrics', '')
                    code_url = paper.get('code_url', '')
                    
                    f.write(f"### {i}. {title}\n\n")
                    if datasets:
                        f.write(f"**数据集**: {datasets}\n")
                    if metrics:
                        f.write(f"**指标**: {metrics}\n")
                    if code_url:
                        f.write(f"**代码**: [{code_url}]({code_url})\n")
                    f.write(f"\n---\n\n")
        
        if self.verbose:
            print(f"✓ 汇总报告已保存至：{output_file}")


# 命令行入口
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="论文关键信息提取"
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
        default="extracted_info.csv",
        help="输出 CSV 路径"
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="导出汇总报告文件路径"
    )
    parser.add_argument(
        "--group-by",
        type=str,
        choices=['method', 'dataset'],
        default='method',
        help="汇总分组方式"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="禁用详细日志"
    )
    
    args = parser.parse_args()
    
    extractor = PaperExtractor(verbose=not args.quiet)
    
    # 提取信息
    results = extractor.extract_csv(
        input_csv=args.input,
        output_csv=args.output
    )
    
    # 导出汇总
    if args.summary:
        extractor.export_summary(
            extracted_papers=results,
            output_file=args.summary,
            group_by=args.group_by
        )
    
    # 打印统计
    if results and not args.quiet:
        with_codes = sum(1 for r in results if r.get('code_url'))
        with_datasets = sum(1 for r in results if r.get('datasets'))
        
        print(f"\n📊 提取统计:")
        print(f"  有代码链接：{with_codes}/{len(results)} ({with_codes/len(results)*100:.1f}%)")
        print(f"  提到数据集：{with_datasets}/{len(results)} ({with_datasets/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
