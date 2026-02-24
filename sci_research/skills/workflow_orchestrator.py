"""
完整工作流编排器

一键执行完整的科研文献分析流程：
1. DBLP 分析
2. 摘要获取
3. 精确分类
4. 信息提取
5. 报告生成
6. Zotero 同步
"""

import csv
import json
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime


class WorkflowOrchestrator:
    """工作流编排器"""
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        output_dir: str = "./analysis_output",
        verbose: bool = True
    ):
        """
        初始化工作流编排器
        
        Args:
            config_file: 配置文件路径 (YAML)
            output_dir: 输出目录
            verbose: 是否打印详细日志
        """
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.config = self._load_config(config_file) if config_file else {}
        
        # 状态追踪
        self.state = {
            'current_step': 0,
            'total_steps': 5,
            'files': {},
            'errors': []
        }
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def run(
        self,
        dblp_url: str,
        keywords: Optional[List[str]] = None,
        exclude_keywords: Optional[List[str]] = None,
        zotero_sync: bool = False,
        zotero_config: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        执行完整工作流
        
        Args:
            dblp_url: DBLP 页面 URL
            keywords: 关键词列表
            exclude_keywords: 排除关键词列表
            zotero_sync: 是否同步到 Zotero
            zotero_config: Zotero 配置
        
        Returns:
            工作流执行结果
        """
        start_time = datetime.now()
        
        if self.verbose:
            print("\n" + "="*60)
            print("🚀 开始执行完整科研工作流")
            print("="*60)
            print(f"📌 DBLP URL: {dblp_url}")
            print(f"📌 输出目录：{self.output_dir}")
            print(f"📌 关键词：{keywords or '无'}")
            print("="*60 + "\n")
        
        try:
            # 步骤 1: DBLP 分析
            self._step_1_dblp_analysis(
                dblp_url=dblp_url,
                keywords=keywords,
                exclude_keywords=exclude_keywords
            )
            
            # 步骤 2: 摘要获取
            self._step_2_abstract_fetching()
            
            # 步骤 3: 精确分类
            self._step_3_precision_classification()
            
            # 步骤 4: 信息提取
            self._step_4_information_extraction()
            
            # 步骤 5: 报告生成
            self._step_5_report_generation()
            
            # 步骤 6 (可选): Zotero 同步
            if zotero_sync and zotero_config:
                self._step_6_zotero_sync(zotero_config)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if self.verbose:
                print("\n" + "="*60)
                print("✅ 工作流执行完成！")
                print(f"⏱️  总耗时：{duration:.1f} 秒")
                print(f"📁 输出目录：{self.output_dir}")
                print("="*60 + "\n")
            
            return {
                'success': True,
                'duration': duration,
                'files': self.state['files'],
                'errors': self.state['errors']
            }
        
        except Exception as e:
            if self.verbose:
                print(f"\n❌ 工作流执行失败：{e}")
            
            return {
                'success': False,
                'error': str(e),
                'files': self.state['files'],
                'errors': self.state['errors']
            }
    
    def _update_progress(self, step_name: str, step_num: int):
        """更新进度显示"""
        self.state['current_step'] = step_num
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"步骤 {step_num}/{self.state['total_steps']}: {step_name}")
            print(f"{'='*60}")
    
    def _step_1_dblp_analysis(
        self,
        dblp_url: str,
        keywords: Optional[List[str]],
        exclude_keywords: Optional[List[str]]
    ):
        """步骤 1: DBLP 分析"""
        self._update_progress("DBLP 论文分析", 1)
        
        from skills.dblp_analyzer import DBLPAnalyzer
        
        output_csv = self.output_dir / "01_raw_papers.csv"
        
        analyzer = DBLPAnalyzer(verbose=self.verbose)
        result = analyzer.analyze_url(
            url=dblp_url,
            keywords=keywords,
            exclude_keywords=exclude_keywords
        )
        
        analyzer.export_csv(result, str(output_csv))
        
        self.state['files']['raw_csv'] = str(output_csv)
        
        if self.verbose:
            print(f"✓ 已保存：{output_csv}")
            print(f"  论文总数：{result['total_papers']}")
            print(f"  领域分布：{result['distribution']}")
    
    def _step_2_abstract_fetching(self):
        """步骤 2: 摘要获取"""
        self._update_progress("批量摘要获取", 2)
        
        from skills.abstract_fetcher import AbstractFetcher
        
        input_csv = self.state['files']['raw_csv']
        output_csv = self.output_dir / "02_with_abstracts.csv"
        
        fetcher = AbstractFetcher(verbose=self.verbose)
        
        # 读取 DOI
        dois = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                doi = row.get('doi', '')
                if doi:
                    dois.append(doi)
        
        if self.verbose:
            print(f"→ 找到 {len(dois)} 个 DOI")
        
        # 批量获取摘要
        abstracts = fetcher.batch_fetch(dois=dois)
        
        # 合并到 CSV
        fetcher.merge_with_csv(
            input_csv=input_csv,
            results=abstracts,
            output_csv=str(output_csv),
            id_column='doi'
        )
        
        self.state['files']['abstracts_csv'] = str(output_csv)
        
        if self.verbose:
            print(f"✓ 已保存：{output_csv}")
            print(f"  成功获取：{len(abstracts)} 个摘要")
    
    def _step_3_precision_classification(self):
        """步骤 3: 精确分类"""
        self._update_progress("精确分类", 3)
        
        from skills.precision_classifier import PrecisionClassifier
        
        input_csv = self.state['files']['abstracts_csv']
        output_csv = self.output_dir / "03_classified.csv"
        output_dir = self.output_dir / "by_category"
        
        # 获取分类体系
        taxonomy_file = self.config.get('taxonomy_file')
        
        classifier = PrecisionClassifier(
            taxonomy_file=taxonomy_file,
            verbose=self.verbose
        )
        
        by_category = classifier.classify_csv(
            input_csv=str(input_csv),
            output_csv=str(output_csv)
        )
        
        # 按类别导出
        classifier.export_by_category(
            by_category=by_category,
            output_dir=str(output_dir),
            format='markdown'
        )
        
        self.state['files']['classified_csv'] = str(output_csv)
        self.state['files']['by_category_dir'] = str(output_dir)
        
        if self.verbose:
            print(f"✓ 已保存：{output_csv}")
            print(f"✓ 按类别导出：{output_dir}")
            print(f"  类别数：{len(by_category)}")
    
    def _step_4_information_extraction(self):
        """步骤 4: 信息提取"""
        self._update_progress("关键信息提取", 4)
        
        from skills.paper_extractor import PaperExtractor
        
        input_csv = self.state['files']['classified_csv']
        output_csv = self.output_dir / "04_extracted.csv"
        summary_md = self.output_dir / "method_summary.md"
        
        extractor = PaperExtractor(verbose=self.verbose)
        results = extractor.extract_csv(
            input_csv=str(input_csv),
            output_csv=str(output_csv)
        )
        
        # 导出汇总
        extractor.export_summary(
            extracted_papers=results,
            output_file=str(summary_md),
            group_by='method'
        )
        
        self.state['files']['extracted_csv'] = str(output_csv)
        self.state['files']['summary_md'] = str(summary_md)
        
        if self.verbose:
            print(f"✓ 已保存：{output_csv}")
            print(f"✓ 方法汇总：{summary_md}")
    
    def _step_5_report_generation(self):
        """步骤 5: 报告生成"""
        self._update_progress("深入分析报告", 5)
        
        from skills.deep_analysis_report import DeepAnalysisReport
        
        input_csv = self.state['files']['extracted_csv']
        output_md = self.output_dir / "05_deep_analysis.md"
        
        report_generator = DeepAnalysisReport(verbose=self.verbose)
        
        report_generator.generate_report(
            input_csv=str(input_csv),
            output_md=str(output_md),
            top_n=self.config.get('analysis', {}).get('top_n_recommendations', 10),
            include_timeline=self.config.get('analysis', {}).get('include_timeline', True),
            include_comparison=self.config.get('analysis', {}).get('include_comparison_table', True)
        )
        
        self.state['files']['deep_analysis_md'] = str(output_md)
        
        if self.verbose:
            print(f"✓ 已保存：{output_md}")
    
    def _step_6_zotero_sync(self, zotero_config: Dict[str, str]):
        """步骤 6: Zotero 同步"""
        self._update_progress("Zotero 同步", 6)
        
        from skills.zotero_dual_sync import ZoteroDualSync
        
        input_csv = self.state['files']['extracted_csv']
        
        sync = ZoteroDualSync(
            library_id=zotero_config['library_id'],
            api_key=zotero_config['api_key'],
            library_type=zotero_config.get('library_type', 'user'),
            verbose=self.verbose
        )
        
        # 读取论文
        papers = []
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                papers.append(row)
        
        # 获取目录配置
        venue_path = zotero_config.get('venue_path', 'Journal/Unknown')
        topic_paths = zotero_config.get('topic_paths', ['Unclassified'])
        
        if isinstance(topic_paths, str):
            topic_paths = [topic_paths]
        
        # 导入
        result = sync.batch_import(
            papers=papers,
            venue_path=venue_path,
            topic_paths=topic_paths,
            auto_tags=True
        )
        
        self.state['files']['zotero_result'] = result
        
        if self.verbose:
            print(f"✓ Zotero 同步完成")
            print(f"  成功：{result['success']}")
            print(f"  失败：{result['failed']}")


# 命令行入口
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="完整科研工作流编排"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径 (YAML)"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="DBLP 页面 URL"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs='+',
        help="关键词列表"
    )
    parser.add_argument(
        "--exclude-keywords",
        type=str,
        nargs='+',
        help="排除关键词列表"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_output",
        help="输出目录"
    )
    parser.add_argument(
        "--zotero-sync",
        action="store_true",
        help="同步到 Zotero"
    )
    parser.add_argument(
        "--zotero-library-id",
        type=str,
        help="Zotero 图书馆 ID"
    )
    parser.add_argument(
        "--zotero-api-key",
        type=str,
        help="Zotero API 密钥"
    )
    parser.add_argument(
        "--zotero-venue-path",
        type=str,
        help="Zotero Venue 目录路径"
    )
    parser.add_argument(
        "--zotero-topic-paths",
        type=str,
        nargs='+',
        help="Zotero Topic 目录路径列表"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="禁用详细日志"
    )
    
    args = parser.parse_args()
    
    # 构建 Zotero 配置
    zotero_config = None
    if args.zotero_sync:
        zotero_config = {
            'library_id': args.zotero_library_id,
            'api_key': args.zotero_api_key,
            'library_type': 'user',
            'venue_path': args.zotero_venue_path or 'Journal/Unknown',
            'topic_paths': args.zotero_topic_paths or ['Unclassified']
        }
    
    # 创建工作流
    orchestrator = WorkflowOrchestrator(
        config_file=args.config,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    # 执行工作流
    result = orchestrator.run(
        dblp_url=args.url,
        keywords=args.keywords,
        exclude_keywords=args.exclude_keywords,
        zotero_sync=args.zotero_sync,
        zotero_config=zotero_config
    )
    
    # 输出结果
    if result['success']:
        print(f"\n✅ 工作流成功完成！")
        print(f"⏱️  总耗时：{result['duration']:.1f} 秒")
    else:
        print(f"\n❌ 工作流失败：{result['error']}")
    
    print(f"\n📁 输出文件:")
    for name, path in result['files'].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
