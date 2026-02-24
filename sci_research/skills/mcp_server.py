"""
MCP Server for Scientific Research Tools
科研工具 MCP 服务器 - 将 Python 研究工具暴露为 MCP 工具接口
"""

import sys
import json
import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path

# Add sci_research to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.dblp_analyzer import DBLPAnalyzer
from skills.abstract_fetcher import AbstractFetcher
from skills.precision_classifier import PrecisionClassifier
from skills.paper_extractor import PaperExtractor
from skills.deep_analysis_report import DeepAnalysisReport
from skills.zotero_dual_sync import ZoteroDualSync
from skills.workflow_orchestrator import WorkflowOrchestrator


class MCPServer:
    """MCP 服务器实现"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.tools = {
            "dblp_analyze": self.dblp_analyze,
            "fetch_abstracts": self.fetch_abstracts,
            "classify_papers": self.classify_papers,
            "extract_paper_info": self.extract_paper_info,
            "generate_analysis_report": self.generate_analysis_report,
            "run_full_workflow": self.run_full_workflow,
        }
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 MCP 请求"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return await self.initialize(params)
            elif method == "tools/list":
                return await self.list_tools()
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                return await self.call_tool(tool_name, tool_args)
            elif method == "notifications/initialized":
                return {"jsonrpc": "2.0", "result": {}}
            else:
                return self._error_response(request_id, f"Unknown method: {method}")
        except Exception as e:
            return self._error_response(request_id, str(e))
    
    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """初始化 MCP 连接"""
        return {
            "jsonrpc": "2.0",
            "id": params.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "sci-research-mcp",
                    "version": "1.0.0"
                }
            }
        }
    
    async def list_tools(self) -> Dict[str, Any]:
        """列出所有可用工具"""
        tools = [
            {
                "name": "dblp_analyze",
                "description": "分析 DBLP 期刊/会议页面，按规则自动分类论文到 10+ 计算机科学领域。零 token 消耗，~50 篇/秒。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "DBLP 期刊/会议页面 URL"
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "标题必须包含的关键词列表"
                        },
                        "exclude_keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "标题必须排除的关键词列表"
                        },
                        "min_year": {"type": "integer"},
                        "max_year": {"type": "integer"},
                        "output_format": {
                            "type": "string",
                            "enum": ["markdown", "csv", "json"]
                        }
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "fetch_abstracts",
                "description": "批量从 Crossref/arXiv API 获取论文摘要，支持缓存和并发请求。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dois": {"type": "array", "items": {"type": "string"}},
                        "arxiv_ids": {"type": "array", "items": {"type": "string"}},
                        "max_workers": {"type": "integer", "default": 5},
                        "cache_dir": {"type": "string"}
                    }
                }
            },
            {
                "name": "classify_papers",
                "description": "基于标题和摘要对论文进行精确多标签分类，支持自定义分类体系和子类别。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "papers": {
                            "type": "array",
                            "items": {"type": "object"}
                        },
                        "taxonomy_file": {"type": "string"},
                        "min_score": {"type": "number"},
                        "output_dir": {"type": "string"}
                    },
                    "required": ["papers"]
                }
            },
            {
                "name": "extract_paper_info",
                "description": "从论文标题和摘要中提取关键信息：方法、数据集、指标、代码链接、贡献点。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "papers": {
                            "type": "array",
                            "items": {"type": "object"}
                        },
                        "group_by": {
                            "type": "string",
                            "enum": ["method", "dataset"]
                        }
                    },
                    "required": ["papers"]
                }
            },
            {
                "name": "generate_analysis_report",
                "description": "基于分类和提取结果生成深入分析报告：对比表格、时间线、趋势总结、Top N 推荐。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "papers": {"type": "array", "items": {"type": "object"}},
                        "top_n": {"type": "integer"},
                        "include_timeline": {"type": "boolean"},
                        "include_comparison": {"type": "boolean"}
                    },
                    "required": ["papers"]
                }
            },
            {
                "name": "run_full_workflow",
                "description": "一键执行完整科研工作流：DBLP 分析 → 摘要获取 → 分类 → 信息提取 → 报告生成。",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dblp_url": {"type": "string"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "exclude_keywords": {"type": "array", "items": {"type": "string"}},
                        "output_dir": {"type": "string"},
                        "zotero_sync": {"type": "boolean"},
                        "zotero_config": {"type": "object"}
                    },
                    "required": ["dblp_url"]
                }
            }
        ]
        
        return {
            "jsonrpc": "2.0",
            "result": {"tools": tools}
        }
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用指定工具"""
        if tool_name not in self.tools:
            return self._error_response(None, f"Unknown tool: {tool_name}")
        
        try:
            result = await self.tools[tool_name](arguments)
            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, indent=2)}]
                }
            }
        except Exception as e:
            return self._error_response(None, f"Tool error: {str(e)}")
    
    async def dblp_analyze(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """DBLP 分析工具"""
        url = args.get("url")
        if not url:
            raise ValueError("url is required")
        
        analyzer = DBLPAnalyzer(verbose=self.verbose)
        result = analyzer.analyze_url(
            dblp_url=url,
            keywords=args.get("keywords"),
            exclude_keywords=args.get("exclude_keywords"),
            min_year=args.get("min_year"),
            max_year=args.get("max_year")
        )
        
        # 导出结果
        output_format = args.get("output_format", "markdown")
        output_path = f"./output/dblp_analysis.{output_format}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == "markdown":
            content = analyzer.export_markdown(result, output_path)
        elif output_format == "json":
            content = analyzer.export_json(result, output_path)
        else:
            content = result
        
        return {
            "success": True,
            "total_papers": result["total_papers"],
            "distribution": result["distribution"],
            "output_path": output_path,
            "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
        }
    
    async def fetch_abstracts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """摘要获取工具"""
        dois = args.get("dois", [])
        arxiv_ids = args.get("arxiv_ids", [])
        
        if not dois and not arxiv_ids:
            raise ValueError("At least one DOI or arXiv ID is required")
        
        fetcher = AbstractFetcher(
            cache_dir=args.get("cache_dir", "./cache/abstracts"),
            max_workers=args.get("max_workers", 5),
            verbose=self.verbose
        )
        
        results = fetcher.batch_fetch(dois=dois, arxiv_ids=arxiv_ids)
        
        return {
            "success": True,
            "fetched_count": len(results),
            "abstracts": results
        }
    
    async def classify_papers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """论文分类工具"""
        papers = args.get("papers", [])
        if not papers:
            raise ValueError("papers list is required")
        
        classifier = PrecisionClassifier(
            taxonomy_file=args.get("taxonomy_file"),
            verbose=self.verbose
        )
        
        # 临时 CSV 用于处理
        import csv
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            if papers:
                fieldnames = list(papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(papers)
            temp_input = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            temp_output = f.name
        
        try:
            by_category = classifier.classify_csv(
                input_csv=temp_input,
                output_csv=temp_output,
                min_score=args.get("min_score", 1.0)
            )
            
            # 读取结果
            classified_papers = []
            with open(temp_output, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    classified_papers.append(row)
            
            # 导出到目录
            if args.get("output_dir"):
                classifier.export_by_category(
                    by_category=by_category,
                    output_dir=args.get("output_dir"),
                    format='markdown'
                )
            
            return {
                "success": True,
                "classified_count": len(classified_papers),
                "categories": list(by_category.keys()),
                "distribution": {cat: len(papers) for cat, papers in by_category.items()},
                "papers": classified_papers
            }
        finally:
            Path(temp_input).unlink(missing_ok=True)
            Path(temp_output).unlink(missing_ok=True)
    
    async def extract_paper_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """论文信息提取工具"""
        papers = args.get("papers", [])
        if not papers:
            raise ValueError("papers list is required")
        
        extractor = PaperExtractor(verbose=self.verbose)
        
        extracted = []
        for paper in papers:
            result = extractor.extract_paper(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", "")
            )
            extracted.append({**paper, **result})
        
        # 生成汇总
        group_by = args.get("group_by", "method")
        import tempfile
        from pathlib import Path
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            summary_path = f.name
        
        extractor.export_summary(
            extracted_papers=extracted,
            output_file=summary_path,
            group_by=group_by
        )
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        
        Path(summary_path).unlink(missing_ok=True)
        
        return {
            "success": True,
            "extracted_count": len(extracted),
            "papers": extracted,
            "summary": summary_content
        }
    
    async def generate_analysis_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """分析报告生成工具"""
        papers = args.get("papers", [])
        if not papers:
            raise ValueError("papers list is required")
        
        generator = DeepAnalysisReport(verbose=self.verbose)
        
        import tempfile
        from pathlib import Path
        
        # 写入临时 CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            if papers:
                fieldnames = list(papers[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(papers)
            temp_input = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_output = f.name
        
        try:
            import csv
            content = generator.generate_report(
                input_csv=temp_input,
                output_md=temp_output,
                top_n=args.get("top_n", 10),
                include_timeline=args.get("include_timeline", True),
                include_comparison=args.get("include_comparison", True)
            )
            
            with open(temp_output, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            return {
                "success": True,
                "paper_count": len(papers),
                "report": report_content
            }
        finally:
            Path(temp_input).unlink(missing_ok=True)
            Path(temp_output).unlink(missing_ok=True)
    
    async def run_full_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """完整工作流工具"""
        dblp_url = args.get("dblp_url")
        if not dblp_url:
            raise ValueError("dblp_url is required")
        
        output_dir = args.get("output_dir", "./analysis_output")
        
        orchestrator = WorkflowOrchestrator(
            output_dir=output_dir,
            verbose=self.verbose
        )
        
        zotero_config = None
        if args.get("zotero_sync"):
            zotero_config = args.get("zotero_config")
        
        result = orchestrator.run(
            dblp_url=dblp_url,
            keywords=args.get("keywords"),
            exclude_keywords=args.get("exclude_keywords"),
            zotero_sync=args.get("zotero_sync", False),
            zotero_config=zotero_config
        )
        
        return {
            "success": result.get("success", False),
            "duration": result.get("duration", 0),
            "files": result.get("files", {}),
            "errors": result.get("errors", [])
        }
    
    def _error_response(self, request_id: Any, message: str) -> Dict[str, Any]:
        """错误响应"""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": message
            }
        }


async def main():
    """主入口"""
    server = MCPServer(verbose=True)
    
    # 读取 stdin 输入
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = await server.handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
