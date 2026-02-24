"""
摘要获取器 - 批量获取论文摘要

通过 Crossref API 和 arXiv API 批量获取论文摘要，
支持缓存、并发请求、失败重试。
"""

import requests
import json
import base64
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib


class AbstractFetcher:
    """论文摘要获取器"""
    
    # API 端点
    CROSSREF_API = "https://api.crossref.org/works/"
    ARXIV_API = "http://export.arxiv.org/api/query"
    OPENALEX_API = "https://api.openalex.org/works/"
    
    # 速率限制 (请求/秒)
    RATE_LIMITS = {
        "crossref": None,  # 无限制
        "arxiv": 0.33,     # 3 秒一次
        "openalex": 1.0    # 1 秒一次
    }
    
    def __init__(
        self,
        cache_dir: str = "./cache/abstracts",
        max_workers: int = 5,
        timeout: int = 10,
        verbose: bool = True
    ):
        """
        初始化摘要获取器
        
        Args:
            cache_dir: 缓存目录
            max_workers: 并发请求数
            timeout: 请求超时时间 (秒)
            verbose: 是否打印详细日志
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "abstract_cache.json"
        
        self.max_workers = max_workers
        self.timeout = timeout
        self.verbose = verbose
        
        # 加载缓存
        self.cache = self._load_cache()
        
        # 速率限制器
        self.last_request_time: Dict[str, float] = {}
    
    def batch_fetch(
        self,
        dois: List[str],
        arxiv_ids: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量获取摘要
        
        Args:
            dois: DOI 列表
            arxiv_ids: arXiv ID 列表 (可选)
            max_workers: 覆盖默认并发数
        
        Returns:
            摘要字典 {doi: {title, abstract, source, ...}}
        """
        if max_workers is None:
            max_workers = self.max_workers
        
        results = {}
        
        # 1. 处理 DOI
        if dois:
            if self.verbose:
                print(f"📥 开始获取 {len(dois)} 个 DOI 的摘要...")
            
            # 过滤缓存中已有的
            dois_to_fetch = [d for d in dois if d not in self.cache]
            cached_count = len(dois) - len(dois_to_fetch)
            
            if self.verbose:
                print(f"  ✓ 缓存命中 {cached_count} 个")
                print(f"  → 需要获取 {len(dois_to_fetch)} 个")
            
            # 并发获取
            if dois_to_fetch:
                doi_results = self._batch_fetch_dois(dois_to_fetch, max_workers)
                results.update(doi_results)
            
            # 合并缓存
            for doi in dois:
                if doi in self.cache:
                    results[doi] = self.cache[doi]
        
        # 2. 处理 arXiv ID
        if arxiv_ids:
            if self.verbose:
                print(f"\n📥 开始获取 {len(arxiv_ids)} 个 arXiv 论文的摘要...")
            
            arxiv_to_fetch = [a for a in arxiv_ids if a not in self.cache]
            cached_count = len(arxiv_ids) - len(arxiv_to_fetch)
            
            if self.verbose:
                print(f"  ✓ 缓存命中 {cached_count} 个")
                print(f"  → 需要获取 {len(arxiv_to_fetch)} 个")
            
            if arxiv_to_fetch:
                arxiv_results = self._batch_fetch_arxiv(arxiv_to_fetch, max_workers)
                results.update(arxiv_results)
            
            for arxiv_id in arxiv_ids:
                if arxiv_id in self.cache:
                    results[arxiv_id] = self.cache[arxiv_id]
        
        # 3. 保存缓存
        self._save_cache()
        
        if self.verbose:
            print(f"\n✅ 完成！成功获取 {len(results)} 个摘要")
        
        return results
    
    def _batch_fetch_dois(
        self,
        dois: List[str],
        max_workers: int
    ) -> Dict[str, Dict[str, Any]]:
        """批量通过 Crossref 获取摘要"""
        results = {}
        
        # 分批请求 (每批 100 个 DOI)
        batch_size = 100
        batches = [dois[i:i+batch_size] for i in range(0, len(dois), batch_size)]
        
        for i, batch in enumerate(batches):
            if self.verbose:
                print(f"  批次 {i+1}/{len(batches)}: {len(batch)} 个 DOI")
            
            # Crossref 支持 | 分隔的多 DOI 查询
            doi_query = "|".join(batch)
            url = self.CROSSREF_API + doi_query
            params = {
                "select": "DOI,title,abstract,published-print,published-online",
                "rows": len(batch)
            }
            
            try:
                self._rate_limit("crossref")
                response = requests.get(url, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("message", {}).get("items", [])
                    
                    for item in items:
                        doi = item.get("DOI")
                        if doi:
                            result = self._parse_crossref_item(item)
                            results[doi] = result
                            self.cache[doi] = result
                else:
                    if self.verbose:
                        print(f"    ⚠️  Crossref 请求失败：{response.status_code}")
                        # 尝试单个获取
                        for doi in batch:
                            result = self._fetch_single_doi(doi)
                            if result:
                                results[doi] = result
                                self.cache[doi] = result
            except Exception as e:
                if self.verbose:
                    print(f"    ❌ 错误：{e}")
                # 失败后尝试单个获取
                for doi in batch:
                    result = self._fetch_single_doi(doi)
                    if result:
                        results[doi] = result
                        self.cache[doi] = result
            
            # 批次间短暂延迟
            time.sleep(0.5)
        
        return results
    
    def _fetch_single_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """获取单个 DOI 的摘要"""
        try:
            self._rate_limit("crossref")
            url = self.CROSSREF_API + doi
            params = {"select": "DOI,title,abstract"}
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                item = response.json().get("message", {})
                return self._parse_crossref_item(item)
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  DOI {doi} 获取失败：{e}")
        
        return None
    
    def _parse_crossref_item(self, item: Dict) -> Dict[str, Any]:
        """解析 Crossref 返回的数据"""
        doi = item.get("DOI", "")
        title = item.get("title", [""])[0]
        abstract = item.get("abstract", "")
        
        # Crossref 返回 base64 编码的摘要
        if abstract:
            try:
                abstract = base64.b64decode(abstract).decode("utf-8")
            except:
                abstract = ""
        
        # 提取出版年份
        year = None
        if "published-print" in item:
            year = item["published-print"].get("date-parts", [[None]])[0][0]
        elif "published-online" in item:
            year = item["published-online"].get("date-parts", [[None]])[0][0]
        
        return {
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "year": year,
            "source": "crossref",
            "fetched_at": datetime.now().isoformat()
        }
    
    def _batch_fetch_arxiv(
        self,
        arxiv_ids: List[str],
        max_workers: int
    ) -> Dict[str, Dict[str, Any]]:
        """批量通过 arXiv API 获取摘要"""
        results = {}
        
        # arXiv 支持多 ID 查询
        # 格式：arxiv_id OR arxiv_id OR ...
        id_query = " OR ".join([f"all:{arxiv_id}" for arxiv_id in arxiv_ids])
        
        url = self.ARXIV_API
        params = {
            "search_query": id_query,
            "max_results": len(arxiv_ids),
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            self._rate_limit("arxiv")
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                # 解析 Atom XML
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.text)
                
                # 命名空间
                ns = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom"
                }
                
                for entry in root.findall("atom:entry", ns):
                    arxiv_id_elem = entry.find("arxiv:comment", ns)
                    arxiv_id = None
                    
                    # 从 URL 提取 arXiv ID
                    id_elem = entry.find("atom:id", ns)
                    if id_elem is not None and id_elem.text:
                        arxiv_id = id_elem.text.split("/abs/")[-1]
                    
                    if arxiv_id:
                        title_elem = entry.find("atom:title", ns)
                        summary_elem = entry.find("atom:summary", ns)
                        published_elem = entry.find("atom:published", ns)
                        
                        title = title_elem.text.strip() if title_elem is not None else ""
                        abstract = summary_elem.text.strip() if summary_elem is not None else ""
                        published = published_elem.text if published_elem is not None else ""
                        
                        # 提取年份
                        year = int(published.split("-")[0]) if published else None
                        
                        result = {
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "abstract": abstract,
                            "year": year,
                            "source": "arxiv",
                            "fetched_at": datetime.now().isoformat()
                        }
                        
                        results[arxiv_id] = result
                        self.cache[arxiv_id] = result
        except Exception as e:
            if self.verbose:
                print(f"    ❌ arXiv 批量请求失败：{e}")
                print(f"    → 尝试单个获取...")
            
            # 失败后尝试单个获取
            for arxiv_id in arxiv_ids:
                result = self._fetch_single_arxiv(arxiv_id)
                if result:
                    results[arxiv_id] = result
                    self.cache[arxiv_id] = result
        
        return results
    
    def _fetch_single_arxiv(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """获取单个 arXiv 论文的摘要"""
        try:
            self._rate_limit("arxiv")
            url = self.ARXIV_API
            params = {
                "search_query": f"all:{arxiv_id}",
                "max_results": 1
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.text)
                ns = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom"
                }
                
                entry = root.find("atom:entry", ns)
                if entry is not None:
                    # 复用批量解析逻辑
                    batch_result = self._batch_fetch_arxiv([arxiv_id], 1)
                    return batch_result.get(arxiv_id)
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  arXiv {arxiv_id} 获取失败：{e}")
        
        return None
    
    def _rate_limit(self, source: str):
        """速率限制"""
        if source in self.RATE_LIMITS and self.RATE_LIMITS[source]:
            rate = self.RATE_LIMITS[source]
            min_interval = 1.0 / rate
            
            if source in self.last_request_time:
                elapsed = time.time() - self.last_request_time[source]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
            
            self.last_request_time[source] = time.time()
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"⚠️  缓存保存失败：{e}")
    
    def save_to_csv(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: str
    ):
        """
        保存结果到 CSV
        
        Args:
            results: fetch 返回的结果字典
            output_path: 输出 CSV 路径
        """
        import csv
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                "ID", "Title", "Abstract", "Year", "Source", "Fetched At"
            ])
            
            # 写入数据
            for id_, data in results.items():
                writer.writerow([
                    id_,
                    data.get("title", ""),
                    data.get("abstract", ""),
                    data.get("year", ""),
                    data.get("source", ""),
                    data.get("fetched_at", "")
                ])
        
        if self.verbose:
            print(f"✓ CSV 已保存至：{output_path}")
    
    def merge_with_csv(
        self,
        input_csv: str,
        results: Dict[str, Dict[str, Any]],
        output_csv: str,
        id_column: str = "doi"
    ):
        """
        将摘要合并到现有 CSV
        
        Args:
            input_csv: 输入 CSV (包含 DOI 或 arXiv ID)
            results: fetch 返回的结果
            output_csv: 输出 CSV
            id_column: ID 列名 (doi 或 arxiv_id)
        """
        import csv
        
        with open(input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames + ["abstract", "abstract_year"]
            
            rows = list(reader)
        
        # 添加摘要
        for row in rows:
            id_ = row.get(id_column, "")
            if id_ and id_ in results:
                row["abstract"] = results[id_].get("abstract", "")
                row["abstract_year"] = results[id_].get("year", "")
            else:
                row["abstract"] = ""
                row["abstract_year"] = ""
        
        # 写入输出
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        if self.verbose:
            print(f"✓ 合并后的 CSV 已保存至：{output_csv}")


# 命令行入口
def main():
    import argparse
    import csv
    
    parser = argparse.ArgumentParser(
        description="批量获取论文摘要 (Crossref/arXiv)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 CSV 文件 (包含 doi 或 arxiv_id 列)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="papers_with_abstracts.csv",
        help="输出 CSV 文件路径"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/abstracts",
        help="缓存目录"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="最大并发数"
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="doi",
        choices=["doi", "arxiv_id"],
        help="ID 列名"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="禁用详细日志"
    )
    
    args = parser.parse_args()
    
    # 读取输入 CSV
    dois = []
    arxiv_ids = []
    
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.id_column == "doi" and row.get("doi"):
                dois.append(row["doi"])
            elif args.id_column == "arxiv_id" and row.get("arxiv_id"):
                arxiv_ids.append(row["arxiv_id"])
    
    if not dois and not arxiv_ids:
        print(f"❌ 未在 {args.input} 中找到 {args.id_column} 列")
        return
    
    # 获取摘要
    fetcher = AbstractFetcher(
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        verbose=not args.quiet
    )
    
    results = fetcher.batch_fetch(dois=dois, arxiv_ids=arxiv_ids)
    
    # 合并到 CSV
    fetcher.merge_with_csv(
        input_csv=args.input,
        results=results,
        output_csv=args.output,
        id_column=args.id_column
    )
    
    # 打印统计
    success_count = sum(1 for r in results.values() if r.get("abstract"))
    print(f"\n📊 统计:")
    print(f"  成功获取：{success_count}/{len(results)}")
    print(f"  有摘要：{success_count}")
    print(f"  无摘要：{len(results) - success_count}")


if __name__ == "__main__":
    main()
