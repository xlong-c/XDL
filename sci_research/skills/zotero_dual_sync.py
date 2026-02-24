"""
Zotero 双目录同步管理器

实现论文的双目录管理：
- By Venue: 按期刊会议分类 (Journal/TIP/2024/Vol.33)
- By Topic: 按研究领域分类 (Segmentation/Semantic/Efficient ML)
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class ZoteroDualSync:
    """Zotero 双目录同步管理器"""
    
    def __init__(
        self,
        library_id: str,
        api_key: str,
        library_type: str = 'user',
        verbose: bool = True
    ):
        """
        初始化 Zotero 同步器
        
        Args:
            library_id: Zotero 图书馆 ID
            api_key: Zotero API 密钥
            library_type: 图书馆类型 (user/group)
            verbose: 是否打印详细日志
        """
        self.library_id = library_id
        self.api_key = api_key
        self.library_type = library_type
        self.verbose = verbose
        
        try:
            from pyzotero import zotero
            self.zot = zotero.Zotero(library_id, library_type, api_key)
            self.enabled = True
        except ImportError:
            if verbose:
                print("⚠️  未安装 pyzotero，请先安装：pip install pyzotero")
            self.enabled = False
            self.zot = None
        
        # 缓存目录信息
        self.collection_cache: Dict[str, str] = {}  # path -> key
    
    def initialize_structure(self, venue_roots: Optional[List[str]] = None, topic_roots: Optional[List[str]] = None):
        """
        初始化双目录结构
        
        Args:
            venue_roots: 按 venue 分类的根目录列表
            topic_roots: 按 topic 分类的根目录列表
        """
        if not self.enabled:
            return
        
        if venue_roots is None:
            venue_roots = ["Journal", "Conference"]
        
        if topic_roots is None:
            topic_roots = ["Backbone", "Segmentation", "Efficient ML", "Application"]
        
        if self.verbose:
            print("📁 初始化双目录结构...")
        
        # 创建 Venue 根目录
        for root in venue_roots:
            self._ensure_collection(root)
        
        # 创建 Topic 根目录
        for root in topic_roots:
            self._ensure_collection(root)
        
        if self.verbose:
            print("✅ 双目录结构初始化完成")
    
    def batch_import(
        self,
        papers: List[Dict[str, Any]],
        venue_path: str,
        topic_paths: List[str],
        auto_tags: bool = True
    ) -> Dict[str, Any]:
        """
        批量导入论文到双目录
        
        Args:
            papers: 论文列表 (含 title, doi, abstract 等)
            venue_path: Venue 目录路径 (如 "Journal/TIP/2024/Vol.33")
            topic_paths: Topic 目录路径列表 (如 ["Segmentation/Semantic", "Efficient ML"])
            auto_tags: 是否自动生成标签
        
        Returns:
            导入结果统计
        """
        if not self.enabled:
            return {"success": 0, "failed": len(papers)}
        
        if self.verbose:
            print(f"📥 开始导入 {len(papers)} 篇论文...")
        
        # 1. 确保目录存在
        if self.verbose:
            print(f"  → 确保目录存在：{venue_path}")
            for topic_path in topic_paths:
                print(f"  → 确保目录存在：{topic_path}")
        
        venue_collection = self._ensure_collection(venue_path)
        topic_collections = [
            self._ensure_collection(path)
            for path in topic_paths
        ]
        
        # 2. 批量创建项目
        items = []
        for paper in papers:
            item = self._create_zotero_item(
                paper=paper,
                collections=[venue_collection['key']] + [c['key'] for c in topic_collections],
                auto_tags=auto_tags
            )
            items.append(item)
        
        # 3. 批量上传
        if self.verbose:
            print(f"  → 上传 {len(items)} 个项目到 Zotero...")
        
        try:
            response = self.zot.create_items(items)
            
            success_count = len(response.get('success', []))
            failed_count = len(response.get('failed', []))
            
            if self.verbose:
                print(f"\n✅ 导入完成！成功：{success_count}, 失败：{failed_count}")
            
            return {
                "success": success_count,
                "failed": failed_count,
                "response": response
            }
        except Exception as e:
            if self.verbose:
                print(f"❌ 导入失败：{e}")
            return {
                "success": 0,
                "failed": len(papers),
                "error": str(e)
            }
    
    def _create_zotero_item(
        self,
        paper: Dict[str, Any],
        collections: List[str],
        auto_tags: bool = True
    ) -> Dict[str, Any]:
        """
        创建 Zotero 项目
        
        Args:
            paper: 论文信息
            collections: 所属目录 key 列表
            auto_tags: 是否自动生成标签
        
        Returns:
            Zotero 项目字典
        """
        # 确定项目类型
        doi = paper.get('doi', '')
        arxiv_id = paper.get('arxiv_id', '')
        
        if doi:
            item_type = 'journalArticle'
        elif arxiv_id:
            item_type = 'preprint'
        else:
            item_type = 'journalArticle'
        
        # 构建项目
        item = {
            'itemType': item_type,
            'title': paper.get('title', ''),
            'creators': self._parse_creators(paper.get('authors', '')),
            'collections': collections,
            'tags': []
        }
        
        # 添加 DOI
        if doi:
            item['DOI'] = doi
        
        # 添加摘要
        abstract = paper.get('abstract', '')
        if abstract:
            item['abstractNote'] = abstract
        
        # 添加年份
        year = paper.get('year', '')
        if year:
            try:
                item['date'] = str(year)
            except:
                pass
        
        # 添加期刊/会议信息
        venue = paper.get('venue', '')
        if venue:
            if 'journal' in venue.lower():
                item['publicationTitle'] = venue
            else:
                item['proceedingsTitle'] = venue
        
        # 自动生成标签
        if auto_tags:
            tags = self._generate_tags(paper)
            item['tags'] = [{'tag': tag} for tag in tags]
        
        return item
    
    def _parse_creators(self, authors_str: str) -> List[Dict[str, str]]:
        """
        解析作者字符串为 Zotero 格式
        
        Args:
            authors_str: 作者字符串 (如 "John Doe, Jane Smith")
        
        Returns:
            创作者列表
        """
        creators = []
        
        if not authors_str:
            return creators
        
        # 分割作者
        authors = authors_str.split(',')
        
        for author in authors:
            author = author.strip()
            if not author:
                continue
            
            # 尝试分割姓和名
            parts = author.split()
            if len(parts) >= 2:
                firstName = ' '.join(parts[:-1])
                lastName = parts[-1]
            else:
                firstName = author
                lastName = ''
            
            creators.append({
                'creatorType': 'author',
                'firstName': firstName,
                'lastName': lastName
            })
        
        return creators[:10]  # 限制最多 10 个作者
    
    def _generate_tags(self, paper: Dict[str, Any]) -> List[str]:
        """
        自动生成标签
        
        Args:
            paper: 论文信息
        
        Returns:
            标签列表
        """
        tags = []
        
        # Venue 标签
        venue = paper.get('venue', '')
        year = paper.get('year', '')
        if venue and year:
            tags.append(f"{venue}_{year}")
        
        # 类别标签
        category = paper.get('primary_category', '')
        if category:
            tags.append(category.replace(' ', '_'))
        
        # 方法标签
        methods = paper.get('methods', '')
        if methods:
            for method in methods.split('|')[:3]:
                tags.append(method.replace(' ', '_'))
        
        # 数据集标签
        datasets = paper.get('datasets', '')
        if datasets:
            for dataset in datasets.split('|')[:3]:
                tags.append(dataset.replace(' ', '_'))
        
        # 去重
        return list(set(tags))
    
    def _ensure_collection(self, path: str) -> Dict[str, Any]:
        """
        递归确保目录路径存在
        
        Args:
            path: 目录路径 (如 "Journal/TIP/2024/Vol.33")
        
        Returns:
            目录信息
        """
        # 检查缓存
        if path in self.collection_cache:
            return {'key': self.collection_cache[path], 'name': path}
        
        parts = path.split('/')
        parent_key = None
        
        for i, part in enumerate(parts):
            # 构建当前路径
            current_path = '/'.join(parts[:i+1])
            
            # 检查缓存
            if current_path in self.collection_cache:
                parent_key = self.collection_cache[current_path]
                continue
            
            # 查找或创建目录
            collection = self._find_or_create_collection(
                name=part,
                parent_key=parent_key
            )
            
            # 更新缓存
            self.collection_cache[current_path] = collection['key']
            parent_key = collection['key']
        
        return {'key': parent_key, 'name': path}
    
    def _find_or_create_collection(
        self,
        name: str,
        parent_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        查找或创建目录
        
        Args:
            name: 目录名
            parent_key: 父目录 key
        
        Returns:
            目录信息
        """
        try:
            # 尝试查找
            collections = self.zot.collections()
            
            for coll in collections:
                coll_name = coll['data']['name']
                coll_parent = coll['data'].get('parentCollection', None)
                
                if coll_name == name and coll_parent == parent_key:
                    return {'key': coll['key'], 'name': coll_name}
            
            # 创建目录
            new_coll = {
                'name': name,
                'parentCollection': parent_key
            }
            
            response = self.zot.create_collection(new_coll)
            
            if self.verbose:
                action = "创建" if parent_key else "创建根目录"
                print(f"    ✓ {action}: {name}")
            
            return {
                'key': response['key'],
                'name': name
            }
        except Exception as e:
            if self.verbose:
                print(f"    ⚠️  目录操作失败：{e}")
            
            # 返回一个虚拟目录（用于测试模式）
            return {
                'key': f'temp_{name}',
                'name': name
            }


# 命令行入口
def main():
    import argparse
    import csv
    
    parser = argparse.ArgumentParser(
        description="Zotero 双目录同步管理"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 CSV (含分类和提取的信息)"
    )
    parser.add_argument(
        "--library-id",
        type=str,
        required=True,
        help="Zotero 图书馆 ID"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Zotero API 密钥"
    )
    parser.add_argument(
        "--venue-path",
        type=str,
        required=True,
        help="Venue 目录路径 (如 Journal/TIP/2024/Vol.33)"
    )
    parser.add_argument(
        "--topic-paths",
        type=str,
        nargs='+',
        required=True,
        help="Topic 目录路径列表 (如 Segmentation/Semantic Efficient ML)"
    )
    parser.add_argument(
        "--library-type",
        type=str,
        choices=['user', 'group'],
        default='user',
        help="图书馆类型"
    )
    parser.add_argument(
        "--no-auto-tags",
        action="store_true",
        help="禁用自动生成标签"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="禁用详细日志"
    )
    
    args = parser.parse_args()
    
    # 读取 CSV
    papers = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)
    
    # 创建同步器
    sync = ZoteroDualSync(
        library_id=args.library_id,
        api_key=args.api_key,
        library_type=args.library_type,
        verbose=not args.quiet
    )
    
    # 导入
    result = sync.batch_import(
        papers=papers,
        venue_path=args.venue_path,
        topic_paths=args.topic_paths,
        auto_tags=not args.no_auto_tags
    )
    
    # 输出结果
    print(f"\n📊 导入统计:")
    print(f"  成功：{result['success']}")
    print(f"  失败：{result['failed']}")


if __name__ == "__main__":
    main()
