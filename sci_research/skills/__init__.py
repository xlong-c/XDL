"""
Scientific Research Skills Package
科研工具技能包 - 零 token 消耗的论文分析工具集
"""

from .dblp_analyzer import DBLPAnalyzer
from .abstract_fetcher import AbstractFetcher
from .precision_classifier import PrecisionClassifier
from .paper_extractor import PaperExtractor
from .deep_analysis_report import DeepAnalysisReport
from .zotero_dual_sync import ZoteroDualSync
from .workflow_orchestrator import WorkflowOrchestrator

__version__ = '1.0.0'
__all__ = [
    'DBLPAnalyzer',
    'AbstractFetcher',
    'PrecisionClassifier',
    'PaperExtractor',
    'DeepAnalysisReport',
    'ZoteroDualSync',
    'WorkflowOrchestrator',
]
