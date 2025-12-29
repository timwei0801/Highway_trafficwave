"""
RAG 系統評估模組
"""

from .evaluator import RAGEvaluator, run_evaluation
from .ab_comparator import ABComparator, ExperimentTracker
from .metrics.retrieval_metrics import (
    RetrievalResult,
    RetrievalMetrics,
    RetrievalEvaluator,
    SimpleRetrievalEvaluator
)
from .metrics.generation_metrics import (
    GenerationResult,
    GenerationMetrics,
    GenerationEvaluator
)

__all__ = [
    # 主評估器
    "RAGEvaluator",
    "run_evaluation",

    # A/B 比較
    "ABComparator",
    "ExperimentTracker",

    # Retrieval 指標
    "RetrievalResult",
    "RetrievalMetrics",
    "RetrievalEvaluator",
    "SimpleRetrievalEvaluator",

    # Generation 指標
    "GenerationResult",
    "GenerationMetrics",
    "GenerationEvaluator",
]
