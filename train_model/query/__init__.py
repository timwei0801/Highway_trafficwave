"""
混合查詢模組
提供意圖分類、精確查詢和混合路由功能
"""

from train_model.query.intent_classifier import (
    IntentClassifier,
    ParsedQuery,
    QueryIntent,
)

from train_model.query.precise_query_handler import (
    PreciseQueryHandler,
    QueryResult,
)

from train_model.query.hybrid_query_router import (
    HybridQueryRouter,
    HybridQueryResult,
    get_hybrid_router,
)

__all__ = [
    # 意圖分類
    'IntentClassifier',
    'ParsedQuery',
    'QueryIntent',
    # 精確查詢
    'PreciseQueryHandler',
    'QueryResult',
    # 混合路由
    'HybridQueryRouter',
    'HybridQueryResult',
    'get_hybrid_router',
]
