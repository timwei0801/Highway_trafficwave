"""
Retrieval 評估指標
計算 RAG 系統檢索階段的品質指標
"""

import numpy as np
from typing import List, Dict, Any, Set
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """單次檢索結果"""
    query: str
    retrieved_chunks: List[str]  # 檢索到的 chunk IDs
    retrieved_scores: List[float]  # 對應的相似度分數
    relevant_chunks: Set[str]  # 標準答案中的相關 chunk IDs


class RetrievalMetrics:
    """Retrieval 評估指標計算器"""

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Recall@K: Top-K 結果中包含相關文件的比例

        Args:
            retrieved: 檢索到的 chunk IDs（按相似度排序）
            relevant: 標準答案中的相關 chunk IDs
            k: 取前 K 個結果

        Returns:
            Recall@K 分數 (0-1)
        """
        if not relevant:
            return 0.0

        top_k = set(retrieved[:k])
        hits = len(top_k & relevant)
        return hits / len(relevant)

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Precision@K: Top-K 結果中相關文件的比例

        Args:
            retrieved: 檢索到的 chunk IDs
            relevant: 標準答案中的相關 chunk IDs
            k: 取前 K 個結果

        Returns:
            Precision@K 分數 (0-1)
        """
        if k == 0:
            return 0.0

        top_k = set(retrieved[:k])
        hits = len(top_k & relevant)
        return hits / k

    @staticmethod
    def mrr(retrieved: List[str], relevant: Set[str]) -> float:
        """
        MRR (Mean Reciprocal Rank): 第一個相關文件的排名倒數

        Args:
            retrieved: 檢索到的 chunk IDs
            relevant: 標準答案中的相關 chunk IDs

        Returns:
            MRR 分數 (0-1)
        """
        for i, chunk_id in enumerate(retrieved, 1):
            if chunk_id in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        NDCG@K (Normalized Discounted Cumulative Gain)
        考慮排名的相關性分數

        Args:
            retrieved: 檢索到的 chunk IDs
            relevant: 標準答案中的相關 chunk IDs
            k: 取前 K 個結果

        Returns:
            NDCG@K 分數 (0-1)
        """
        if not relevant:
            return 0.0

        # 計算 DCG
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved[:k], 1):
            if chunk_id in relevant:
                dcg += 1.0 / np.log2(i + 1)

        # 計算 IDCG (理想情況下的 DCG)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def hit_rate(retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Hit Rate@K: 是否至少命中一個相關文件

        Args:
            retrieved: 檢索到的 chunk IDs
            relevant: 標準答案中的相關 chunk IDs
            k: 取前 K 個結果

        Returns:
            1.0 如果命中，0.0 如果沒命中
        """
        top_k = set(retrieved[:k])
        return 1.0 if (top_k & relevant) else 0.0


class RetrievalEvaluator:
    """Retrieval 評估器"""

    def __init__(self, k_values: List[int] = None):
        """
        Args:
            k_values: 要計算的 K 值列表，預設 [1, 3, 5, 10]
        """
        self.k_values = k_values or [1, 3, 5, 10]
        self.metrics = RetrievalMetrics()
        self.results: List[Dict[str, Any]] = []

    def evaluate_single(self, result: RetrievalResult) -> Dict[str, Any]:
        """
        評估單次檢索結果

        Args:
            result: RetrievalResult 物件

        Returns:
            包含各項指標的字典
        """
        scores = {
            "query": result.query,
            "num_retrieved": len(result.retrieved_chunks),
            "num_relevant": len(result.relevant_chunks),
        }

        # 計算各 K 值的指標
        for k in self.k_values:
            scores[f"recall@{k}"] = self.metrics.recall_at_k(
                result.retrieved_chunks, result.relevant_chunks, k
            )
            scores[f"precision@{k}"] = self.metrics.precision_at_k(
                result.retrieved_chunks, result.relevant_chunks, k
            )
            scores[f"ndcg@{k}"] = self.metrics.ndcg_at_k(
                result.retrieved_chunks, result.relevant_chunks, k
            )
            scores[f"hit@{k}"] = self.metrics.hit_rate(
                result.retrieved_chunks, result.relevant_chunks, k
            )

        # MRR 不需要 K
        scores["mrr"] = self.metrics.mrr(
            result.retrieved_chunks, result.relevant_chunks
        )

        self.results.append(scores)
        return scores

    def evaluate_batch(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        批次評估多個檢索結果

        Args:
            results: RetrievalResult 列表

        Returns:
            包含平均指標的字典
        """
        self.results = []

        for result in results:
            self.evaluate_single(result)

        return self.aggregate_results()

    def aggregate_results(self) -> Dict[str, Any]:
        """
        彙總所有評估結果

        Returns:
            包含平均指標和詳細結果的字典
        """
        if not self.results:
            return {"error": "No results to aggregate"}

        # 計算各指標的平均值
        metric_keys = [k for k in self.results[0].keys()
                       if k not in ["query", "num_retrieved", "num_relevant"]]

        aggregated = {
            "total_queries": len(self.results),
            "metrics": {}
        }

        for key in metric_keys:
            values = [r[key] for r in self.results]
            aggregated["metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        # 加入詳細結果
        aggregated["detailed_results"] = self.results

        return aggregated

    def print_report(self, aggregated: Dict[str, Any] = None):
        """印出評估報告"""
        if aggregated is None:
            aggregated = self.aggregate_results()

        print("\n" + "=" * 60)
        print("Retrieval 評估報告")
        print("=" * 60)
        print(f"總查詢數: {aggregated['total_queries']}")
        print()

        # 按 K 值分組顯示
        for k in self.k_values:
            print(f"--- K = {k} ---")
            for metric in ["recall", "precision", "ndcg", "hit"]:
                key = f"{metric}@{k}"
                if key in aggregated["metrics"]:
                    data = aggregated["metrics"][key]
                    print(f"  {metric.upper():12} : {data['mean']:.4f} (± {data['std']:.4f})")
            print()

        # MRR
        if "mrr" in aggregated["metrics"]:
            data = aggregated["metrics"]["mrr"]
            print(f"MRR: {data['mean']:.4f} (± {data['std']:.4f})")

        print("=" * 60)


# 簡化版評估器（當沒有 ground truth chunk IDs 時使用）
class SimpleRetrievalEvaluator:
    """
    簡化版 Retrieval 評估器
    當沒有標註的相關 chunk IDs 時，使用啟發式方法評估
    """

    def __init__(self):
        self.results = []

    def evaluate_by_keywords(
        self,
        query: str,
        retrieved_texts: List[str],
        expected_keywords: List[str],
        scores: List[float]
    ) -> Dict[str, Any]:
        """
        基於關鍵詞匹配評估檢索品質

        Args:
            query: 查詢問題
            retrieved_texts: 檢索到的文本內容
            expected_keywords: 期望出現的關鍵詞
            scores: 相似度分數

        Returns:
            評估結果
        """
        result = {
            "query": query,
            "num_retrieved": len(retrieved_texts),
            "avg_score": float(np.mean(scores)) if scores else 0.0,
            "max_score": float(max(scores)) if scores else 0.0,
            "min_score": float(min(scores)) if scores else 0.0,
        }

        # 計算每個檢索結果的關鍵詞覆蓋率
        keyword_coverages = []
        for text in retrieved_texts:
            text_lower = text.lower()
            hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
            coverage = hits / len(expected_keywords) if expected_keywords else 0
            keyword_coverages.append(coverage)

        result["keyword_coverage"] = {
            "mean": float(np.mean(keyword_coverages)) if keyword_coverages else 0.0,
            "max": float(max(keyword_coverages)) if keyword_coverages else 0.0,
            "any_hit": any(c > 0 for c in keyword_coverages),
        }

        # 計算最佳結果是否在 Top-K
        if keyword_coverages:
            best_idx = np.argmax(keyword_coverages)
            result["best_result_rank"] = int(best_idx) + 1
            result["best_result_coverage"] = float(keyword_coverages[best_idx])

        self.results.append(result)
        return result

    def aggregate_results(self) -> Dict[str, Any]:
        """彙總結果"""
        if not self.results:
            return {"error": "No results"}

        return {
            "total_queries": len(self.results),
            "avg_retrieval_score": float(np.mean([r["avg_score"] for r in self.results])),
            "avg_keyword_coverage": float(np.mean([r["keyword_coverage"]["mean"] for r in self.results])),
            "hit_rate": sum(1 for r in self.results if r["keyword_coverage"]["any_hit"]) / len(self.results),
            "detailed_results": self.results,
        }
