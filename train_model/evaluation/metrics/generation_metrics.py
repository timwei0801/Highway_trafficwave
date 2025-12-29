"""
Generation 評估指標
計算 RAG 系統生成階段的品質指標
"""

import numpy as np
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """單次生成結果"""
    question: str
    generated_answer: str
    ground_truth_answer: str
    retrieved_context: str
    evaluation_criteria: Dict[str, Any]


class GenerationMetrics:
    """Generation 評估指標計算器"""

    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: 用於計算語意相似度的嵌入模型
                           如果為 None，將跳過語意相似度計算
        """
        self.embedding_model = embedding_model

    def exact_match(self, generated: str, ground_truth: str) -> float:
        """
        精確匹配：生成答案是否包含標準答案的關鍵內容

        Returns:
            1.0 如果完全匹配，0.0 如果不匹配
        """
        # 正規化：移除空白和標點
        def normalize(text):
            return re.sub(r'\s+', '', text.lower())

        return 1.0 if normalize(ground_truth) in normalize(generated) else 0.0

    def keyword_precision(
        self,
        generated: str,
        must_contain: List[str],
        should_mention: List[str] = None
    ) -> Dict[str, float]:
        """
        關鍵詞精確度

        Args:
            generated: 生成的答案
            must_contain: 必須包含的關鍵詞（任一即可）
            should_mention: 應該提及的關鍵詞

        Returns:
            包含 must_score 和 should_score 的字典
        """
        generated_lower = generated.lower()

        # must_contain: 只要包含任一個就算通過
        must_score = 0.0
        if must_contain:
            must_score = 1.0 if any(
                kw.lower() in generated_lower for kw in must_contain
            ) else 0.0

        # should_mention: 計算覆蓋比例
        should_score = 0.0
        if should_mention:
            hits = sum(1 for kw in should_mention if kw.lower() in generated_lower)
            should_score = hits / len(should_mention)

        return {
            "must_contain_score": must_score,
            "should_mention_score": should_score,
            "combined_score": (must_score * 0.6 + should_score * 0.4)
        }

    def faithfulness(self, generated: str, context: str) -> Dict[str, float]:
        """
        Faithfulness（忠實度）：回答是否基於檢索的內容

        這個指標檢測 LLM 是否「幻覺」出不在 context 中的資訊

        Args:
            generated: 生成的答案
            context: 檢索到的上下文

        Returns:
            faithfulness 分數和相關指標
        """
        if not context:
            return {"faithfulness": 0.0, "context_overlap": 0.0}

        # 提取生成答案中的數字
        generated_numbers = set(re.findall(r'\d+\.?\d*', generated))
        context_numbers = set(re.findall(r'\d+\.?\d*', context))

        # 計算數字的重疊（數字通常是關鍵事實）
        if generated_numbers:
            number_overlap = len(generated_numbers & context_numbers) / len(generated_numbers)
        else:
            number_overlap = 1.0  # 沒有數字時假設沒問題

        # 提取關鍵詞（去除常見詞）
        stopwords = {'的', '是', '在', '有', '和', '了', '與', '為', '及', '或', '等', '之'}

        def extract_keywords(text):
            # 簡單分詞：按標點和空白切分
            words = re.findall(r'[\u4e00-\u9fff]+|\w+', text)
            return set(w for w in words if len(w) > 1 and w not in stopwords)

        gen_keywords = extract_keywords(generated)
        ctx_keywords = extract_keywords(context)

        # 計算關鍵詞重疊
        if gen_keywords:
            keyword_overlap = len(gen_keywords & ctx_keywords) / len(gen_keywords)
        else:
            keyword_overlap = 0.0

        # 綜合分數
        faithfulness = (number_overlap * 0.5 + keyword_overlap * 0.5)

        return {
            "faithfulness": faithfulness,
            "number_overlap": number_overlap,
            "keyword_overlap": keyword_overlap,
            "generated_numbers": list(generated_numbers),
            "context_numbers": list(context_numbers),
        }

    def semantic_similarity(self, generated: str, ground_truth: str) -> float:
        """
        語意相似度：使用嵌入模型計算

        Args:
            generated: 生成的答案
            ground_truth: 標準答案

        Returns:
            相似度分數 (0-1)
        """
        if self.embedding_model is None:
            return -1.0  # 表示未計算

        try:
            gen_embedding = self.embedding_model.encode([generated])[0]
            truth_embedding = self.embedding_model.encode([ground_truth])[0]

            # 餘弦相似度
            similarity = np.dot(gen_embedding, truth_embedding) / (
                np.linalg.norm(gen_embedding) * np.linalg.norm(truth_embedding)
            )
            return float(similarity)
        except Exception as e:
            print(f"語意相似度計算失敗: {e}")
            return -1.0

    def answer_quality(self, generated: str) -> Dict[str, float]:
        """
        回答品質評估（結構、長度、專業性）

        Args:
            generated: 生成的答案

        Returns:
            品質指標字典
        """
        # 長度評分
        length = len(generated)
        if 50 <= length <= 500:
            length_score = 1.0
        elif length < 50:
            length_score = length / 50
        else:
            length_score = max(0.5, 1 - (length - 500) / 1000)

        # 結構評分（是否有分段、條列）
        has_structure = bool(re.search(r'[•\-\d+\.\n]', generated))
        structure_score = 1.0 if has_structure else 0.7

        # 專業術語（高速公路領域）
        technical_terms = [
            "車道", "公尺", "坡度", "半徑", "曲率", "路肩", "鋪面",
            "輔助車道", "國道", "高速公路", "里程", "槽化", "避車彎"
        ]
        found_terms = sum(1 for term in technical_terms if term in generated)
        technical_score = min(1.0, found_terms / 4)

        # 負面指標（不確定性表達）
        uncertainty_phrases = ["不確定", "無法確認", "抱歉", "沒有資料", "無相關"]
        has_uncertainty = any(phrase in generated for phrase in uncertainty_phrases)
        confidence_score = 0.3 if has_uncertainty else 1.0

        return {
            "length_score": length_score,
            "structure_score": structure_score,
            "technical_score": technical_score,
            "confidence_score": confidence_score,
            "overall_quality": (
                length_score * 0.2 +
                structure_score * 0.2 +
                technical_score * 0.3 +
                confidence_score * 0.3
            )
        }

    def negative_check(
        self,
        generated: str,
        must_not_contain: List[str]
    ) -> Dict[str, Any]:
        """
        負面檢查：確保答案不包含不應該有的內容

        Args:
            generated: 生成的答案
            must_not_contain: 不應包含的關鍵詞

        Returns:
            檢查結果
        """
        generated_lower = generated.lower()
        violations = [
            kw for kw in must_not_contain
            if kw.lower() in generated_lower
        ]

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "score": 1.0 if not violations else 0.0
        }


class GenerationEvaluator:
    """Generation 評估器"""

    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: 可選的嵌入模型，用於語意相似度計算
        """
        self.metrics = GenerationMetrics(embedding_model)
        self.results: List[Dict[str, Any]] = []

    def evaluate_single(self, result: GenerationResult) -> Dict[str, Any]:
        """
        評估單次生成結果

        Args:
            result: GenerationResult 物件

        Returns:
            包含各項指標的字典
        """
        criteria = result.evaluation_criteria

        scores = {
            "question": result.question,
        }

        # 1. 關鍵詞精確度
        keyword_scores = self.metrics.keyword_precision(
            result.generated_answer,
            criteria.get("must_contain_any", []),
            criteria.get("should_mention", [])
        )
        scores.update({f"keyword_{k}": v for k, v in keyword_scores.items()})

        # 2. Faithfulness（忠實度）
        if result.retrieved_context:
            faithfulness = self.metrics.faithfulness(
                result.generated_answer,
                result.retrieved_context
            )
            scores.update({f"faith_{k}": v for k, v in faithfulness.items()
                          if not isinstance(v, list)})

        # 3. 語意相似度
        if result.ground_truth_answer:
            scores["semantic_similarity"] = self.metrics.semantic_similarity(
                result.generated_answer,
                result.ground_truth_answer
            )

        # 4. 回答品質
        quality = self.metrics.answer_quality(result.generated_answer)
        scores.update({f"quality_{k}": v for k, v in quality.items()})

        # 5. 負面檢查
        negative = self.metrics.negative_check(
            result.generated_answer,
            criteria.get("must_not_contain", [])
        )
        scores["negative_check_passed"] = negative["passed"]
        scores["negative_check_score"] = negative["score"]

        # 6. 綜合分數
        scores["overall_score"] = self._calculate_overall_score(scores)

        self.results.append(scores)
        return scores

    def _calculate_overall_score(self, scores: Dict[str, Any]) -> float:
        """計算綜合分數"""
        weights = {
            "keyword_combined_score": 0.25,
            "faith_faithfulness": 0.20,
            "quality_overall_quality": 0.20,
            "negative_check_score": 0.15,
            "semantic_similarity": 0.20,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for key, weight in weights.items():
            if key in scores and scores[key] >= 0:  # -1 表示未計算
                weighted_sum += scores[key] * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def evaluate_batch(self, results: List[GenerationResult]) -> Dict[str, Any]:
        """批次評估"""
        self.results = []
        for result in results:
            self.evaluate_single(result)
        return self.aggregate_results()

    def aggregate_results(self) -> Dict[str, Any]:
        """彙總結果"""
        if not self.results:
            return {"error": "No results to aggregate"}

        # 找出數值型欄位
        numeric_keys = [
            k for k in self.results[0].keys()
            if isinstance(self.results[0][k], (int, float)) and k != "question"
        ]

        aggregated = {
            "total_questions": len(self.results),
            "metrics": {}
        }

        for key in numeric_keys:
            values = [r[key] for r in self.results if r.get(key, -1) >= 0]
            if values:
                aggregated["metrics"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        aggregated["detailed_results"] = self.results
        return aggregated

    def print_report(self, aggregated: Dict[str, Any] = None):
        """印出評估報告"""
        if aggregated is None:
            aggregated = self.aggregate_results()

        print("\n" + "=" * 60)
        print("Generation 評估報告")
        print("=" * 60)
        print(f"總問題數: {aggregated['total_questions']}")
        print()

        # 主要指標
        key_metrics = [
            ("overall_score", "綜合分數"),
            ("keyword_combined_score", "關鍵詞匹配"),
            ("faith_faithfulness", "忠實度"),
            ("quality_overall_quality", "回答品質"),
            ("semantic_similarity", "語意相似度"),
            ("negative_check_score", "負面檢查"),
        ]

        print("主要指標:")
        for key, name in key_metrics:
            if key in aggregated["metrics"]:
                data = aggregated["metrics"][key]
                print(f"  {name:12}: {data['mean']:.4f} (± {data['std']:.4f})")

        print("=" * 60)
