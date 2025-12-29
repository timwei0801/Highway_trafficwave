"""
RAG 系統評估器
整合 Retrieval 和 Generation 評估
"""

import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

# 設置 logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from .metrics.retrieval_metrics import (
    RetrievalResult,
    RetrievalEvaluator,
    SimpleRetrievalEvaluator
)
from .metrics.generation_metrics import (
    GenerationResult,
    GenerationEvaluator
)
from .ab_comparator import ABComparator, ExperimentTracker


class RAGEvaluator:
    """RAG 系統完整評估器"""

    def __init__(
        self,
        rag_system,
        vector_store=None,
        embedding_model=None,
        reports_dir: str = None
    ):
        """
        Args:
            rag_system: RAG 聊天系統實例
            vector_store: 向量存儲實例（用於取得檢索結果）
            embedding_model: 嵌入模型（用於語意相似度）
            reports_dir: 報告輸出目錄
        """
        self.rag_system = rag_system
        self.vector_store = vector_store
        self.embedding_model = embedding_model

        self.reports_dir = Path(reports_dir) if reports_dir else Path(__file__).parent / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 初始化子評估器
        self.retrieval_evaluator = SimpleRetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator(embedding_model)
        self.comparator = ABComparator(str(self.reports_dir))

        # 追蹤器
        self.tracker = ExperimentTracker(str(self.reports_dir / "experiments.json"))

    def load_test_dataset(self, dataset_path: str = None) -> Dict[str, Any]:
        """載入測試資料集"""
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "datasets" / "golden_qa.json"

        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    async def evaluate_single_question(
        self,
        question_data: Dict[str, Any],
        include_retrieval: bool = True
    ) -> Dict[str, Any]:
        """
        評估單一問題

        Args:
            question_data: 問題資料（來自 golden_qa.json）
            include_retrieval: 是否評估檢索階段

        Returns:
            評估結果
        """
        question = question_data["question"]
        logger.info(f"評估問題: {question[:50]}...")

        result = {
            "question_id": question_data["id"],
            "question": question,
            "category": question_data.get("category", "unknown"),
            "difficulty": question_data.get("difficulty", "unknown"),
        }

        try:
            # 1. 取得 RAG 回答
            answer = await self.rag_system.chat(question)
            result["generated_answer"] = answer

            # 2. 取得檢索結果（如果有 vector_store）
            retrieved_context = ""
            if include_retrieval and self.vector_store:
                try:
                    retrieval_results = self.vector_store.search(question, top_k=5)
                    retrieved_texts = [r.get("text", "") for r in retrieval_results]
                    retrieved_scores = [r.get("score", 0) for r in retrieval_results]
                    retrieved_context = "\n".join(retrieved_texts)

                    # Retrieval 評估
                    expected_keywords = question_data.get("evaluation_criteria", {}).get(
                        "should_mention", []
                    )
                    retrieval_eval = self.retrieval_evaluator.evaluate_by_keywords(
                        question,
                        retrieved_texts,
                        expected_keywords,
                        retrieved_scores
                    )
                    result["retrieval_metrics"] = retrieval_eval
                except Exception as e:
                    logger.warning(f"Retrieval 評估失敗: {e}")

            # 3. Generation 評估
            ground_truth = question_data.get("ground_truth", {})
            criteria = question_data.get("evaluation_criteria", {})

            gen_result = GenerationResult(
                question=question,
                generated_answer=answer,
                ground_truth_answer=ground_truth.get("answer", ""),
                retrieved_context=retrieved_context,
                evaluation_criteria=criteria
            )

            generation_eval = self.generation_evaluator.evaluate_single(gen_result)
            result["generation_metrics"] = generation_eval

            # 4. 綜合分數
            result["overall_score"] = generation_eval.get("overall_score", 0)
            result["success"] = True

        except Exception as e:
            logger.error(f"評估問題時發生錯誤: {e}")
            result["success"] = False
            result["error"] = str(e)
            result["overall_score"] = 0

        return result

    async def run_full_evaluation(
        self,
        dataset_path: str = None,
        include_retrieval: bool = True,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        執行完整評估

        Args:
            dataset_path: 測試資料集路徑
            include_retrieval: 是否評估檢索階段
            save_report: 是否儲存報告

        Returns:
            完整評估報告
        """
        logger.info("開始執行 RAG 系統完整評估...")

        # 載入測試資料
        dataset = self.load_test_dataset(dataset_path)
        questions = dataset.get("questions", [])
        logger.info(f"載入 {len(questions)} 個測試問題")

        # 清除聊天歷史
        if hasattr(self.rag_system, 'clear_history'):
            self.rag_system.clear_history()

        # 評估每個問題
        results = []
        for i, q in enumerate(questions, 1):
            logger.info(f"進度: {i}/{len(questions)}")
            result = await self.evaluate_single_question(q, include_retrieval)
            results.append(result)
            await asyncio.sleep(1)  # 避免過快請求

        # 彙總結果
        report = self._compile_report(results, dataset)

        # 儲存報告
        if save_report:
            self._save_report(report)

        return report

    def _compile_report(
        self,
        results: List[Dict[str, Any]],
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """彙總評估報告"""
        successful_results = [r for r in results if r.get("success", False)]

        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_version": dataset.get("version", "unknown"),
            "total_questions": len(results),
            "successful_evaluations": len(successful_results),
            "failed_evaluations": len(results) - len(successful_results),
        }

        if not successful_results:
            report["error"] = "No successful evaluations"
            return report

        # 計算整體指標
        overall_scores = [r["overall_score"] for r in successful_results]
        report["metrics"] = {
            "overall_score": {
                "mean": float(np.mean(overall_scores)),
                "std": float(np.std(overall_scores)),
                "min": float(np.min(overall_scores)),
                "max": float(np.max(overall_scores)),
            }
        }

        # 彙總 Generation 指標
        gen_metrics = self.generation_evaluator.aggregate_results()
        if "metrics" in gen_metrics:
            report["metrics"].update(gen_metrics["metrics"])

        # 彙總 Retrieval 指標
        ret_metrics = self.retrieval_evaluator.aggregate_results()
        if "error" not in ret_metrics:
            report["retrieval_summary"] = {
                "avg_score": ret_metrics.get("avg_retrieval_score", 0),
                "avg_keyword_coverage": ret_metrics.get("avg_keyword_coverage", 0),
                "hit_rate": ret_metrics.get("hit_rate", 0),
            }

        # 按難度分組
        difficulty_groups = {}
        for r in successful_results:
            diff = r.get("difficulty", "unknown")
            if diff not in difficulty_groups:
                difficulty_groups[diff] = []
            difficulty_groups[diff].append(r["overall_score"])

        report["by_difficulty"] = {
            diff: {
                "count": len(scores),
                "mean_score": float(np.mean(scores))
            }
            for diff, scores in difficulty_groups.items()
        }

        # 按類別分組
        category_groups = {}
        for r in successful_results:
            cat = r.get("category", "unknown")
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(r["overall_score"])

        report["by_category"] = {
            cat: {
                "count": len(scores),
                "mean_score": float(np.mean(scores))
            }
            for cat, scores in category_groups.items()
        }

        # 詳細結果
        report["detailed_results"] = results

        return report

    def _save_report(self, report: Dict[str, Any]) -> str:
        """儲存報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{timestamp}.json"
        filepath = self.reports_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"評估報告已儲存至: {filepath}")
        return str(filepath)

    def print_report(self, report: Dict[str, Any]):
        """印出評估報告"""
        print("\n" + "=" * 70)
        print("RAG 系統評估報告")
        print("=" * 70)
        print(f"時間: {report['timestamp']}")
        print(f"資料集版本: {report['dataset_version']}")
        print(f"總問題數: {report['total_questions']}")
        print(f"成功評估: {report['successful_evaluations']}")
        print(f"失敗評估: {report['failed_evaluations']}")
        print()

        if "error" in report:
            print(f"錯誤: {report['error']}")
            return

        # 主要指標
        print("-" * 70)
        print("主要指標:")
        print("-" * 70)

        metrics = report.get("metrics", {})
        key_metrics = [
            ("overall_score", "綜合分數"),
            ("keyword_combined_score", "關鍵詞匹配"),
            ("faith_faithfulness", "忠實度"),
            ("quality_overall_quality", "回答品質"),
        ]

        for key, name in key_metrics:
            if key in metrics:
                data = metrics[key]
                print(f"  {name:15}: {data['mean']:.4f} (± {data['std']:.4f})")

        # Retrieval 摘要
        if "retrieval_summary" in report:
            print()
            print("Retrieval 指標:")
            ret = report["retrieval_summary"]
            print(f"  平均分數: {ret['avg_score']:.4f}")
            print(f"  關鍵詞覆蓋: {ret['avg_keyword_coverage']:.4f}")
            print(f"  命中率: {ret['hit_rate']:.4f}")

        # 按難度
        print()
        print("按難度分組:")
        for diff, data in report.get("by_difficulty", {}).items():
            print(f"  {diff}: {data['mean_score']:.4f} ({data['count']} 題)")

        # 按類別
        print()
        print("按類別分組:")
        for cat, data in report.get("by_category", {}).items():
            print(f"  {cat}: {data['mean_score']:.4f} ({data['count']} 題)")

        print("=" * 70)

    def record_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        report: Dict[str, Any],
        notes: str = ""
    ):
        """記錄實驗結果"""
        self.tracker.record_experiment(name, config, report, notes)

    def compare_with_baseline(
        self,
        baseline_report: Dict[str, Any],
        current_report: Dict[str, Any],
        baseline_name: str = "Baseline",
        current_name: str = "Current"
    ) -> Dict[str, Any]:
        """與基準比較"""
        comparison = self.comparator.compare(
            baseline_report,
            current_report,
            baseline_name,
            current_name
        )
        self.comparator.print_comparison(comparison)
        return comparison


# 獨立執行評估的輔助函數
async def run_evaluation(
    rag_system,
    dataset_path: str = None,
    vector_store=None,
    embedding_model=None
) -> Dict[str, Any]:
    """
    執行評估的便捷函數

    Args:
        rag_system: RAG 聊天系統
        dataset_path: 測試資料集路徑
        vector_store: 向量存儲
        embedding_model: 嵌入模型

    Returns:
        評估報告
    """
    evaluator = RAGEvaluator(
        rag_system=rag_system,
        vector_store=vector_store,
        embedding_model=embedding_model
    )

    report = await evaluator.run_full_evaluation(dataset_path)
    evaluator.print_report(report)

    return report
