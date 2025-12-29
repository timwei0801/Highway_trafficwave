#!/usr/bin/env python3
"""
RAG 系統評估執行腳本

使用方式:
    # 執行完整評估
    python run_evaluation.py

    # 指定測試資料集
    python run_evaluation.py --dataset path/to/dataset.json

    # 比較兩個評估結果
    python run_evaluation.py --compare baseline.json experiment.json

    # 檢視實驗歷史
    python run_evaluation.py --history
"""

import sys
import asyncio
import argparse
from pathlib import Path

# 加入專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def run_full_evaluation(args):
    """執行完整評估"""
    from train_model.evaluation import RAGEvaluator
    from train_model.models.ollama_client import OllamaClient, RAGOllamaChat
    from train_model.embeddings.vector_store import VectorStore, RAGRetriever

    logger.info("初始化 RAG 系統...")

    # 初始化組件
    try:
        # 初始化 Ollama 客戶端
        ollama_client = OllamaClient()

        # 檢查 Ollama 連接
        if not await ollama_client.check_connection():
            logger.error("無法連接到 Ollama 服務，請確認 Ollama 正在執行")
            return None

        # 初始化向量存儲和檢索器
        vector_store = VectorStore()
        retriever = RAGRetriever(vector_store)

        # 初始化 RAG 聊天系統
        rag_chat = RAGOllamaChat(ollama_client=ollama_client, vector_store_retriever=retriever)
    except Exception as e:
        logger.error(f"初始化失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 建立評估器
    evaluator = RAGEvaluator(
        rag_system=rag_chat,
        vector_store=vector_store
    )

    # 執行評估
    dataset_path = args.dataset if args.dataset else None
    report = await evaluator.run_full_evaluation(
        dataset_path=dataset_path,
        include_retrieval=(vector_store is not None),
        save_report=True
    )

    # 印出報告
    evaluator.print_report(report)

    # 記錄實驗（可選）
    if args.name:
        config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "deepseek-r1:32b",
            "top_k": 5,
            "threshold": 0.7,
        }
        evaluator.record_experiment(
            name=args.name,
            config=config,
            report=report,
            notes=args.notes or ""
        )

    return report


def compare_results(args):
    """比較兩個評估結果"""
    from train_model.evaluation import ABComparator

    comparator = ABComparator()

    comparison = comparator.compare_from_files(
        baseline_file=args.compare[0],
        experiment_file=args.compare[1],
        baseline_name=args.baseline_name or "Baseline",
        experiment_name=args.experiment_name or "Experiment"
    )

    comparator.print_comparison(comparison)

    if args.save:
        comparator.save_comparison(comparison)


def show_history(args):
    """顯示實驗歷史"""
    from train_model.evaluation import ExperimentTracker

    reports_dir = Path(__file__).parent / "reports"
    tracker = ExperimentTracker(str(reports_dir / "experiments.json"))
    tracker.print_history(last_n=args.last or 10)


def quick_test():
    """快速測試評估框架是否正常運作"""
    from train_model.evaluation.metrics.retrieval_metrics import (
        RetrievalMetrics, SimpleRetrievalEvaluator
    )
    from train_model.evaluation.metrics.generation_metrics import (
        GenerationMetrics, GenerationEvaluator
    )

    print("=" * 60)
    print("評估框架快速測試")
    print("=" * 60)

    # 測試 Retrieval 指標
    print("\n1. Retrieval 指標測試:")
    metrics = RetrievalMetrics()

    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
    relevant = {"chunk_2", "chunk_5"}

    print(f"   Recall@3: {metrics.recall_at_k(retrieved, relevant, 3):.2f}")
    print(f"   Precision@3: {metrics.precision_at_k(retrieved, relevant, 3):.2f}")
    print(f"   MRR: {metrics.mrr(retrieved, relevant):.2f}")
    print(f"   NDCG@5: {metrics.ndcg_at_k(retrieved, relevant, 5):.2f}")

    # 測試 Generation 指標
    print("\n2. Generation 指標測試:")
    gen_metrics = GenerationMetrics()

    generated = "國道一號34K處車道寬度約3.5公尺，共有6車道配置。"
    context = "該路段有6車道，車道1寬約3.56公尺，車道2寬約3.49公尺。"

    keyword_result = gen_metrics.keyword_precision(
        generated,
        must_contain=["3.5", "車道"],
        should_mention=["公尺", "6車道"]
    )
    print(f"   關鍵詞匹配: {keyword_result['combined_score']:.2f}")

    faithfulness = gen_metrics.faithfulness(generated, context)
    print(f"   忠實度: {faithfulness['faithfulness']:.2f}")

    quality = gen_metrics.answer_quality(generated)
    print(f"   回答品質: {quality['overall_quality']:.2f}")

    print("\n3. 簡化版 Retrieval 評估測試:")
    simple_eval = SimpleRetrievalEvaluator()
    result = simple_eval.evaluate_by_keywords(
        query="車道寬度",
        retrieved_texts=["車道寬3.5公尺", "路段資訊", "交通資料"],
        expected_keywords=["車道", "寬度", "公尺"],
        scores=[0.9, 0.7, 0.5]
    )
    print(f"   關鍵詞覆蓋率: {result['keyword_coverage']['mean']:.2f}")
    print(f"   最佳結果排名: {result['best_result_rank']}")

    print("\n" + "=" * 60)
    print("測試完成！評估框架運作正常。")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="RAG 系統評估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 執行完整評估
  python run_evaluation.py

  # 使用自訂資料集
  python run_evaluation.py --dataset my_dataset.json

  # 記錄實驗
  python run_evaluation.py --name "升級embedding" --notes "使用bge-large-zh"

  # 比較兩個結果
  python run_evaluation.py --compare baseline.json experiment.json

  # 快速測試
  python run_evaluation.py --test
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="測試資料集路徑"
    )
    parser.add_argument(
        "--name",
        type=str,
        help="實驗名稱（用於記錄）"
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="實驗備註"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "EXPERIMENT"),
        help="比較兩個評估結果檔案"
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        help="基準組名稱"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="實驗組名稱"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="儲存比較結果"
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="顯示實驗歷史"
    )
    parser.add_argument(
        "--last",
        type=int,
        default=10,
        help="顯示最近 N 筆實驗"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="快速測試評估框架"
    )

    args = parser.parse_args()

    if args.test:
        quick_test()
    elif args.compare:
        compare_results(args)
    elif args.history:
        show_history(args)
    else:
        asyncio.run(run_full_evaluation(args))


if __name__ == "__main__":
    main()
