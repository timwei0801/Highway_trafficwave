"""
混合查詢架構評估腳本
比較混合架構與純 RAG 的效果
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_model.query.hybrid_query_router import HybridQueryRouter
from train_model.query.intent_classifier import QueryIntent


class HybridEvaluator:
    """混合架構評估器"""

    def __init__(self, test_dataset_path: str = None):
        """初始化評估器"""
        self.router = HybridQueryRouter()

        # 載入測試資料集
        if test_dataset_path is None:
            test_dataset_path = os.path.join(
                os.path.dirname(__file__),
                'datasets',
                'golden_qa.json'
            )

        with open(test_dataset_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        self.questions = self.test_data['questions']
        logger.info(f"載入 {len(self.questions)} 個測試問題")

    async def evaluate_single(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """評估單一問題"""
        query = question['question']
        criteria = question['evaluation_criteria']

        # 執行查詢
        start_time = time.time()
        result = await self.router.query(query)
        elapsed_time = (time.time() - start_time) * 1000

        # 評估答案
        answer = result.answer.lower() if result.answer else ""

        # 檢查 must_contain_any
        must_contain_passed = False
        if criteria.get('must_contain_any'):
            for keyword in criteria['must_contain_any']:
                if keyword.lower() in answer:
                    must_contain_passed = True
                    break
        else:
            must_contain_passed = True  # 沒有必須包含的關鍵字

        # 檢查 should_mention
        should_mention_count = 0
        should_mention_total = len(criteria.get('should_mention', []))
        for keyword in criteria.get('should_mention', []):
            if keyword.lower() in answer:
                should_mention_count += 1
        should_mention_score = should_mention_count / should_mention_total if should_mention_total > 0 else 1.0

        # 檢查 must_not_contain
        must_not_contain_passed = True
        for keyword in criteria.get('must_not_contain', []):
            if keyword.lower() in answer:
                must_not_contain_passed = False
                break

        # 計算總分
        score = 0.0
        if must_contain_passed:
            score += 0.5
        score += should_mention_score * 0.3
        if must_not_contain_passed:
            score += 0.2

        return {
            'question_id': question['id'],
            'question': query,
            'category': question['category'],
            'difficulty': question['difficulty'],
            'ground_truth': question['ground_truth']['answer'],
            'model_answer': result.answer,
            'query_type': result.query_type,
            'intent': result.intent.value,
            'confidence': result.confidence,
            'processing_time_ms': elapsed_time,
            'must_contain_passed': must_contain_passed,
            'should_mention_score': should_mention_score,
            'must_not_contain_passed': must_not_contain_passed,
            'total_score': score,
            'success': result.success,
        }

    async def evaluate_all(self) -> Dict[str, Any]:
        """評估所有問題"""
        results = []
        total_questions = len(self.questions)

        for i, question in enumerate(self.questions):
            logger.info(f"評估進度: {i+1}/{total_questions} - {question['id']}")

            try:
                result = await self.evaluate_single(question)
                results.append(result)
            except Exception as e:
                logger.error(f"評估問題 {question['id']} 時發生錯誤: {e}")
                results.append({
                    'question_id': question['id'],
                    'error': str(e),
                    'total_score': 0.0,
                })

        # 計算統計
        stats = self._calculate_statistics(results)

        return {
            'timestamp': datetime.now().isoformat(),
            'total_questions': total_questions,
            'results': results,
            'statistics': stats,
        }

    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """計算統計資料"""
        # 過濾有效結果
        valid_results = [r for r in results if 'total_score' in r and 'error' not in r]

        if not valid_results:
            return {'error': 'No valid results'}

        # 總體統計
        total_score = sum(r['total_score'] for r in valid_results) / len(valid_results)
        must_contain_rate = sum(1 for r in valid_results if r.get('must_contain_passed', False)) / len(valid_results)
        must_not_contain_rate = sum(1 for r in valid_results if r.get('must_not_contain_passed', False)) / len(valid_results)
        avg_should_mention = sum(r.get('should_mention_score', 0) for r in valid_results) / len(valid_results)

        # 按查詢類型統計
        by_query_type = {}
        for r in valid_results:
            qtype = r.get('query_type', 'unknown')
            if qtype not in by_query_type:
                by_query_type[qtype] = {'count': 0, 'total_score': 0, 'total_time': 0}
            by_query_type[qtype]['count'] += 1
            by_query_type[qtype]['total_score'] += r['total_score']
            by_query_type[qtype]['total_time'] += r.get('processing_time_ms', 0)

        for qtype in by_query_type:
            count = by_query_type[qtype]['count']
            by_query_type[qtype]['avg_score'] = by_query_type[qtype]['total_score'] / count
            by_query_type[qtype]['avg_time_ms'] = by_query_type[qtype]['total_time'] / count

        # 按問題類別統計
        by_category = {}
        for r in valid_results:
            cat = r.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = {'count': 0, 'total_score': 0}
            by_category[cat]['count'] += 1
            by_category[cat]['total_score'] += r['total_score']

        for cat in by_category:
            count = by_category[cat]['count']
            by_category[cat]['avg_score'] = by_category[cat]['total_score'] / count

        # 按難度統計
        by_difficulty = {}
        for r in valid_results:
            diff = r.get('difficulty', 'unknown')
            if diff not in by_difficulty:
                by_difficulty[diff] = {'count': 0, 'total_score': 0}
            by_difficulty[diff]['count'] += 1
            by_difficulty[diff]['total_score'] += r['total_score']

        for diff in by_difficulty:
            count = by_difficulty[diff]['count']
            by_difficulty[diff]['avg_score'] = by_difficulty[diff]['total_score'] / count

        # 平均處理時間
        avg_processing_time = sum(r.get('processing_time_ms', 0) for r in valid_results) / len(valid_results)

        # 查詢類型分佈
        precise_count = sum(1 for r in valid_results if r.get('query_type') == 'precise')
        rag_count = sum(1 for r in valid_results if r.get('query_type') == 'rag')

        return {
            'overall': {
                'total_score': total_score,
                'must_contain_rate': must_contain_rate,
                'must_not_contain_rate': must_not_contain_rate,
                'avg_should_mention': avg_should_mention,
                'avg_processing_time_ms': avg_processing_time,
            },
            'by_query_type': by_query_type,
            'by_category': by_category,
            'by_difficulty': by_difficulty,
            'query_type_distribution': {
                'precise': precise_count,
                'rag': rag_count,
                'precise_ratio': precise_count / len(valid_results) if valid_results else 0,
            },
        }


async def main():
    """執行評估"""
    print("=" * 70)
    print("混合查詢架構評估")
    print("=" * 70)

    evaluator = HybridEvaluator()
    results = await evaluator.evaluate_all()

    # 輸出結果
    stats = results['statistics']

    print("\n" + "=" * 70)
    print("評估結果摘要")
    print("=" * 70)

    print(f"\n📊 總體分數: {stats['overall']['total_score']:.2%}")
    print(f"   必須包含通過率: {stats['overall']['must_contain_rate']:.2%}")
    print(f"   應該提及平均分: {stats['overall']['avg_should_mention']:.2%}")
    print(f"   禁止包含通過率: {stats['overall']['must_not_contain_rate']:.2%}")
    print(f"   平均處理時間: {stats['overall']['avg_processing_time_ms']:.1f}ms")

    print(f"\n📈 查詢類型分佈:")
    dist = stats['query_type_distribution']
    print(f"   精確查詢: {dist['precise']} ({dist['precise_ratio']:.1%})")
    print(f"   RAG 查詢: {dist['rag']} ({1-dist['precise_ratio']:.1%})")

    print(f"\n📊 按查詢類型:")
    for qtype, data in stats['by_query_type'].items():
        print(f"   {qtype}: {data['avg_score']:.2%} (n={data['count']}, avg={data['avg_time_ms']:.0f}ms)")

    print(f"\n📊 按問題類別:")
    for cat, data in stats['by_category'].items():
        print(f"   {cat}: {data['avg_score']:.2%} (n={data['count']})")

    print(f"\n📊 按難度:")
    for diff, data in stats['by_difficulty'].items():
        print(f"   {diff}: {data['avg_score']:.2%} (n={data['count']})")

    # 儲存結果
    output_path = os.path.join(
        os.path.dirname(__file__),
        'reports',
        f'hybrid_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📁 詳細結果已儲存至: {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
