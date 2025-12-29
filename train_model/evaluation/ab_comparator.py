"""
A/B 比較工具
比較兩個 RAG 系統配置的效果差異
"""

import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# 簡易 t-test 實作，避免 scipy 依賴
def simple_ttest_ind(a, b):
    """簡易獨立樣本 t 檢定"""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None

    mean1, mean2 = np.mean(a), np.mean(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)

    # Pooled standard error
    se = np.sqrt(var1/n1 + var2/n2)
    if se == 0:
        return None

    t_stat = (mean1 - mean2) / se

    # 簡化的 p-value 估計（使用常態近似）
    # 對於大樣本這是合理的近似
    df = n1 + n2 - 2
    # 簡單閾值判斷
    if abs(t_stat) > 2.0:
        return 0.05  # 可能顯著
    elif abs(t_stat) > 1.5:
        return 0.15
    else:
        return 0.5  # 不顯著


class ABComparator:
    """A/B 比較工具"""

    def __init__(self, reports_dir: str = None):
        """
        Args:
            reports_dir: 報告儲存目錄
        """
        self.reports_dir = Path(reports_dir) if reports_dir else Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def compare(
        self,
        baseline_results: Dict[str, Any],
        experiment_results: Dict[str, Any],
        baseline_name: str = "Baseline",
        experiment_name: str = "Experiment",
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        比較兩組評估結果

        Args:
            baseline_results: 基準組評估結果
            experiment_results: 實驗組評估結果
            baseline_name: 基準組名稱
            experiment_name: 實驗組名稱
            significance_level: 統計顯著性水準

        Returns:
            比較結果
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "baseline": baseline_name,
            "experiment": experiment_name,
            "significance_level": significance_level,
            "metrics_comparison": {},
            "summary": {},
        }

        # 取得兩組的指標
        baseline_metrics = baseline_results.get("metrics", {})
        experiment_metrics = experiment_results.get("metrics", {})

        # 找出共同的指標
        common_metrics = set(baseline_metrics.keys()) & set(experiment_metrics.keys())

        improvements = 0
        degradations = 0
        no_change = 0

        for metric in sorted(common_metrics):
            base_data = baseline_metrics[metric]
            exp_data = experiment_metrics[metric]

            # 計算改善幅度
            base_mean = base_data["mean"]
            exp_mean = exp_data["mean"]

            if base_mean > 0:
                improvement_pct = ((exp_mean - base_mean) / base_mean) * 100
            else:
                improvement_pct = 0.0 if exp_mean == 0 else float('inf')

            # 進行統計顯著性檢定（如果有詳細結果）
            p_value = None
            is_significant = False

            baseline_detailed = baseline_results.get("detailed_results", [])
            experiment_detailed = experiment_results.get("detailed_results", [])

            if baseline_detailed and experiment_detailed:
                base_values = [r.get(metric, 0) for r in baseline_detailed
                              if metric in r and isinstance(r.get(metric), (int, float))]
                exp_values = [r.get(metric, 0) for r in experiment_detailed
                             if metric in r and isinstance(r.get(metric), (int, float))]

                if len(base_values) >= 2 and len(exp_values) >= 2:
                    # 使用簡易獨立樣本 t 檢定
                    try:
                        p_value = simple_ttest_ind(base_values, exp_values)
                        if p_value is not None:
                            is_significant = p_value < significance_level
                    except Exception:
                        pass

            # 判斷結果
            if improvement_pct > 1:  # 超過 1% 視為改善
                improvements += 1
                verdict = "improved"
            elif improvement_pct < -1:  # 低於 -1% 視為退步
                degradations += 1
                verdict = "degraded"
            else:
                no_change += 1
                verdict = "no_change"

            comparison["metrics_comparison"][metric] = {
                "baseline_mean": base_mean,
                "experiment_mean": exp_mean,
                "improvement_pct": improvement_pct,
                "p_value": p_value,
                "is_significant": is_significant,
                "verdict": verdict,
            }

        # 總結
        total_metrics = len(common_metrics)
        comparison["summary"] = {
            "total_metrics_compared": total_metrics,
            "improved": improvements,
            "degraded": degradations,
            "no_change": no_change,
            "improvement_rate": improvements / total_metrics if total_metrics > 0 else 0,
            "overall_verdict": self._get_overall_verdict(improvements, degradations, total_metrics),
        }

        return comparison

    def _get_overall_verdict(
        self,
        improvements: int,
        degradations: int,
        total: int
    ) -> str:
        """判斷整體結果"""
        if total == 0:
            return "insufficient_data"

        improvement_rate = improvements / total
        degradation_rate = degradations / total

        if improvement_rate >= 0.6 and degradation_rate < 0.2:
            return "significant_improvement"
        elif improvement_rate >= 0.4 and degradation_rate < 0.3:
            return "moderate_improvement"
        elif degradation_rate >= 0.4:
            return "degradation"
        else:
            return "inconclusive"

    def print_comparison(self, comparison: Dict[str, Any]):
        """印出比較結果"""
        print("\n" + "=" * 70)
        print("A/B 比較報告")
        print("=" * 70)
        print(f"基準組: {comparison['baseline']}")
        print(f"實驗組: {comparison['experiment']}")
        print(f"時間: {comparison['timestamp']}")
        print()

        # 總結
        summary = comparison["summary"]
        print(f"總指標數: {summary['total_metrics_compared']}")
        print(f"  改善: {summary['improved']} ({summary['improved']/summary['total_metrics_compared']*100:.1f}%)")
        print(f"  退步: {summary['degraded']} ({summary['degraded']/summary['total_metrics_compared']*100:.1f}%)")
        print(f"  持平: {summary['no_change']}")
        print()

        verdict_map = {
            "significant_improvement": "顯著改善 ✅",
            "moderate_improvement": "中等改善 ⬆️",
            "degradation": "效果退步 ⚠️",
            "inconclusive": "結果不確定 ❓",
            "insufficient_data": "資料不足 ❌",
        }
        print(f"整體判定: {verdict_map.get(summary['overall_verdict'], summary['overall_verdict'])}")
        print()

        # 詳細指標比較
        print("-" * 70)
        print(f"{'指標':<30} {'基準':<10} {'實驗':<10} {'變化%':<10} {'顯著性':<10}")
        print("-" * 70)

        for metric, data in comparison["metrics_comparison"].items():
            sig_marker = "*" if data["is_significant"] else ""
            change_sign = "+" if data["improvement_pct"] > 0 else ""

            # 顏色標記
            if data["verdict"] == "improved":
                verdict_icon = "↑"
            elif data["verdict"] == "degraded":
                verdict_icon = "↓"
            else:
                verdict_icon = "→"

            print(
                f"{metric:<30} "
                f"{data['baseline_mean']:<10.4f} "
                f"{data['experiment_mean']:<10.4f} "
                f"{change_sign}{data['improvement_pct']:<9.2f} "
                f"{verdict_icon}{sig_marker}"
            )

        print("=" * 70)
        print("* 表示統計顯著 (p < 0.05)")

    def save_comparison(
        self,
        comparison: Dict[str, Any],
        filename: str = None
    ) -> str:
        """儲存比較結果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ab_comparison_{timestamp}.json"

        filepath = self.reports_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2, default=str)

        print(f"比較結果已儲存至: {filepath}")
        return str(filepath)

    def load_evaluation_result(self, filepath: str) -> Dict[str, Any]:
        """載入評估結果"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def compare_from_files(
        self,
        baseline_file: str,
        experiment_file: str,
        baseline_name: str = None,
        experiment_name: str = None
    ) -> Dict[str, Any]:
        """從檔案載入並比較"""
        baseline = self.load_evaluation_result(baseline_file)
        experiment = self.load_evaluation_result(experiment_file)

        if baseline_name is None:
            baseline_name = Path(baseline_file).stem
        if experiment_name is None:
            experiment_name = Path(experiment_file).stem

        return self.compare(baseline, experiment, baseline_name, experiment_name)


class ExperimentTracker:
    """實驗追蹤器 - 記錄多次實驗的結果"""

    def __init__(self, tracker_file: str = "experiments.json"):
        self.tracker_file = Path(tracker_file)
        self.experiments = self._load_experiments()

    def _load_experiments(self) -> List[Dict[str, Any]]:
        """載入實驗記錄"""
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_experiments(self):
        """儲存實驗記錄"""
        with open(self.tracker_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiments, f, ensure_ascii=False, indent=2, default=str)

    def record_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        results: Dict[str, Any],
        notes: str = ""
    ):
        """記錄一次實驗"""
        experiment = {
            "id": len(self.experiments) + 1,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "results_summary": {
                k: v.get("mean") if isinstance(v, dict) else v
                for k, v in results.get("metrics", {}).items()
            },
            "notes": notes,
        }

        self.experiments.append(experiment)
        self._save_experiments()

        print(f"實驗 #{experiment['id']} '{name}' 已記錄")
        return experiment["id"]

    def get_best_experiment(self, metric: str = "overall_score") -> Optional[Dict[str, Any]]:
        """取得某指標最佳的實驗"""
        if not self.experiments:
            return None

        best = max(
            self.experiments,
            key=lambda x: x["results_summary"].get(metric, 0)
        )
        return best

    def print_history(self, last_n: int = 10):
        """印出實驗歷史"""
        print("\n" + "=" * 80)
        print("實驗歷史")
        print("=" * 80)

        for exp in self.experiments[-last_n:]:
            print(f"\n#{exp['id']} {exp['name']} ({exp['timestamp'][:10]})")
            print(f"  Config: {json.dumps(exp['config'], ensure_ascii=False)[:60]}...")

            # 顯示主要指標
            if "overall_score" in exp["results_summary"]:
                print(f"  Overall Score: {exp['results_summary']['overall_score']:.4f}")

            if exp["notes"]:
                print(f"  Notes: {exp['notes']}")

        print("=" * 80)
