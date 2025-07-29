from typing import Any, Dict, List

from ..utils.config import Config
from ..utils.exceptions import QualityEvaluationError
from ..utils.logger import Logger


class QualityEvaluator:
    """品質評価クラス"""

    def __init__(self, config: Config):
        """
        初期化

        Args:
            config: 設定管理オブジェクト
        """
        self.config = config
        self.logger = Logger(config.get_logging_config()).get_logger("quality_evaluator")
        self.quality_config = config.get_quality_config()

    def evaluate_quality_level(self, r2_score: float) -> str:
        """
        R²スコアに基づく品質レベル評価（過度な精度チェック付き）

        Args:
            r2_score: R²スコア

        Returns:
            品質レベル（Premium/Standard/Basic/Rejected）
        """
        thresholds = self.quality_config.get("thresholds", {})
        premium_threshold = thresholds.get("premium", 0.7)
        standard_threshold = thresholds.get("standard", 0.5)
        basic_threshold = thresholds.get("basic", 0.3)
        max_r2_score = self.quality_config.get("max_r2_score", 0.90)

        # 過度に高いR²スコアの警告・降格
        if r2_score > max_r2_score:
            self.logger.warning(
                f"過度に高いR²スコア検出: {r2_score:.4f} > {max_r2_score} "
                f"(過学習または実装問題の可能性)"
            )
            # 品質レベルを1段階降格
            if r2_score >= premium_threshold:
                return "Standard"  # Premium → Standard
            elif r2_score >= standard_threshold:
                return "Basic"     # Standard → Basic
            else:
                return "Rejected"  # Basic → Rejected

        # 通常の品質評価
        if r2_score >= premium_threshold:
            return "Premium"
        elif r2_score >= standard_threshold:
            return "Standard"
        elif r2_score >= basic_threshold:
            return "Basic"
        else:
            return "Rejected"

    def assess_implementation_readiness(self, model_metrics: Dict[str, float]) -> str:
        """
        実用化準備状況を評価

        Args:
            model_metrics: モデル評価メトリクス

        Returns:
            実用化準備状況
        """
        r2_score = model_metrics.get("r2_score", 0.0)
        overfitting_score = model_metrics.get("overfitting_score", 1.0)

        quality_level = self.evaluate_quality_level(r2_score)

        # 過学習チェック（厳格化）
        overfitting_threshold = self.quality_config.get("overfitting_threshold", 0.01)
        is_overfitting = overfitting_score > overfitting_threshold

        # Train/Validation差によるさらなる過学習検知
        cv_mean = model_metrics.get("cv_mean_r2", 0.0)
        train_val_gap = abs(r2_score - cv_mean)
        severe_overfitting = train_val_gap > 0.1  # Train/Val差が10%以上は重篤な過学習

        if severe_overfitting:
            self.logger.warning(f"重篤な過学習を検知: Train/Val差={train_val_gap:.3f}")
            return "改善必要"
        elif is_overfitting:
            self.logger.warning(f"過学習を検知: スコア={overfitting_score:.3f}")
            return "改善必要"
        elif quality_level == "Premium":
            return "即座実行"
        elif quality_level == "Standard":
            return "慎重実行"
        elif quality_level == "Basic":
            return "要考慮"
        else:
            return "改善必要"

    def calculate_category_success_rate(self, results: List[Dict]) -> Dict[str, float]:
        """
        商品カテゴリ別成功率を算出

        Args:
            results: 分析結果リスト

        Returns:
            カテゴリ別成功率辞書
        """
        category_stats = {}

        for result in results:
            category = result.get("category", "その他")
            quality_level = result.get("quality_level", "Rejected")

            if category not in category_stats:
                category_stats[category] = {"total": 0, "success": 0}

            category_stats[category]["total"] += 1

            # Premium/Standardを成功とみなす
            if quality_level in ["Premium", "Standard"]:
                category_stats[category]["success"] += 1

        # 成功率計算
        success_rates = {}
        for category, stats in category_stats.items():
            if stats["total"] > 0:
                success_rates[category] = stats["success"] / stats["total"]
            else:
                success_rates[category] = 0.0

        self.logger.info(f"カテゴリ別成功率: {success_rates}")

        return success_rates

    def generate_quality_dashboard_data(self, results: List[Dict]) -> Dict[str, Any]:
        """
        品質ダッシュボード用データを生成

        Args:
            results: 分析結果リスト

        Returns:
            ダッシュボードデータ
        """
        if not results:
            return {}

        # 品質レベル分布
        quality_distribution = {}
        implementation_distribution = {}
        r2_scores = []
        overfitting_scores = []

        for result in results:
            # 品質レベル分布
            quality_level = result.get("quality_level", "Rejected")
            quality_distribution[quality_level] = quality_distribution.get(quality_level, 0) + 1

            # 実用化準備状況分布
            implementation_status = result.get("implementation_readiness", "改善必要")
            implementation_distribution[implementation_status] = (
                implementation_distribution.get(implementation_status, 0) + 1
            )

            # スコア収集
            if "test_metrics" in result:
                r2_scores.append(result["test_metrics"].get("r2_score", 0.0))
                overfitting_scores.append(result.get("overfitting_score", 1.0))

        # 統計値計算
        import numpy as np

        dashboard_data = {
            "total_products": len(results),
            "quality_distribution": quality_distribution,
            "implementation_distribution": implementation_distribution,
            "success_rate": self._calculate_overall_success_rate(quality_distribution),
            "average_r2": np.mean(r2_scores) if r2_scores else 0.0,
            "r2_std": np.std(r2_scores) if r2_scores else 0.0,
            "average_overfitting": np.mean(overfitting_scores) if overfitting_scores else 1.0,
            "category_success_rates": self.calculate_category_success_rate(results),
        }

        return dashboard_data

    def _calculate_overall_success_rate(self, quality_distribution: Dict[str, int]) -> float:
        """全体成功率を計算"""
        total = sum(quality_distribution.values())
        if total == 0:
            return 0.0

        success_count = quality_distribution.get("Premium", 0) + quality_distribution.get(
            "Standard", 0
        )
        return success_count / total

    def evaluate_model_reliability(self, model_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        モデルの信頼性を評価

        Args:
            model_metrics: モデル評価メトリクス

        Returns:
            信頼性評価結果
        """
        r2_score = model_metrics.get("r2_score", 0.0)
        cv_mean = model_metrics.get("cv_mean_r2", 0.0)
        cv_std = model_metrics.get("cv_std_r2", 1.0)
        overfitting_score = model_metrics.get("overfitting_score", 1.0)

        # 信頼性指標
        reliability_indicators = {
            "quality_level": self.evaluate_quality_level(r2_score),
            "implementation_readiness": self.assess_implementation_readiness(model_metrics),
            "cv_stability": "安定" if cv_std < 0.1 else "不安定" if cv_std > 0.2 else "普通",
            "overfitting_risk": (
                "高" if overfitting_score > 0.01 else "低" if overfitting_score < 0.005 else "中"
            ),
            "performance_consistency": abs(r2_score - cv_mean) < 0.05,
        }

        # 総合信頼性スコア
        reliability_score = self._calculate_reliability_score(
            reliability_indicators, model_metrics
        )

        return {
            "reliability_score": reliability_score,
            "indicators": reliability_indicators,
            "recommendations": self._generate_recommendations(reliability_indicators),
        }

    def _calculate_reliability_score(
        self, indicators: Dict[str, Any], metrics: Dict[str, float]
    ) -> float:
        """信頼性スコアを計算"""
        score = 0.0

        # 品質レベルによる配点
        quality_scores = {"Premium": 0.4, "Standard": 0.3, "Basic": 0.2, "Rejected": 0.0}
        score += quality_scores.get(indicators["quality_level"], 0.0)

        # CV安定性による配点
        stability_scores = {"安定": 0.2, "普通": 0.1, "不安定": 0.0}
        score += stability_scores.get(indicators["cv_stability"], 0.0)

        # 過学習リスクによる配点
        overfitting_scores = {"低": 0.2, "中": 0.1, "高": 0.0}
        score += overfitting_scores.get(indicators["overfitting_risk"], 0.0)

        # 性能一貫性による配点
        if indicators["performance_consistency"]:
            score += 0.2

        return min(1.0, score)

    def _generate_recommendations(self, indicators: Dict[str, Any]) -> List[str]:
        """改善推奨事項を生成"""
        recommendations = []

        if indicators["quality_level"] == "Rejected":
            recommendations.append("特徴量エンジニアリングの見直しが必要です")
            recommendations.append("より多くのデータの収集を検討してください")

        if indicators["cv_stability"] == "不安定":
            recommendations.append("交差検証の安定性向上のため、データの質を確認してください")

        if indicators["overfitting_risk"] == "高":
            recommendations.append("過学習対策として正則化の強化を検討してください")
            recommendations.append("訓練データ量の増加を推奨します")

        if not indicators["performance_consistency"]:
            recommendations.append(
                "モデルの一貫性向上のため、ハイパーパラメータの調整を行ってください"
            )

        if not recommendations:
            recommendations.append("現在のモデル品質は良好です")

        return recommendations

    def create_quality_report(self, results: List[Dict]) -> Dict[str, Any]:
        """
        品質レポートを作成

        Args:
            results: 分析結果リスト

        Returns:
            品質レポート
        """
        dashboard_data = self.generate_quality_dashboard_data(results)

        # 詳細分析
        detailed_analysis = []
        for result in results:
            if "test_metrics" in result:
                reliability = self.evaluate_model_reliability(result["test_metrics"])
                detailed_analysis.append(
                    {
                        "product_name": result.get("product_name", "Unknown"),
                        "quality_level": result.get("quality_level", "Rejected"),
                        "r2_score": result["test_metrics"].get("r2_score", 0.0),
                        "reliability_score": reliability["reliability_score"],
                        "recommendations": reliability["recommendations"],
                    }
                )

        quality_report = {
            "summary": dashboard_data,
            "detailed_analysis": detailed_analysis,
            "overall_assessment": self._generate_overall_assessment(dashboard_data),
            "improvement_priorities": self._identify_improvement_priorities(detailed_analysis),
        }

        return quality_report

    def _generate_overall_assessment(self, dashboard_data: Dict[str, Any]) -> str:
        """全体評価を生成"""
        success_rate = dashboard_data.get("success_rate", 0.0)

        if success_rate >= 0.8:
            return "優秀: システムは高い品質で安定して動作しています"
        elif success_rate >= 0.6:
            return "良好: 概ね満足できる品質ですが、一部改善の余地があります"
        elif success_rate >= 0.4:
            return "要改善: 品質向上のための対策が必要です"
        else:
            return "大幅改善必要: システム全体の見直しが必要です"

    def _identify_improvement_priorities(self, detailed_analysis: List[Dict]) -> List[str]:
        """改善優先度を特定"""
        priorities = []

        # Rejectedが多い場合
        rejected_count = sum(
            1 for item in detailed_analysis if item["quality_level"] == "Rejected"
        )
        if rejected_count > len(detailed_analysis) * 0.3:
            priorities.append("低品質モデルの大幅な改善")

        # 信頼性スコアが低い場合
        low_reliability = [item for item in detailed_analysis if item["reliability_score"] < 0.5]
        if len(low_reliability) > len(detailed_analysis) * 0.2:
            priorities.append("モデル信頼性の向上")

        # 共通の推奨事項
        all_recommendations = []
        for item in detailed_analysis:
            all_recommendations.extend(item["recommendations"])

        # 頻出する推奨事項を優先度として追加
        from collections import Counter

        common_recommendations = Counter(all_recommendations).most_common(3)
        for rec, count in common_recommendations:
            if count > len(detailed_analysis) * 0.2:
                priorities.append(f"共通課題: {rec}")

        return priorities[:5]  # 上位5つの優先度
