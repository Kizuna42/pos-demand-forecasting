import pytest
import pandas as pd
import numpy as np

from src.demand_forecasting.utils.quality_evaluator import QualityEvaluator
from src.demand_forecasting.utils.config import Config


class TestQualityEvaluator:
    """QualityEvaluatorクラスのテスト"""

    @pytest.fixture
    def sample_config(self):
        """テスト用設定"""
        return Config()

    @pytest.fixture
    def quality_evaluator(self, sample_config):
        """QualityEvaluatorインスタンス"""
        return QualityEvaluator(sample_config)

    @pytest.fixture
    def sample_analysis_results(self):
        """テスト用分析結果"""
        return [
            {
                "product_name": "りんご",
                "category": "果物",
                "quality_level": "Premium",
                "implementation_readiness": "即座実行",
                "test_metrics": {
                    "r2_score": 0.85,
                    "rmse": 10.2,
                    "mae": 8.1,
                    "cv_mean_r2": 0.82,
                    "cv_std_r2": 0.05,
                },
                "overfitting_score": 0.03,
            },
            {
                "product_name": "キャベツ",
                "category": "野菜",
                "quality_level": "Standard",
                "implementation_readiness": "慎重実行",
                "test_metrics": {"r2_score": 0.65, "rmse": 15.3, "mae": 12.4},
                "overfitting_score": 0.08,
            },
            {
                "product_name": "牛肉",
                "category": "肉類",
                "quality_level": "Basic",
                "implementation_readiness": "要考慮",
                "test_metrics": {"r2_score": 0.35, "rmse": 25.1, "mae": 20.2},
                "overfitting_score": 0.15,
            },
        ]

    def test_evaluate_quality_level(self, quality_evaluator):
        """品質レベル評価のテスト"""
        # Premium品質
        assert quality_evaluator.evaluate_quality_level(0.8) == "Premium"
        assert quality_evaluator.evaluate_quality_level(0.75) == "Premium"

        # Standard品質
        assert quality_evaluator.evaluate_quality_level(0.65) == "Standard"
        assert quality_evaluator.evaluate_quality_level(0.55) == "Standard"

        # Basic品質
        assert quality_evaluator.evaluate_quality_level(0.45) == "Basic"
        assert quality_evaluator.evaluate_quality_level(0.35) == "Basic"

        # Rejected品質
        assert quality_evaluator.evaluate_quality_level(0.25) == "Rejected"
        assert quality_evaluator.evaluate_quality_level(0.1) == "Rejected"

    def test_assess_implementation_readiness(self, quality_evaluator):
        """実用化準備状況評価のテスト"""
        # 即座実行（Premium + 過学習なし）
        metrics = {"r2_score": 0.8, "overfitting_score": 0.05}
        assert quality_evaluator.assess_implementation_readiness(metrics) == "即座実行"

        # 慎重実行（Standard + 過学習なし）
        metrics = {"r2_score": 0.6, "overfitting_score": 0.05}
        assert quality_evaluator.assess_implementation_readiness(metrics) == "慎重実行"

        # 要考慮（Basic品質）
        metrics = {"r2_score": 0.4, "overfitting_score": 0.05}
        assert quality_evaluator.assess_implementation_readiness(metrics) == "要考慮"

        # 改善必要（Rejected品質）
        metrics = {"r2_score": 0.2, "overfitting_score": 0.05}
        assert quality_evaluator.assess_implementation_readiness(metrics) == "改善必要"

        # 改善必要（過学習あり）
        metrics = {"r2_score": 0.8, "overfitting_score": 0.15}
        assert quality_evaluator.assess_implementation_readiness(metrics) == "改善必要"

    def test_calculate_category_success_rate(self, quality_evaluator, sample_analysis_results):
        """カテゴリ別成功率算出のテスト"""
        success_rates = quality_evaluator.calculate_category_success_rate(sample_analysis_results)

        # 各カテゴリの成功率を確認
        assert "果物" in success_rates
        assert "野菜" in success_rates
        assert "肉類" in success_rates

        # 果物：Premium（成功）
        assert success_rates["果物"] == 1.0
        # 野菜：Standard（成功）
        assert success_rates["野菜"] == 1.0
        # 肉類：Basic（失敗）
        assert success_rates["肉類"] == 0.0

    def test_generate_quality_dashboard_data(self, quality_evaluator, sample_analysis_results):
        """品質ダッシュボードデータ生成のテスト"""
        dashboard_data = quality_evaluator.generate_quality_dashboard_data(sample_analysis_results)

        # 必要なキーが存在することを確認
        required_keys = [
            "total_products",
            "quality_distribution",
            "implementation_distribution",
            "success_rate",
            "average_r2",
            "r2_std",
            "category_success_rates",
        ]
        for key in required_keys:
            assert key in dashboard_data

        # 商品数が正しいことを確認
        assert dashboard_data["total_products"] == 3

        # 品質分布が正しいことを確認
        quality_dist = dashboard_data["quality_distribution"]
        assert quality_dist["Premium"] == 1
        assert quality_dist["Standard"] == 1
        assert quality_dist["Basic"] == 1

        # 成功率が正しいことを確認（Premium + Standard = 2/3）
        assert abs(dashboard_data["success_rate"] - 2 / 3) < 0.01

    def test_evaluate_model_reliability(self, quality_evaluator):
        """モデル信頼性評価のテスト"""
        # 高信頼性モデル
        metrics = {
            "r2_score": 0.85,
            "cv_mean_r2": 0.82,
            "cv_std_r2": 0.05,
            "overfitting_score": 0.03,
        }
        reliability = quality_evaluator.evaluate_model_reliability(metrics)

        # 信頼性スコアが高いことを確認
        assert reliability["reliability_score"] > 0.8
        assert reliability["indicators"]["quality_level"] == "Premium"
        assert reliability["indicators"]["cv_stability"] == "安定"
        assert reliability["indicators"]["overfitting_risk"] == "低"

        # 低信頼性モデル
        metrics = {
            "r2_score": 0.25,
            "cv_mean_r2": 0.15,
            "cv_std_r2": 0.25,
            "overfitting_score": 0.25,
        }
        reliability = quality_evaluator.evaluate_model_reliability(metrics)

        # 信頼性スコアが低いことを確認
        assert reliability["reliability_score"] < 0.3
        assert reliability["indicators"]["quality_level"] == "Rejected"
        assert reliability["indicators"]["cv_stability"] == "不安定"
        assert reliability["indicators"]["overfitting_risk"] == "高"

    def test_create_quality_report(self, quality_evaluator, sample_analysis_results):
        """品質レポート作成のテスト"""
        quality_report = quality_evaluator.create_quality_report(sample_analysis_results)

        # レポート構造を確認
        required_keys = [
            "summary",
            "detailed_analysis",
            "overall_assessment",
            "improvement_priorities",
        ]
        for key in required_keys:
            assert key in quality_report

        # 詳細分析データを確認
        detailed_analysis = quality_report["detailed_analysis"]
        assert len(detailed_analysis) == 3

        # 各商品の分析結果を確認
        for analysis in detailed_analysis:
            assert "product_name" in analysis
            assert "quality_level" in analysis
            assert "r2_score" in analysis
            assert "reliability_score" in analysis
            assert "recommendations" in analysis

        # 全体評価が生成されていることを確認
        assert isinstance(quality_report["overall_assessment"], str)
        assert len(quality_report["overall_assessment"]) > 0

        # 改善優先度が生成されていることを確認
        assert isinstance(quality_report["improvement_priorities"], list)

    def test_empty_results_handling(self, quality_evaluator):
        """空の結果に対する処理のテスト"""
        empty_results = []

        # 空の結果でもエラーが発生しないことを確認
        dashboard_data = quality_evaluator.generate_quality_dashboard_data(empty_results)
        assert dashboard_data == {}

        success_rates = quality_evaluator.calculate_category_success_rate(empty_results)
        assert success_rates == {}

        quality_report = quality_evaluator.create_quality_report(empty_results)
        assert "summary" in quality_report
        assert quality_report["summary"] == {}
