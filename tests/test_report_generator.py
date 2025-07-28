import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.demand_forecasting.reports.report_generator import ReportGenerator
from src.demand_forecasting.utils.config import Config
from src.demand_forecasting.utils.exceptions import ReportGenerationError


class TestReportGenerator:
    """ReportGeneratorクラスのテスト"""

    @pytest.fixture
    def sample_config(self):
        """テスト用設定"""
        return Config()

    @pytest.fixture
    def report_generator(self, sample_config):
        """ReportGeneratorインスタンス"""
        return ReportGenerator(sample_config)

    @pytest.fixture
    def sample_analysis_results(self):
        """テスト用分析結果"""
        return [
            {
                "product_name": "りんご",
                "category": "果物",
                "quality_level": "Premium",
                "implementation_readiness": "即座実行",
                "test_metrics": {"r2_score": 0.85, "rmse": 10.2, "mae": 8.1},
                "cv_scores": {"mean_score": 0.82, "std_score": 0.05},
                "overfitting_score": 0.03,
                "feature_importance": {"価格": 0.35, "曜日": 0.20, "気温": 0.15},
                "demand_results": {
                    "optimal_price": 120,
                    "current_price": 100,
                    "price_elasticity": -0.5,
                    "r2_score": 0.90,
                    "data_points": 150,
                },
            },
            {
                "product_name": "キャベツ",
                "category": "野菜",
                "quality_level": "Standard",
                "implementation_readiness": "慎重実行",
                "test_metrics": {"r2_score": 0.65, "rmse": 15.3, "mae": 12.4},
                "cv_scores": {"mean_score": 0.63, "std_score": 0.08},
                "overfitting_score": 0.08,
                "feature_importance": {"気温": 0.30, "価格": 0.25, "時間帯": 0.20},
                "demand_results": {
                    "optimal_price": 80,
                    "current_price": 75,
                    "price_elasticity": -0.3,
                    "r2_score": 0.75,
                    "data_points": 120,
                },
            },
        ]

    @pytest.fixture
    def sample_quality_report(self):
        """テスト用品質レポート"""
        return {
            "summary": {
                "total_products": 2,
                "success_rate": 1.0,
                "average_r2": 0.75,
                "quality_distribution": {"Premium": 1, "Standard": 1},
                "implementation_distribution": {"即座実行": 1, "慎重実行": 1},
                "category_success_rates": {"果物": 1.0, "野菜": 1.0},
            },
            "overall_assessment": "良好: 概ね満足できる品質ですが、一部改善の余地があります",
            "improvement_priorities": ["低品質モデルの改善", "特徴量エンジニアリングの見直し"],
        }

    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    def test_generate_markdown_report(
        self, report_generator, sample_analysis_results, sample_quality_report
    ):
        """Markdownレポート生成のテスト"""
        markdown_report = report_generator.generate_markdown_report(
            sample_analysis_results, sample_quality_report
        )

        # レポートが文字列として生成されていることを確認
        assert isinstance(markdown_report, str)
        assert len(markdown_report) > 0

        # 主要セクションが含まれていることを確認
        assert "# 生鮮食品需要予測・分析レポート" in markdown_report
        assert "## エグゼクティブサマリー" in markdown_report
        assert "## 品質評価概要" in markdown_report
        assert "## 個別商品分析結果" in markdown_report
        assert "## 改善提案" in markdown_report
        assert "## 段階的実装計画" in markdown_report
        assert "## 付録" in markdown_report

        # 具体的なデータが含まれていることを確認
        assert "りんご" in markdown_report
        assert "キャベツ" in markdown_report
        assert "Premium" in markdown_report
        assert "Standard" in markdown_report

    def test_save_markdown_report(self, report_generator, temp_dir):
        """Markdownレポート保存のテスト"""
        test_report = "# テストレポート\n\nこれはテスト用のMarkdownレポートです。"
        save_path = Path(temp_dir) / "test_report.md"

        result_path = report_generator.save_markdown_report(test_report, str(save_path))

        # ファイルが作成されていることを確認
        assert os.path.exists(result_path)
        assert Path(result_path).suffix == ".md"

        # ファイル内容が正しいことを確認
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == test_report

        # 戻り値が正しいパスであることを確認
        assert result_path == str(save_path)

    def test_save_markdown_report_default_path(self, report_generator):
        """Markdownレポート（デフォルトパス）保存のテスト"""
        test_report = "# テストレポート（デフォルトパス）"

        result_path = report_generator.save_markdown_report(test_report)

        # ファイルが作成されていることを確認
        assert os.path.exists(result_path)
        assert "demand_forecasting_report_" in result_path
        assert ".md" in result_path

        # 後処理：テストファイルを削除
        if os.path.exists(result_path):
            os.remove(result_path)
            # 空のディレクトリも削除
            parent_dir = Path(result_path).parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                parent_dir.rmdir()

    def test_generate_csv_reports(self, report_generator, sample_analysis_results, temp_dir):
        """CSVレポート生成のテスト"""
        saved_files = report_generator.generate_csv_reports(sample_analysis_results, temp_dir)

        # 複数のCSVファイルが作成されていることを確認
        assert len(saved_files) >= 2  # 最低でもperformanceとimportanceファイル

        # 全てのファイルが存在することを確認
        for file_path in saved_files:
            assert os.path.exists(file_path)
            assert Path(file_path).suffix == ".csv"

        # 期待されるファイル名が含まれていることを確認
        file_names = [Path(path).name for path in saved_files]
        assert "model_performance.csv" in file_names
        assert "feature_importance.csv" in file_names
        assert "demand_analysis.csv" in file_names

    def test_csv_report_content(self, report_generator, sample_analysis_results, temp_dir):
        """CSVレポート内容のテスト"""
        saved_files = report_generator.generate_csv_reports(sample_analysis_results, temp_dir)

        # モデル性能CSVの内容を確認
        performance_file = next(f for f in saved_files if "model_performance.csv" in f)
        performance_df = pd.read_csv(performance_file)

        # 期待される列が存在することを確認
        expected_columns = ["商品名", "品質レベル", "R²スコア", "RMSE", "MAE"]
        for col in expected_columns:
            assert col in performance_df.columns

        # データ数が正しいことを確認
        assert len(performance_df) == 2  # 2商品

        # 特徴量重要度CSVの内容を確認
        importance_file = next(f for f in saved_files if "feature_importance.csv" in f)
        importance_df = pd.read_csv(importance_file)

        # 期待される列が存在することを確認
        expected_columns = ["商品名", "特徴量", "重要度"]
        for col in expected_columns:
            assert col in importance_df.columns

        # データが存在することを確認
        assert len(importance_df) > 0

    def test_generate_csv_reports_default_path(self, report_generator, sample_analysis_results):
        """CSVレポート（デフォルトパス）生成のテスト"""
        saved_files = report_generator.generate_csv_reports(sample_analysis_results)

        # ファイルが作成されていることを確認
        assert len(saved_files) > 0

        # 全てのファイルが存在することを確認
        for file_path in saved_files:
            assert os.path.exists(file_path)
            assert "reports" in file_path  # デフォルトディレクトリ

        # 後処理：テストファイルを削除
        for file_path in saved_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        # 空のディレクトリも削除
        reports_dir = Path("reports")
        if reports_dir.exists() and not any(reports_dir.iterdir()):
            reports_dir.rmdir()

    def test_header_generation(self, report_generator):
        """ヘッダー生成のテスト"""
        header = report_generator._generate_header()

        # ヘッダーの構造を確認
        assert "# 生鮮食品需要予測・分析レポート" in header
        assert "**作成日時**:" in header
        assert "**システム**:" in header
        assert "生鮮食品需要予測・分析システム" in header

    def test_executive_summary_generation(self, report_generator, sample_quality_report):
        """エグゼクティブサマリー生成のテスト"""
        summary = report_generator._generate_executive_summary(sample_quality_report)

        # サマリーの構造を確認
        assert "## エグゼクティブサマリー" in summary
        assert "### 主要な成果" in summary
        assert "### ビジネスインパクト" in summary

        # 具体的なデータが含まれていることを確認
        assert "分析対象商品数**: 2商品" in summary
        assert "成功率**: 100.0%" in summary
        assert "平均R²スコア**: 0.750" in summary

    def test_empty_results_handling(self, report_generator):
        """空の結果に対する処理のテスト"""
        empty_results = []
        empty_quality_report = {
            "summary": {"total_products": 0, "success_rate": 0.0, "average_r2": 0.0},
            "overall_assessment": "データなし",
            "improvement_priorities": [],
        }

        # エラーが発生しないことを確認
        markdown_report = report_generator.generate_markdown_report(
            empty_results, empty_quality_report
        )
        assert isinstance(markdown_report, str)
        assert "分析結果がありません" in markdown_report

        # CSV生成でもエラーが発生しないことを確認
        saved_files = report_generator.generate_csv_reports(empty_results)
        # 空の結果でも最低限のファイルは作成される
        assert isinstance(saved_files, list)

    def test_invalid_data_handling(self, report_generator):
        """不正なデータに対する処理のテスト"""
        # 不完全な分析結果
        invalid_results = [
            {
                "product_name": "テスト商品"
                # 必要なデータが不足
            }
        ]

        invalid_quality_report = {}

        # エラーが適切に処理されることを確認
        try:
            markdown_report = report_generator.generate_markdown_report(
                invalid_results, invalid_quality_report
            )
            # エラーが発生しなくても、適切にデフォルト値で処理されることを確認
            assert isinstance(markdown_report, str)
        except ReportGenerationError:
            # ReportGenerationErrorが発生することも許容
            pass
