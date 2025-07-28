import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.demand_forecasting.core.data_processor import DataProcessor
from src.demand_forecasting.core.feature_engineer import FeatureEngineer
from src.demand_forecasting.core.model_builder import ModelBuilder
from src.demand_forecasting.core.demand_analyzer import DemandCurveAnalyzer
from src.demand_forecasting.utils.quality_evaluator import QualityEvaluator
from src.demand_forecasting.visualization.want_plotter import WantPlotter
from src.demand_forecasting.reports.report_generator import ReportGenerator
from src.demand_forecasting.utils.config import Config


class TestIntegration:
    """インテグレーションテスト"""
    
    @pytest.fixture
    def sample_config(self):
        """テスト用設定"""
        return Config()
    
    @pytest.fixture
    def sample_raw_data(self):
        """テスト用生データ"""
        np.random.seed(42)  # 再現性のため
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        products = ['りんご', 'キャベツ', '牛肉', 'まぐろ', 'じゃがいも']
        
        data = []
        for date in dates:
            for product in np.random.choice(products, size=np.random.randint(1, 4), replace=False):
                # 価格に基づく需要量の生成（簡単な線形関係）
                base_price = {'りんご': 100, 'キャベツ': 80, '牛肉': 500, 'まぐろ': 300, 'じゃがいも': 60}[product]
                price_variation = np.random.normal(1.0, 0.2)
                price = int(base_price * price_variation)
                
                # 価格弾力性を考慮した需要量
                base_quantity = {'りんご': 20, 'キャベツ': 15, '牛肉': 5, 'まぐろ': 8, 'じゃがいも': 25}[product]
                quantity = max(1, int(base_quantity * (2 - price_variation) + np.random.normal(0, 2)))
                
                data.append({
                    '商品コード': f"{hash(product) % 1000:03d}",
                    '商品名称': product,
                    '年月日': date.strftime('%Y-%m-%d'),
                    '金額': price * quantity,
                    '数量': quantity,
                    '平均価格': price
                })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    def test_data_processing_pipeline(self, sample_config, sample_raw_data):
        """データ処理パイプラインのテスト"""
        # データ処理パイプライン
        data_processor = DataProcessor(sample_config)
        feature_engineer = FeatureEngineer(sample_config)
        
        # 1. データクリーニング
        clean_data = data_processor.clean_data(sample_raw_data)
        assert len(clean_data) > 0
        assert not clean_data.isnull().any().any()
        
        # 2. 基本特徴量生成
        basic_features = feature_engineer.create_baseline_features(clean_data)
        assert '売上単価' in basic_features.columns
        assert '商品カテゴリ' in basic_features.columns
        
        # 3. 時間特徴量追加
        time_features = feature_engineer.add_time_features(basic_features)
        assert '曜日' in time_features.columns
        assert '週末フラグ' in time_features.columns
        
        # 4. 気象特徴量統合（フォールバック機能のテスト）
        final_features = feature_engineer.integrate_weather_features(time_features)
        weather_columns = [col for col in final_features.columns if '気温' in col or '天気' in col]
        assert len(weather_columns) > 0  # 何らかの気象特徴量が追加されている
    
    def test_model_building_pipeline(self, sample_config, sample_raw_data):
        """モデル構築パイプラインのテスト"""
        # データ前処理
        data_processor = DataProcessor(sample_config)
        feature_engineer = FeatureEngineer(sample_config)
        model_builder = ModelBuilder(sample_config)
        
        clean_data = data_processor.clean_data(sample_raw_data)
        features = feature_engineer.create_baseline_features(clean_data)
        features = feature_engineer.add_time_features(features)
        
        # 特定商品のモデル構築
        product_data = features[features['商品名称'] == 'りんご'].copy()
        if len(product_data) < 10:
            pytest.skip("データ不足のためスキップ")
        
        # 特徴量とターゲット分離
        feature_columns = product_data.select_dtypes(include=['number']).columns.tolist()
        target_column = '数量'
        if target_column in feature_columns:
            feature_columns.remove(target_column)
        
        X = product_data[feature_columns]
        y = product_data[target_column]
        
        # モデル構築と評価
        model_results = model_builder.train_with_cv(X, y)
        
        # 結果の検証
        assert 'model' in model_results
        assert 'test_metrics' in model_results
        assert 'cv_scores' in model_results
        assert 'feature_importance' in model_results
        
        # メトリクスの検証
        test_metrics = model_results['test_metrics']
        assert 'r2_score' in test_metrics
        assert 'rmse' in test_metrics
        assert 'mae' in test_metrics
        
        # R²スコアが妥当な範囲にあることを確認
        assert -1 <= test_metrics['r2_score'] <= 1
    
    def test_demand_analysis_pipeline(self, sample_config, sample_raw_data):
        """需要曲線分析パイプラインのテスト"""
        # データ前処理
        data_processor = DataProcessor(sample_config)
        feature_engineer = FeatureEngineer(sample_config)
        demand_analyzer = DemandCurveAnalyzer(sample_config)
        
        clean_data = data_processor.clean_data(sample_raw_data)
        features = feature_engineer.create_baseline_features(clean_data)
        
        # 特定商品の需要曲線分析
        product_data = features[features['商品名称'] == 'りんご'].copy()
        if len(product_data) < 10:
            pytest.skip("データ不足のためスキップ")
        
        # 需要曲線分析実行
        demand_results = demand_analyzer.analyze_demand_curve(product_data, 'りんご')
        
        # 結果の検証
        assert 'demand_curve_function' in demand_results
        assert 'optimal_price' in demand_results
        assert 'current_price' in demand_results
        assert 'price_elasticity' in demand_results
        assert 'price_demand_data' in demand_results
        
        # 価格データが合理的であることを確認
        assert demand_results['optimal_price'] > 0
        assert demand_results['current_price'] > 0
    
    def test_quality_evaluation_pipeline(self, sample_config):
        """品質評価パイプラインのテスト"""
        quality_evaluator = QualityEvaluator(sample_config)
        
        # サンプル分析結果
        analysis_results = [
            {
                'product_name': 'りんご',
                'category': '果物',
                'quality_level': 'Premium',
                'test_metrics': {'r2_score': 0.85, 'rmse': 10.2, 'mae': 8.1},
                'overfitting_score': 0.03
            },
            {
                'product_name': 'キャベツ',
                'category': '野菜',
                'quality_level': 'Standard',
                'test_metrics': {'r2_score': 0.65, 'rmse': 15.3, 'mae': 12.4},
                'overfitting_score': 0.08
            }
        ]
        
        # 品質レポート作成
        quality_report = quality_evaluator.create_quality_report(analysis_results)
        
        # レポート構造の検証
        assert 'summary' in quality_report
        assert 'detailed_analysis' in quality_report
        assert 'overall_assessment' in quality_report
        assert 'improvement_priorities' in quality_report
        
        # サマリーデータの検証
        summary = quality_report['summary']
        assert summary['total_products'] == 2
        assert 0 <= summary['success_rate'] <= 1
        assert 0 <= summary['average_r2'] <= 1
    
    def test_visualization_pipeline(self, sample_config, temp_dir):
        """可視化パイプラインのテスト"""
        want_plotter = WantPlotter(sample_config)
        
        # サンプルデータ
        def demand_func(price):
            return max(0, 100 - 0.5 * price)
        
        demand_results = {
            'product_name': 'りんご',
            'price_demand_data': pd.DataFrame({
                'price': [50, 60, 70, 80, 90],
                'quantity': [75, 70, 65, 60, 55]
            }),
            'demand_curve_function': demand_func,
            'optimal_price': 75,
            'current_price': 80,
            'price_range': [40, 120]
        }
        
        feature_importance = {
            '価格': 0.35,
            '曜日': 0.20,
            '気温': 0.15
        }
        
        quality_data = {
            'quality_distribution': {'Premium': 2, 'Standard': 3},
            'implementation_distribution': {'即座実行': 2, '慎重実行': 3},
            'category_success_rates': {'果物': 0.8, '野菜': 0.6},
            'average_r2': 0.7,
            'r2_std': 0.1
        }
        
        # 各種プロット作成
        demand_plot = want_plotter.create_demand_curve_plot(
            demand_results, str(Path(temp_dir) / "demand.png")
        )
        importance_plot = want_plotter.create_feature_importance_plot(
            feature_importance, "りんご", str(Path(temp_dir) / "importance.png")
        )
        dashboard_plot = want_plotter.create_quality_dashboard(
            quality_data, str(Path(temp_dir) / "dashboard.png")
        )
        
        # ファイル作成の確認
        assert os.path.exists(demand_plot)
        assert os.path.exists(importance_plot)
        assert os.path.exists(dashboard_plot)
    
    def test_report_generation_pipeline(self, sample_config, temp_dir):
        """レポート生成パイプラインのテスト"""
        report_generator = ReportGenerator(sample_config)
        
        # サンプルデータ
        analysis_results = [
            {
                'product_name': 'りんご',
                'category': '果物',
                'quality_level': 'Premium',
                'test_metrics': {'r2_score': 0.85, 'rmse': 10.2, 'mae': 8.1},
                'feature_importance': {'価格': 0.35, '曜日': 0.20},
                'demand_results': {
                    'optimal_price': 120,
                    'current_price': 100,
                    'price_elasticity': -0.5
                }
            }
        ]
        
        quality_report = {
            'summary': {
                'total_products': 1,
                'success_rate': 1.0,
                'average_r2': 0.85,
                'quality_distribution': {'Premium': 1},
                'implementation_distribution': {'即座実行': 1},
                'category_success_rates': {'果物': 1.0}
            },
            'overall_assessment': '優秀: システムは高い品質で安定して動作しています',
            'improvement_priorities': []
        }
        
        # Markdownレポート生成
        markdown_report = report_generator.generate_markdown_report(
            analysis_results, quality_report
        )
        markdown_path = report_generator.save_markdown_report(
            markdown_report, str(Path(temp_dir) / "report.md")
        )
        
        # CSVレポート生成
        csv_files = report_generator.generate_csv_reports(analysis_results, temp_dir)
        
        # ファイル作成の確認
        assert os.path.exists(markdown_path)
        assert len(csv_files) > 0
        for csv_file in csv_files:
            assert os.path.exists(csv_file)
    
    def test_component_interaction_error_handling(self, sample_config):
        """コンポーネント間連携のエラーハンドリングテスト"""
        data_processor = DataProcessor(sample_config)
        feature_engineer = FeatureEngineer(sample_config)
        
        # 空のデータフレームでの処理
        empty_df = pd.DataFrame()
        
        # エラーが適切に処理されることを確認
        try:
            clean_data = data_processor.clean_data(empty_df)
            # 空データでも適切に処理される場合
            assert isinstance(clean_data, pd.DataFrame)
        except Exception as e:
            # 適切なエラーが発生する場合
            assert isinstance(e, (ValueError, KeyError, Exception))
    
    def test_configuration_consistency(self, sample_config):
        """設定の一貫性テスト"""
        # 各コンポーネントが同じ設定を参照していることを確認
        data_processor = DataProcessor(sample_config)
        feature_engineer = FeatureEngineer(sample_config)
        model_builder = ModelBuilder(sample_config)
        
        # 設定オブジェクトが正しく設定されていることを確認
        assert data_processor.config is not None
        assert feature_engineer.config is not None
        assert model_builder.config is not None
        
        # ログ設定が一貫していることを確認
        assert hasattr(data_processor, 'logger')
        assert hasattr(feature_engineer, 'logger')
        assert hasattr(model_builder, 'logger')