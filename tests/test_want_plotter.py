import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.demand_forecasting.visualization.want_plotter import WantPlotter
from src.demand_forecasting.utils.config import Config
from src.demand_forecasting.utils.exceptions import VisualizationError


class TestWantPlotter:
    """WantPlotterクラスのテスト"""
    
    @pytest.fixture
    def sample_config(self):
        """テスト用設定"""
        return Config()
    
    @pytest.fixture
    def want_plotter(self, sample_config):
        """WantPlotterインスタンス"""
        return WantPlotter(sample_config)
    
    @pytest.fixture
    def sample_demand_results(self):
        """テスト用需要曲線分析結果"""
        # サンプルの需要関数
        def demand_func(price):
            return max(0, 100 - 0.5 * price)
        
        price_demand_data = pd.DataFrame({
            'price': [50, 60, 70, 80, 90, 100],
            'quantity': [75, 70, 65, 60, 55, 50]
        })
        
        return {
            'product_name': 'りんご',
            'price_demand_data': price_demand_data,
            'demand_curve_function': demand_func,
            'optimal_price': 75,
            'current_price': 80,
            'price_range': [40, 120],
            'price_elasticity': -0.5,
            'r2_score': 0.95
        }
    
    @pytest.fixture
    def sample_feature_importance(self):
        """テスト用特徴量重要度"""
        return {
            '価格': 0.35,
            '曜日': 0.20,
            '気温': 0.15,
            '時間帯': 0.12,
            '週末フラグ': 0.10,
            '月': 0.08,
            '祝日フラグ': 0.06,
            '雨量': 0.05,
            '湿度': 0.04,
            '風速': 0.03,
            '商品カテゴリ': 0.02
        }
    
    @pytest.fixture
    def sample_quality_data(self):
        """テスト用品質データ"""
        return {
            'total_products': 10,
            'quality_distribution': {
                'Premium': 3,
                'Standard': 4,
                'Basic': 2,
                'Rejected': 1
            },
            'implementation_distribution': {
                '即座実行': 3,
                '慎重実行': 4,
                '要考慮': 2,
                '改善必要': 1
            },
            'success_rate': 0.7,
            'average_r2': 0.65,
            'r2_std': 0.15,
            'category_success_rates': {
                '果物': 0.8,
                '野菜': 0.6,
                '肉類': 0.7,
                '魚類': 0.5
            }
        }
    
    @pytest.fixture
    def temp_dir(self):
        """一時ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    def test_setup_style(self, want_plotter):
        """スタイル設定のテスト"""
        # Wantカラーパレットが設定されていることを確認
        assert hasattr(want_plotter, 'want_colors')
        assert 'primary' in want_plotter.want_colors
        assert 'secondary' in want_plotter.want_colors
        assert 'accent' in want_plotter.want_colors
        
        # カラーコードが正しいことを確認
        assert want_plotter.want_colors['primary'] == '#FF6B35'
        assert want_plotter.want_colors['secondary'] == '#004E89'
    
    def test_create_demand_curve_plot(self, want_plotter, sample_demand_results, temp_dir):
        """需要曲線プロット作成のテスト"""
        save_path = Path(temp_dir) / "test_demand_curve.png"
        
        result_path = want_plotter.create_demand_curve_plot(
            sample_demand_results, str(save_path)
        )
        
        # ファイルが作成されていることを確認
        assert os.path.exists(result_path)
        assert Path(result_path).suffix == '.png'
        
        # 戻り値が正しいパスであることを確認
        assert result_path == str(save_path)
    
    def test_create_demand_curve_plot_default_path(self, want_plotter, sample_demand_results):
        """需要曲線プロット（デフォルトパス）作成のテスト"""
        result_path = want_plotter.create_demand_curve_plot(sample_demand_results)
        
        # ファイルが作成されていることを確認
        assert os.path.exists(result_path)
        assert 'demand_curve_' in result_path
        assert '.png' in result_path
        
        # 後処理：テストファイルを削除
        if os.path.exists(result_path):
            os.remove(result_path)
    
    def test_create_feature_importance_plot(self, want_plotter, sample_feature_importance, temp_dir):
        """特徴量重要度プロット作成のテスト"""
        save_path = Path(temp_dir) / "test_feature_importance.png"
        
        result_path = want_plotter.create_feature_importance_plot(
            sample_feature_importance, "りんご", str(save_path)
        )
        
        # ファイルが作成されていることを確認
        assert os.path.exists(result_path)
        assert Path(result_path).suffix == '.png'
        
        # 戻り値が正しいパスであることを確認
        assert result_path == str(save_path)
    
    def test_create_feature_importance_plot_default_path(self, want_plotter, sample_feature_importance):
        """特徴量重要度プロット（デフォルトパス）作成のテスト"""
        result_path = want_plotter.create_feature_importance_plot(
            sample_feature_importance, "りんご"
        )
        
        # ファイルが作成されていることを確認
        assert os.path.exists(result_path)
        assert 'feature_importance_' in result_path
        assert '.png' in result_path
        
        # 後処理：テストファイルを削除
        if os.path.exists(result_path):
            os.remove(result_path)
    
    def test_create_quality_dashboard(self, want_plotter, sample_quality_data, temp_dir):
        """品質ダッシュボード作成のテスト"""
        save_path = Path(temp_dir) / "test_quality_dashboard.png"
        
        result_path = want_plotter.create_quality_dashboard(
            sample_quality_data, str(save_path)
        )
        
        # ファイルが作成されていることを確認
        assert os.path.exists(result_path)
        assert Path(result_path).suffix == '.png'
        
        # 戻り値が正しいパスであることを確認
        assert result_path == str(save_path)
    
    def test_create_quality_dashboard_default_path(self, want_plotter, sample_quality_data):
        """品質ダッシュボード（デフォルトパス）作成のテスト"""
        result_path = want_plotter.create_quality_dashboard(sample_quality_data)
        
        # ファイルが作成されていることを確認
        assert os.path.exists(result_path)
        assert 'quality_dashboard.png' in result_path
        
        # 後処理：テストファイルを削除
        if os.path.exists(result_path):
            os.remove(result_path)
    
    def test_create_comprehensive_report_plots(self, want_plotter, sample_demand_results, 
                                             sample_feature_importance, temp_dir):
        """包括的レポートプロット作成のテスト"""
        analysis_results = [
            {
                'product_name': 'りんご',
                'demand_results': sample_demand_results,
                'feature_importance': sample_feature_importance
            },
            {
                'product_name': 'キャベツ',
                'feature_importance': sample_feature_importance
            }
        ]
        
        saved_files = want_plotter.create_comprehensive_report_plots(
            analysis_results, temp_dir
        )
        
        # 複数のファイルが作成されていることを確認
        assert len(saved_files) > 0
        
        # 全てのファイルが存在することを確認
        for file_path in saved_files:
            assert os.path.exists(file_path)
            assert Path(file_path).suffix == '.png'
    
    def test_empty_data_handling(self, want_plotter, temp_dir):
        """空データに対する処理のテスト"""
        save_path = Path(temp_dir) / "test_empty_dashboard.png"
        
        # 空の品質データでダッシュボードを作成
        empty_quality_data = {
            'quality_distribution': {},
            'implementation_distribution': {},
            'category_success_rates': {},
            'average_r2': 0,
            'r2_std': 0
        }
        
        result_path = want_plotter.create_quality_dashboard(
            empty_quality_data, str(save_path)
        )
        
        # エラーが発生せずファイルが作成されることを確認
        assert os.path.exists(result_path)
    
    def test_invalid_demand_results_handling(self, want_plotter):
        """不正な需要曲線データに対する処理のテスト"""
        invalid_demand_results = {
            'product_name': 'テスト商品'
            # 必要なデータが不足
        }
        
        # エラーが適切に発生することを確認
        with pytest.raises(VisualizationError):
            want_plotter.create_demand_curve_plot(invalid_demand_results)
    
    def test_plot_methods_with_matplotlib_backend(self, want_plotter, sample_demand_results):
        """matplotlibバックエンドでのプロット処理のテスト"""
        import matplotlib
        matplotlib.use('Agg')  # GUI不要のバックエンドに設定
        
        # エラーが発生しないことを確認
        result_path = want_plotter.create_demand_curve_plot(sample_demand_results)
        assert os.path.exists(result_path)
        
        # 後処理：テストファイルを削除
        if os.path.exists(result_path):
            os.remove(result_path)