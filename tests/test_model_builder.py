import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

from src.demand_forecasting.core.model_builder import ModelBuilder
from src.demand_forecasting.utils.config import Config
from src.demand_forecasting.utils.exceptions import ModelBuildingError


class TestModelBuilder:
    """ModelBuilderクラスのテスト"""
    
    @pytest.fixture
    def sample_config(self):
        """テスト用設定"""
        return Config()
    
    @pytest.fixture
    def model_builder(self, sample_config):
        """ModelBuilderインスタンス"""
        return ModelBuilder(sample_config)
    
    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        n_samples = 100
        
        # 特徴量データ
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(5, 2, n_samples),
            'feature3': np.random.uniform(0, 10, n_samples),
            'feature4': np.random.exponential(2, n_samples)
        })
        
        # ターゲット（特徴量の線形結合 + ノイズ）
        y = pd.Series(
            2 * X['feature1'] + 3 * X['feature2'] + 0.5 * X['feature3'] + 
            np.random.normal(0, 0.5, n_samples)
        )
        
        return X, y
    
    def test_build_model(self, model_builder, sample_data):
        """モデル構築のテスト"""
        X, y = sample_data
        
        model = model_builder.build_model(X, y)
        
        # RandomForestRegressorが返されることを確認
        assert isinstance(model, RandomForestRegressor)
        
        # モデルが訓練されていることを確認
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == X.shape[1]
    
    def test_preprocess_features(self, model_builder, sample_data):
        """特徴量前処理のテスト"""
        X, _ = sample_data
        
        # 無限値やNaNを含むデータを作成
        X_with_issues = X.copy()
        X_with_issues.loc[0, 'feature1'] = np.inf
        X_with_issues.loc[1, 'feature2'] = -np.inf
        X_with_issues.loc[2, 'feature3'] = np.nan
        
        X_processed = model_builder._preprocess_features(X_with_issues)
        
        # 無限値・NaN値が処理されていることを確認
        assert not np.isinf(X_processed).any().any()
        assert not np.isnan(X_processed).any().any()
        
        # 数値列のみが残っていることを確認
        assert all(X_processed.dtypes.apply(lambda x: np.issubdtype(x, np.number)))
    
    def test_train_with_cv(self, model_builder, sample_data):
        """交差検証付き訓練のテスト"""
        X, y = sample_data
        
        results = model_builder.train_with_cv(X, y)
        
        # 必要なキーが結果に含まれていることを確認
        expected_keys = [
            'model', 'cv_scores', 'train_metrics', 'test_metrics', 
            'overfitting_score', 'feature_importance', 'feature_names'
        ]
        for key in expected_keys:
            assert key in results
        
        # モデルが正しく訓練されていることを確認
        assert isinstance(results['model'], RandomForestRegressor)
        
        # 交差検証結果が妥当であることを確認
        assert 'mean_score' in results['cv_scores']
        assert 'std_score' in results['cv_scores']
        
        # メトリクスが計算されていることを確認
        assert 'r2_score' in results['test_metrics']
        assert 'rmse' in results['test_metrics']
        assert 'mae' in results['test_metrics']
    
    def test_evaluate_model(self, model_builder, sample_data):
        """モデル評価のテスト"""
        X, y = sample_data
        
        # モデルを訓練
        model = model_builder.build_model(X, y)
        
        # 評価実行
        metrics = model_builder.evaluate_model(model, X, y)
        
        # 必要なメトリクスが含まれていることを確認
        expected_metrics = ['r2_score', 'rmse', 'mae', 'pred_mean', 'pred_std', 'n_samples']
        for metric in expected_metrics:
            assert metric in metrics
        
        # メトリクス値が妥当な範囲にあることを確認
        assert 0 <= metrics['r2_score'] <= 1  # R²は通常0-1の範囲
        assert metrics['rmse'] >= 0  # RMSEは非負
        assert metrics['mae'] >= 0   # MAEは非負
        assert metrics['n_samples'] == len(X)
    
    def test_get_feature_importance(self, model_builder, sample_data):
        """特徴量重要度取得のテスト"""
        X, y = sample_data
        
        # モデルを訓練
        model = model_builder.build_model(X, y)
        
        # 特徴量重要度を取得
        importance = model_builder.get_feature_importance(model, X.columns.tolist())
        
        # 重要度辞書が返されることを確認
        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
        
        # すべての特徴量の重要度が非負であることを確認
        for feature, imp in importance.items():
            assert imp >= 0
            assert feature in X.columns
        
        # 重要度の合計が1に近いことを確認（RandomForestの性質）
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 1e-6
    
    def test_detect_overfitting(self, model_builder):
        """過学習検出のテスト"""
        # 正常なケース（過学習なし）
        overfitting_score = model_builder.detect_overfitting(0.8, 0.75)
        assert 0 <= overfitting_score <= 1
        
        # 過学習のケース
        overfitting_score_high = model_builder.detect_overfitting(0.95, 0.6)
        assert overfitting_score_high > overfitting_score
        
        # 完全に同じスコア（理想的）
        overfitting_score_zero = model_builder.detect_overfitting(0.8, 0.8)
        assert overfitting_score_zero == 0
    
    def test_save_and_load_model(self, model_builder, sample_data):
        """モデル保存・読み込みのテスト"""
        X, y = sample_data
        
        # モデルを訓練
        model = model_builder.build_model(X, y)
        
        # 一時ファイルに保存
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.pkl"
            
            # 保存
            saved_path = model_builder.save_model(model, str(model_path))
            assert Path(saved_path).exists()
            
            # 読み込み
            loaded_model = model_builder.load_model(saved_path)
            
            # 読み込まれたモデルが同じ予測をすることを確認
            original_predictions = model.predict(model_builder._preprocess_features(X))
            loaded_predictions = loaded_model.predict(model_builder._preprocess_features(X))
            
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
    
    def test_predict(self, model_builder, sample_data):
        """予測のテスト"""
        X, y = sample_data
        
        # モデルを訓練
        model = model_builder.build_model(X, y)
        
        # 予測実行
        predictions = model_builder.predict(model, X)
        
        # 予測結果が妥当であることを確認
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(np.isfinite(predictions))  # 有限値であることを確認
    
    def test_get_model_summary(self, model_builder, sample_data):
        """モデルサマリー生成のテスト"""
        X, y = sample_data
        
        # 交差検証付き訓練を実行
        results = model_builder.train_with_cv(X, y)
        
        # サマリー生成
        summary = model_builder.get_model_summary(results)
        
        # 必要なキーが含まれていることを確認
        expected_keys = [
            'model_type', 'cv_mean_r2', 'cv_std_r2', 'test_r2', 
            'test_rmse', 'test_mae', 'overfitting_score', 
            'n_features', 'top_features'
        ]
        for key in expected_keys:
            assert key in summary
        
        # 値が妥当であることを確認
        assert summary['model_type'] == 'RandomForestRegressor'
        assert summary['n_features'] == len(X.columns)
        assert len(summary['top_features']) <= 5
    
    def test_perform_cross_validation(self, model_builder, sample_data):
        """交差検証実行のテスト"""
        X, y = sample_data
        
        # 特徴量前処理
        X_processed = model_builder._preprocess_features(X)
        
        # 交差検証実行
        cv_results = model_builder._perform_cross_validation(X_processed, y)
        
        # 結果が妥当であることを確認
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'scores' in cv_results
        assert 'folds' in cv_results
        
        # スコア配列の長さがfold数と一致することを確認
        assert len(cv_results['scores']) == cv_results['folds']