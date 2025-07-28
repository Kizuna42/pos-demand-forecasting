import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.demand_forecasting.core.feature_engineer import FeatureEngineer
from src.demand_forecasting.utils.config import Config
from src.demand_forecasting.utils.exceptions import FeatureEngineeringError


class TestFeatureEngineer:
    """FeatureEngineerクラスのテスト"""
    
    @pytest.fixture
    def sample_config(self):
        """テスト用設定"""
        return Config()
    
    @pytest.fixture
    def feature_engineer(self, sample_config):
        """FeatureEngineerインスタンス"""
        return FeatureEngineer(sample_config)
    
    @pytest.fixture
    def sample_dataframe(self):
        """テスト用サンプルデータ"""
        data = {
            '商品コード': ['001', '002', '003', '004', '005'],
            '商品名称': ['りんご', 'キャベツ', '牛肉', 'まぐろ', 'バナナ'],
            '年月日': ['2024-01-15', '2024-02-20', '2024-06-10', '2024-09-05', '2024-12-25'],
            '金額': [300, 150, 800, 500, 200],
            '数量': [3, 1, 2, 1, 2],
            '平均価格': [100, 150, 400, 500, 100],
            '商品カテゴリ': ['果物', '野菜', '肉類', '魚類', '果物'],
            '月': [1, 2, 6, 9, 12],
            '曜日': [0, 1, 2, 3, 4],
            '週末フラグ': [0, 0, 0, 0, 0]
        }
        df = pd.DataFrame(data)
        df['年月日'] = pd.to_datetime(df['年月日'])
        return df
    
    def test_create_baseline_features(self, feature_engineer, sample_dataframe):
        """ベースライン特徴量作成のテスト"""
        result = feature_engineer.create_baseline_features(sample_dataframe)
        
        # 価格関連特徴量が作成されていることを確認
        assert '価格帯' in result.columns
        assert '価格_log' in result.columns
        
        # 売上関連特徴量が作成されていることを確認
        assert '売上ボリューム' in result.columns
        assert '数量_log' in result.columns
        
        # カテゴリダミー変数が作成されていることを確認
        category_dummy_cols = [col for col in result.columns if col.startswith('カテゴリ_')]
        assert len(category_dummy_cols) > 0
        
        # 価格_logの計算が正しいことを確認
        assert result.loc[0, '価格_log'] == np.log1p(100)
    
    def test_add_time_features(self, feature_engineer, sample_dataframe):
        """時間特徴量追加のテスト"""
        result = feature_engineer.add_time_features(sample_dataframe)
        
        # 基本的な時間特徴量が作成されていることを確認
        expected_time_features = ['年', '四半期', '月初フラグ', '月末フラグ', '季節', '祝日フラグ']
        for feature in expected_time_features:
            assert feature in result.columns
        
        # 季節の割り当てが正しいことを確認
        assert result.loc[0, '季節'] == '冬'  # 1月
        assert result.loc[2, '季節'] == '夏'  # 6月
        assert result.loc[3, '季節'] == '秋'  # 9月
        
        # 祝日フラグの確認（12/25はクリスマス）
        assert result.loc[4, '祝日フラグ'] == 1
    
    def test_add_synthetic_weather_features(self, feature_engineer, sample_dataframe):
        """合成気象特徴量追加のテスト"""
        result = feature_engineer._add_synthetic_weather_features(sample_dataframe)
        
        # 気象特徴量が作成されていることを確認
        weather_features = ['平均気温', '最低気温', '最高気温', '降水量', '気温区分', '雨の日フラグ']
        for feature in weather_features:
            assert feature in result.columns
        
        # 気温の妥当性確認
        assert all(result['平均気温'] > -50)  # 常識的な範囲
        assert all(result['平均気温'] < 50)
        assert all(result['最低気温'] <= result['最高気温'])
    
    def test_select_features(self, feature_engineer, sample_dataframe):
        """特徴量選択のテスト"""
        # 数値特徴量を追加
        np.random.seed(42)  # 再現性のためのseed設定
        df_extended = sample_dataframe.copy()
        df_extended['target'] = df_extended['金額']
        df_extended['feature1'] = df_extended['金額'] * 0.8 + np.random.normal(0, 1, len(df_extended))
        df_extended['feature2'] = np.random.normal(0, 0.1, len(df_extended))  # 低相関
        
        selected_features = feature_engineer.select_features(df_extended, 'target')
        
        # 何らかの特徴量が選択されていることを確認
        assert len(selected_features) > 0
        assert isinstance(selected_features, list)
    
    def test_get_feature_importance(self, feature_engineer, sample_dataframe):
        """特徴量重要度計算のテスト"""
        # 数値特徴量を追加
        df_extended = sample_dataframe.copy()
        df_extended['target'] = df_extended['金額']
        
        features = ['金額', '数量', '平均価格']
        importance = feature_engineer.get_feature_importance(df_extended, features, 'target')
        
        # 重要度辞書が返されることを確認
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
        # 金額とtargetは同じなので重要度が高いことを確認
        assert importance.get('金額', 0) > 0.9
    
    def test_remove_highly_correlated_features(self, feature_engineer):
        """高相関特徴量除去のテスト"""
        # 高相関のテストデータを作成
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.1, 3.1, 4.1, 5.1],  # feature1と高相関
            'feature3': [5, 4, 3, 2, 1]  # feature1と負の相関
        })
        
        features = ['feature1', 'feature2', 'feature3']
        selected = feature_engineer._remove_highly_correlated_features(data, features, threshold=0.8)
        
        # 高相関の特徴量が除去されていることを確認
        assert len(selected) < len(features)
    
    def test_holiday_features(self, feature_engineer, sample_dataframe):
        """祝日特徴量のテスト"""
        result = feature_engineer._add_holiday_features(sample_dataframe)
        
        # 祝日フラグが作成されていることを確認
        assert '祝日フラグ' in result.columns
        assert 'セール期間フラグ' in result.columns
        
        # データ型の確認
        assert result['祝日フラグ'].dtype == 'int64'
        assert result['セール期間フラグ'].dtype == 'int64'