from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.demand_forecasting.core.data_processor import DataProcessor
from src.demand_forecasting.utils.config import Config
from src.demand_forecasting.utils.exceptions import DataProcessingError


class TestDataProcessor:
    """DataProcessorクラスのテスト"""

    @pytest.fixture
    def sample_config(self):
        """テスト用設定"""
        return Config()

    @pytest.fixture
    def data_processor(self, sample_config):
        """DataProcessorインスタンス"""
        return DataProcessor(sample_config)

    @pytest.fixture
    def sample_dataframe(self):
        """テスト用サンプルデータ"""
        data = {
            "商品コード": ["001", "002", "003", "001", "004"],
            "商品名称": ["りんご", "キャベツ", "牛肉", "りんご", "まぐろ"],
            "年月日": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-04"],
            "金額": [300, 150, 800, 300, 500],
            "数量": [3, 1, 2, 3, 1],
            "平均価格": [100, 150, 400, 100, 500],
        }
        return pd.DataFrame(data)

    def test_handle_missing_values(self, data_processor, sample_dataframe):
        """欠損値処理のテスト"""
        # 意図的に欠損値を作成
        df_with_missing = sample_dataframe.copy()
        df_with_missing.loc[0, "金額"] = np.nan
        df_with_missing.loc[1, "商品名称"] = np.nan

        result = data_processor.handle_missing_values(df_with_missing)

        # 欠損値が処理されていることを確認
        assert not result.isnull().any().any()

    def test_remove_duplicates(self, data_processor, sample_dataframe):
        """重複除去のテスト"""
        # 重複行がある場合
        result = data_processor.remove_duplicates(sample_dataframe)

        # 重複が除去されていることを確認（商品コード'001'の重複）
        assert len(result) == 4  # 元の5行から1行削除

    def test_create_basic_features(self, data_processor, sample_dataframe):
        """基本特徴量生成のテスト"""
        result = data_processor.create_basic_features(sample_dataframe)

        # 新しい特徴量が作成されていることを確認
        expected_columns = ["売上単価", "月", "曜日", "週末フラグ", "商品カテゴリ"]
        for col in expected_columns:
            assert col in result.columns

        # 売上単価の計算が正しいことを確認
        assert result.loc[0, "売上単価"] == 100  # 300/3
        assert result.loc[1, "売上単価"] == 150  # 150/1

    def test_categorize_products(self, data_processor, sample_dataframe):
        """商品カテゴリ化のテスト"""
        result = data_processor.create_basic_features(sample_dataframe)

        # カテゴリが正しく設定されていることを確認
        categories = result["商品カテゴリ"].tolist()
        assert "果物" in categories  # りんご
        assert "野菜" in categories  # キャベツ
        assert "肉類" in categories  # 牛肉
        assert "魚類" in categories  # まぐろ

    def test_remove_outliers_iqr(self, data_processor):
        """IQR法による外れ値除去のテスト"""
        # 外れ値を含むデータを作成
        data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})  # 100は外れ値

        result = data_processor.remove_outliers(data, method="iqr", columns=["value"])

        # 外れ値が除去されていることを確認
        assert len(result) < len(data)
        assert 100 not in result["value"].values
