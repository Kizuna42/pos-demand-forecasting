import pytest
import pandas as pd
import numpy as np

from src.demand_forecasting.core.demand_analyzer import DemandCurveAnalyzer
from src.demand_forecasting.utils.config import Config
from src.demand_forecasting.utils.exceptions import DemandAnalysisError


class TestDemandCurveAnalyzer:
    """DemandCurveAnalyzerクラスのテスト"""

    @pytest.fixture
    def sample_config(self):
        """テスト用設定"""
        return Config()

    @pytest.fixture
    def demand_analyzer(self, sample_config):
        """DemandCurveAnalyzerインスタンス"""
        return DemandCurveAnalyzer(sample_config)

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)

        # 複数商品のデータを作成
        products = ["りんご", "キャベツ", "牛肉"]
        data = []

        for product in products:
            # 各商品について価格と数量の関係を設定（逆相関）
            n_samples = 20
            base_price = np.random.uniform(100, 500)
            prices = np.linspace(base_price * 0.8, base_price * 1.2, n_samples)

            # 線形需要関数を仮定: Q = a - b*P + noise
            a = 100
            b = 0.2
            quantities = a - b * prices + np.random.normal(0, 5, n_samples)
            quantities = np.maximum(quantities, 1)  # 負の数量を避ける

            for i in range(n_samples):
                data.append(
                    {
                        "商品名称": product,
                        "平均価格": prices[i],
                        "数量": quantities[i],
                        "金額": prices[i] * quantities[i],
                    }
                )

        return pd.DataFrame(data)

    def test_extract_product_data(self, demand_analyzer, sample_data):
        """商品データ抽出のテスト"""
        # 完全一致での抽出
        product_data = demand_analyzer._extract_product_data(sample_data, "りんご")
        assert len(product_data) > 0
        assert all(product_data["商品名称"] == "りんご")

        # 存在しない商品の場合
        with pytest.raises(DemandAnalysisError):
            demand_analyzer._extract_product_data(sample_data, "存在しない商品")

    def test_prepare_price_demand_data(self, demand_analyzer, sample_data):
        """価格-需要データ準備のテスト"""
        product_data = sample_data[sample_data["商品名称"] == "りんご"]

        price_demand_data = demand_analyzer._prepare_price_demand_data(product_data)

        # 必要な列が存在することを確認
        assert "price" in price_demand_data.columns
        assert "quantity" in price_demand_data.columns

        # 正の値のみが残っていることを確認
        assert all(price_demand_data["price"] > 0)
        assert all(price_demand_data["quantity"] > 0)

        # 価格でソートされていることを確認
        assert price_demand_data["price"].is_monotonic_increasing

    def test_remove_outliers_iqr(self, demand_analyzer):
        """IQR法による外れ値除去のテスト"""
        # 外れ値を含むテストデータ
        data = pd.DataFrame(
            {
                "price": [10, 20, 30, 40, 50, 1000],  # 1000は外れ値
                "quantity": [50, 40, 30, 20, 10, 5],
            }
        )

        cleaned_data = demand_analyzer._remove_outliers_iqr(data)

        # 外れ値が除去されていることを確認
        assert len(cleaned_data) < len(data)
        assert 1000 not in cleaned_data["price"].values

    def test_apply_smoothing_savgol(self, demand_analyzer):
        """Savitzky-Golay平滑化のテスト"""
        data = pd.DataFrame(
            {
                "price": np.linspace(10, 50, 10),
                "quantity": [50, 45, 42, 38, 35, 32, 28, 25, 22, 20],
            }
        )

        smoothed_data = demand_analyzer._apply_smoothing(data, method="savgol")

        # データ数が変わらないことを確認
        assert len(smoothed_data) == len(data)

        # priceは変更されず、quantityが平滑化されることを確認
        pd.testing.assert_series_equal(smoothed_data["price"], data["price"])
        # quantityは変更されている（ただし完全に同じでない）
        assert not smoothed_data["quantity"].equals(data["quantity"])

    def test_apply_smoothing_small_data(self, demand_analyzer):
        """少ないデータでの平滑化のテスト"""
        small_data = pd.DataFrame({"price": [10, 20], "quantity": [50, 40]})

        # 少ないデータの場合は平滑化がスキップされる
        result = demand_analyzer._apply_smoothing(small_data)
        pd.testing.assert_frame_equal(result, small_data)

    def test_linear_demand(self, demand_analyzer):
        """線形需要関数のテスト"""
        prices = np.array([10, 20, 30])
        a, b = 100, 2

        quantities = demand_analyzer._linear_demand(prices, a, b)
        expected = a - b * prices

        np.testing.assert_array_equal(quantities, expected)

    def test_exponential_demand(self, demand_analyzer):
        """指数需要関数のテスト"""
        prices = np.array([1, 2, 3])
        a, b = 100, 0.5

        quantities = demand_analyzer._exponential_demand(prices, a, b)
        expected = a * np.exp(-b * prices)

        np.testing.assert_array_almost_equal(quantities, expected)

    def test_power_demand(self, demand_analyzer):
        """べき乗需要関数のテスト"""
        prices = np.array([1, 2, 4])
        a, b = 100, 1.5

        quantities = demand_analyzer._power_demand(prices, a, b)
        expected = a * np.power(prices, -b)

        np.testing.assert_array_almost_equal(quantities, expected)

    def test_analyze_demand_curve(self, demand_analyzer, sample_data):
        """需要曲線分析のテスト"""
        results = demand_analyzer.analyze_demand_curve(sample_data, "りんご")

        # 必要なキーが含まれていることを確認
        expected_keys = [
            "product_name",
            "optimal_price",
            "current_price",
            "price_elasticity",
            "demand_curve_function",
            "fit_params",
            "r2_score",
            "price_range",
            "quantity_range",
            "data_points",
            "price_demand_data",
        ]

        for key in expected_keys:
            assert key in results

        # 値が妥当であることを確認
        assert results["product_name"] == "りんご"
        assert results["optimal_price"] > 0
        assert results["current_price"] > 0
        assert isinstance(results["price_elasticity"], float)
        assert callable(results["demand_curve_function"])
        assert 0 <= results["r2_score"] <= 1
        assert len(results["price_range"]) == 2
        assert len(results["quantity_range"]) == 2
        assert results["data_points"] > 0

    def test_analyze_demand_curve_insufficient_data(self, demand_analyzer):
        """データ不足時の需要曲線分析のテスト"""
        # 少ないデータを作成
        insufficient_data = pd.DataFrame(
            {"商品名称": ["りんご", "りんご"], "平均価格": [100, 200], "数量": [10, 5]}
        )

        # データ不足エラーが発生することを確認
        with pytest.raises(DemandAnalysisError, match="データ不足"):
            demand_analyzer.analyze_demand_curve(insufficient_data, "りんご")

    def test_analyze_multiple_products(self, demand_analyzer, sample_data):
        """複数商品需要曲線分析のテスト"""
        products = ["りんご", "キャベツ"]
        results = demand_analyzer.analyze_multiple_products(sample_data, products)

        # 指定した商品の結果が返されることを確認
        assert len(results) <= len(products)
        for product in results.keys():
            assert product in products

    def test_get_demand_summary(self, demand_analyzer, sample_data):
        """需要分析サマリー生成のテスト"""
        # まず需要曲線分析を実行
        analysis_results = demand_analyzer.analyze_demand_curve(sample_data, "りんご")

        # サマリーを生成
        summary = demand_analyzer.get_demand_summary(analysis_results)

        # 必要なキーが含まれていることを確認
        expected_keys = [
            "product_name",
            "current_price",
            "optimal_price",
            "price_change_pct",
            "price_elasticity",
            "elasticity_category",
            "model_fit_quality",
            "data_points",
        ]

        for key in expected_keys:
            assert key in summary

        # 値の妥当性確認
        assert summary["product_name"] == "りんご"
        assert isinstance(summary["price_change_pct"], float)
        assert summary["elasticity_category"] in ["高弾力性", "中弾力性", "低弾力性"]

    def test_categorize_elasticity(self, demand_analyzer):
        """価格弾力性カテゴリ化のテスト"""
        # 高弾力性
        assert demand_analyzer._categorize_elasticity(-2.0) == "高弾力性"
        assert demand_analyzer._categorize_elasticity(2.0) == "高弾力性"

        # 中弾力性
        assert demand_analyzer._categorize_elasticity(-1.0) == "中弾力性"
        assert demand_analyzer._categorize_elasticity(1.0) == "中弾力性"

        # 低弾力性
        assert demand_analyzer._categorize_elasticity(-0.3) == "低弾力性"
        assert demand_analyzer._categorize_elasticity(0.3) == "低弾力性"

    def test_calculate_price_elasticity(self, demand_analyzer):
        """価格弾力性計算のテスト"""

        # 簡単な線形需要関数でテスト
        def simple_demand_func(price):
            return 100 - 2 * price  # Q = 100 - 2P

        elasticity = demand_analyzer._calculate_price_elasticity(simple_demand_func, 20)

        # 弾力性が負の値であることを確認（正常な需要曲線）
        assert elasticity < 0
        assert isinstance(elasticity, float)
