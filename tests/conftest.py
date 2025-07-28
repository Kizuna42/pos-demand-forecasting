import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import tempfile

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# matplotlibのバックエンドをテスト用に設定
import matplotlib
matplotlib.use('Agg')


@pytest.fixture(scope="session")
def test_data_dir():
    """テスト用データディレクトリ"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(scope="session")
def sample_raw_data():
    """テスト用生データ（セッション全体で共有）"""
    np.random.seed(42)  # 再現性のため
    
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    products = ['りんご', 'キャベツ', '牛肉', 'まぐろ', 'じゃがいも']
    
    data = []
    for date in dates:
        for product in np.random.choice(products, size=np.random.randint(1, 4), replace=False):
            # 価格に基づく需要量の生成
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
def temp_workspace():
    """一時的なワークスペース"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 必要なディレクトリ構造を作成
        workspace = Path(tmp_dir)
        
        # プロジェクト構造を模倣
        (workspace / "data" / "raw").mkdir(parents=True)
        (workspace / "data" / "processed").mkdir(parents=True)
        (workspace / "models").mkdir(parents=True)
        (workspace / "reports").mkdir(parents=True)
        (workspace / "output" / "visualizations").mkdir(parents=True)
        (workspace / "logs").mkdir(parents=True)
        (workspace / "config").mkdir(parents=True)
        
        yield workspace


@pytest.fixture
def mock_weather_api(monkeypatch):
    """気象APIのモック"""
    def mock_get_weather_data(*args, **kwargs):
        # モックの気象データを返す
        return pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'temperature': np.random.normal(15, 5, 10),
            'humidity': np.random.normal(60, 10, 10),
            'precipitation': np.random.exponential(2, 10)
        })
    
    # 実際の気象API呼び出しをモックに置き換え
    monkeypatch.setattr(
        "src.demand_forecasting.core.feature_engineer.FeatureEngineer._fetch_weather_data",
        mock_get_weather_data
    )


@pytest.fixture(autouse=True)
def setup_test_environment():
    """テスト環境のセットアップ（自動実行）"""
    # テスト実行前の準備
    original_cwd = os.getcwd()
    
    yield
    
    # テスト実行後のクリーンアップ
    os.chdir(original_cwd)
    
    # テスト中に作成された一時ファイルをクリーンアップ
    test_files = [
        "test_output.png",
        "test_report.md",
        "test_data.csv"
    ]
    
    for file_name in test_files:
        if os.path.exists(file_name):
            os.remove(file_name)


@pytest.fixture
def suppress_logs(caplog):
    """ログ出力を抑制"""
    import logging
    caplog.set_level(logging.WARNING)
    return caplog


# テスト実行時の設定
def pytest_configure(config):
    """pytestの設定"""
    # カスタムマーカーの登録
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """テストアイテムの修正"""
    # 統合テストとE2Eテストにマーカーを自動追加
    for item in items:
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "test_end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)


# テスト用のユーティリティ関数
class TestHelper:
    """テスト用ヘルパークラス"""
    
    @staticmethod
    def create_sample_csv(file_path: Path, data: pd.DataFrame):
        """サンプルCSVファイルを作成"""
        data.to_csv(file_path, index=False, encoding='utf-8')
    
    @staticmethod
    def assert_dataframe_structure(df: pd.DataFrame, expected_columns: list):
        """データフレーム構造の検証"""
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in expected_columns:
            assert col in df.columns
    
    @staticmethod
    def assert_file_exists_and_valid(file_path: str, expected_extension: str = None):
        """ファイルの存在と妥当性を検証"""
        assert os.path.exists(file_path)
        if expected_extension:
            assert file_path.endswith(expected_extension)
        assert os.path.getsize(file_path) > 0


@pytest.fixture
def test_helper():
    """テストヘルパーのインスタンス"""
    return TestHelper()