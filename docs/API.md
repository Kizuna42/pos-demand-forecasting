# 生鮮食品需要予測・分析システム API ドキュメント

## 概要

このドキュメントでは、生鮮食品需要予測・分析システムの各コンポーネントのAPI仕様を説明します。

## 📋 目次

1. [メインパイプライン API](#メインパイプライン-api)
2. [データ処理 API](#データ処理-api)
3. [特徴量エンジニアリング API](#特徴量エンジニアリング-api)
4. [モデル構築 API](#モデル構築-api)
5. [需要曲線分析 API](#需要曲線分析-api)
6. [品質評価 API](#品質評価-api)
7. [可視化 API](#可視化-api)
8. [レポート生成 API](#レポート生成-api)
9. [ユーティリティ API](#ユーティリティ-api)

---

## メインパイプライン API

### DemandForecastingPipeline

需要予測分析の統合パイプライン

#### クラス初期化

```python
DemandForecastingPipeline(config_path: str = None)
```

**パラメータ:**
- `config_path` (str, optional): 設定ファイルのパス。省略時はデフォルト設定を使用

**例:**
```python
pipeline = DemandForecastingPipeline("config/config.yaml")
```

#### run_full_analysis

完全分析の実行

```python
run_full_analysis(
    target_products: List[str] = None,
    max_products: int = 10
) -> Dict[str, Any]
```

**パラメータ:**
- `target_products` (List[str], optional): 分析対象商品リスト
- `max_products` (int, default=10): 最大処理商品数

**戻り値:**
```python
{
    'analysis_results': List[Dict],      # 商品別分析結果
    'quality_report': Dict,              # 品質評価レポート
    'visualization_files': List[str],    # 生成された可視化ファイル
    'report_files': List[str],           # 生成されたレポートファイル
    'summary': {                         # サマリー情報
        'total_products_analyzed': int,
        'success_rate': float,
        'average_r2': float
    }
}
```

**例:**
```python
results = pipeline.run_full_analysis(
    target_products=['りんご', 'キャベツ'],
    max_products=5
)
```

---

## データ処理 API

### DataProcessor

データ読み込み・クリーニング・前処理

#### クラス初期化

```python
DataProcessor(config: Config)
```

#### load_raw_data

生データの読み込み

```python
load_raw_data(file_path: str = None) -> pd.DataFrame
```

**パラメータ:**
- `file_path` (str, optional): データファイルパス。省略時は設定ファイルから取得

**戻り値:** pandas.DataFrame

**例:**
```python
processor = DataProcessor(config)
data = processor.load_raw_data("data/raw/sales_data.csv")
```

#### clean_data

データクリーニング

```python
clean_data(df: pd.DataFrame) -> pd.DataFrame
```

**パラメータ:**
- `df` (pandas.DataFrame): 生データ

**戻り値:** クリーニング済みデータ

**処理内容:**
- 欠損値処理
- 重複除去
- データ型変換
- 基本的な妥当性チェック

#### remove_outliers

外れ値除去

```python
remove_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    columns: List[str] = None
) -> pd.DataFrame
```

**パラメータ:**
- `df` (pandas.DataFrame): 入力データ
- `method` (str, default='iqr'): 外れ値検出手法 ('iqr', 'zscore')
- `columns` (List[str], optional): 対象列。省略時は数値列全て

**戻り値:** 外れ値除去後のデータ

#### detect_encoding

ファイルエンコーディングの自動検出

```python
detect_encoding(file_path: str) -> str
```

**パラメータ:**
- `file_path` (str): ファイルパス

**戻り値:** 検出されたエンコーディング名

---

## 特徴量エンジニアリング API

### FeatureEngineer

特徴量生成・変換

#### create_baseline_features

ベースライン特徴量の生成

```python
create_baseline_features(df: pd.DataFrame) -> pd.DataFrame
```

**生成される特徴量:**
- `売上単価`: 金額 ÷ 数量
- `月`: 販売月
- `曜日`: 販売曜日（0=月曜日）
- `週末フラグ`: 土日の場合1
- `商品カテゴリ`: 商品名から推定されるカテゴリ

#### add_time_features

時間特徴量の追加

```python
add_time_features(df: pd.DataFrame) -> pd.DataFrame
```

**生成される特徴量:**
- `時間帯`: 営業時間帯の区分
- `祝日フラグ`: 祝日の場合1
- `月初フラグ`: 月初3日間の場合1
- `月末フラグ`: 月末3日間の場合1

#### integrate_weather_features

気象特徴量の統合

```python
integrate_weather_features(df: pd.DataFrame) -> pd.DataFrame
```

**パラメータ:**
- `df` (pandas.DataFrame): 時間特徴量を含むデータ

**戻り値:** 気象特徴量を追加したデータ

**生成される特徴量:**
- `気温`: 日平均気温
- `湿度`: 日平均湿度
- `降水量`: 日降水量
- `気温_高`: 高温日フラグ（30°C以上）
- `気温_低`: 低温日フラグ（5°C以下）
- `雨天`: 降雨日フラグ

---

## モデル構築 API

### ModelBuilder

機械学習モデルの構築と評価

#### train_with_cv

交差検証によるモデル訓練

```python
train_with_cv(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]
```

**パラメータ:**
- `X` (pandas.DataFrame): 特徴量データ
- `y` (pandas.Series): ターゲット変数

**戻り値:**
```python
{
    'model': sklearn.model,           # 訓練済みモデル
    'test_metrics': {                 # テストデータでの性能
        'r2_score': float,
        'rmse': float,
        'mae': float
    },
    'cv_scores': {                    # 交差検証結果
        'mean_score': float,
        'std_score': float,
        'scores': List[float]
    },
    'feature_importance': Dict[str, float],  # 特徴量重要度
    'overfitting_score': float,       # 過学習スコア
    'feature_names': List[str]        # 特徴量名一覧
}
```

#### calculate_shap_values

SHAP値の計算

```python
calculate_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 100
) -> np.ndarray
```

**パラメータ:**
- `model`: 訓練済みモデル
- `X` (pandas.DataFrame): 特徴量データ
- `max_samples` (int, default=100): SHAP計算の最大サンプル数

**戻り値:** SHAP値の配列

---

## 需要曲線分析 API

### DemandCurveAnalyzer

需要曲線分析と価格最適化

#### analyze_demand_curve

需要曲線の分析

```python
analyze_demand_curve(
    data: pd.DataFrame,
    product_name: str
) -> Dict[str, Any]
```

**パラメータ:**
- `data` (pandas.DataFrame): 商品データ
- `product_name` (str): 商品名

**戻り値:**
```python
{
    'product_name': str,
    'demand_curve_function': Callable,    # 需要関数
    'optimal_price': float,               # 最適価格
    'current_price': float,               # 現在価格
    'price_elasticity': float,            # 価格弾力性
    'r2_score': float,                    # 回帰のR²スコア
    'price_demand_data': pd.DataFrame,    # 価格-需要データ
    'price_range': Tuple[float, float],   # 分析価格範囲
    'revenue_function': Callable,         # 収益関数
    'data_points': int                    # 使用データ点数
}
```

---

## 品質評価 API

### QualityEvaluator

モデル品質の評価と分類

#### evaluate_quality_level

品質レベルの評価

```python
evaluate_quality_level(r2_score: float) -> str
```

**パラメータ:**
- `r2_score` (float): R²スコア

**戻り値:** 品質レベル ('Premium', 'Standard', 'Basic', 'Rejected')

#### assess_implementation_readiness

実用化準備状況の評価

```python
assess_implementation_readiness(model_metrics: Dict[str, float]) -> str
```

**パラメータ:**
- `model_metrics` (Dict): モデル評価指標

**戻り値:** 実用化準備状況 ('即座実行', '慎重実行', '要考慮', '改善必要')

#### create_quality_report

品質レポートの作成

```python
create_quality_report(results: List[Dict]) -> Dict[str, Any]
```

**パラメータ:**
- `results` (List[Dict]): 分析結果リスト

**戻り値:**
```python
{
    'summary': {                          # サマリー情報
        'total_products': int,
        'success_rate': float,
        'average_r2': float,
        'quality_distribution': Dict[str, int],
        'implementation_distribution': Dict[str, int],
        'category_success_rates': Dict[str, float]
    },
    'detailed_analysis': List[Dict],      # 詳細分析結果
    'overall_assessment': str,            # 全体評価
    'improvement_priorities': List[str]   # 改善優先度
}
```

---

## 可視化 API

### WantPlotter

Want風スタイルでの可視化

#### create_demand_curve_plot

需要曲線プロットの作成

```python
create_demand_curve_plot(
    demand_results: Dict[str, Any],
    save_path: str = None
) -> str
```

**パラメータ:**
- `demand_results` (Dict): 需要曲線分析結果
- `save_path` (str, optional): 保存先パス

**戻り値:** 保存されたファイルのパス

#### create_feature_importance_plot

特徴量重要度プロットの作成

```python
create_feature_importance_plot(
    feature_importance: Dict[str, float],
    product_name: str,
    save_path: str = None
) -> str
```

#### create_quality_dashboard

品質ダッシュボードの作成

```python
create_quality_dashboard(
    quality_data: Dict[str, Any],
    save_path: str = None
) -> str
```

**機能:**
- 品質レベル分布（円グラフ）
- 実用化準備状況（棒グラフ）
- カテゴリ別成功率（棒グラフ）
- R²スコア統計表示

---

## レポート生成 API

### ReportGenerator

各種レポートの生成

#### generate_markdown_report

Markdownレポートの生成

```python
generate_markdown_report(
    analysis_results: List[Dict[str, Any]],
    quality_report: Dict[str, Any]
) -> str
```

**パラメータ:**
- `analysis_results` (List[Dict]): 分析結果
- `quality_report` (Dict): 品質レポート

**戻り値:** Markdown形式のレポート文字列

**レポート構成:**
1. エグゼクティブサマリー
2. 品質評価概要
3. 個別商品分析結果
4. 改善提案
5. 段階的実装計画
6. 付録

#### generate_csv_reports

CSV形式レポートの生成

```python
generate_csv_reports(
    analysis_results: List[Dict[str, Any]],
    save_dir: str = None
) -> List[str]
```

**戻り値:** 生成されたCSVファイルのパスリスト

**生成されるCSV:**
- `model_performance.csv`: モデル性能指標
- `feature_importance.csv`: 特徴量重要度
- `demand_analysis.csv`: 需要曲線分析結果

---

## ユーティリティ API

### Config

設定管理

#### クラス初期化

```python
Config(config_path: str = None)
```

#### 設定取得メソッド

```python
get_data_config() -> Dict[str, Any]           # データ設定
get_model_config() -> Dict[str, Any]          # モデル設定
get_quality_config() -> Dict[str, Any]        # 品質評価設定
get_visualization_config() -> Dict[str, Any]  # 可視化設定
get_logging_config() -> Dict[str, Any]        # ログ設定
```

### Logger

ログ管理

```python
Logger(config: Dict)
get_logger(name: str) -> logging.Logger
```

### 例外クラス

#### DemandForecastingError
システム共通の基底例外クラス

#### DataProcessingError
データ処理関連の例外

#### FeatureEngineeringError
特徴量エンジニアリング関連の例外

#### ModelBuildingError
モデル構築関連の例外

#### VisualizationError
可視化関連の例外

#### ReportGenerationError
レポート生成関連の例外

#### QualityEvaluationError
品質評価関連の例外

---

## 使用例

### 基本的な使用パターン

```python
from src.main import DemandForecastingPipeline

# パイプライン初期化
pipeline = DemandForecastingPipeline("config/config.yaml")

# 分析実行
results = pipeline.run_full_analysis(
    target_products=['りんご', 'キャベツ'],
    max_products=10
)

# 結果確認
print(f"分析商品数: {results['summary']['total_products_analyzed']}")
print(f"成功率: {results['summary']['success_rate']*100:.1f}%")
```

### 個別コンポーネントの使用

```python
from src.demand_forecasting.core.data_processor import DataProcessor
from src.demand_forecasting.utils.config import Config

config = Config("config/config.yaml")
processor = DataProcessor(config)

# データ読み込みとクリーニング
data = processor.load_raw_data()
clean_data = processor.clean_data(data)
```

### エラーハンドリング

```python
from src.demand_forecasting.utils.exceptions import DemandForecastingError

try:
    results = pipeline.run_full_analysis()
except DemandForecastingError as e:
    logger.error(f"分析エラー: {e}")
    # エラー処理
```

---

## パフォーマンス考慮事項

### メモリ使用量
- 大きなデータセット（>100MB）の場合、チャンク処理を検討
- SHAP分析は`max_samples`パラメータで制限

### 実行時間
- 気象データ取得でAPI呼び出しが発生（フォールバック機能あり）
- 交差検証の`cv_folds`数で実行時間が変わる
- 商品数が多い場合は`max_products`で制限

### 並列処理
- 現在の実装は単一プロセス
- 大規模データでは並列処理の実装を検討

---

## バージョン情報

**Current Version**: 1.0.0  
**Last Updated**: 2024年  
**Python Version**: 3.11+  
**Dependencies**: pandas, scikit-learn, matplotlib, seaborn, PyYAML, requests, scipy

---

## サポート

技術的な質問やバグレポートは、プロジェクトの [Issues](https://github.com/your-repo/issues) からお問い合わせください。