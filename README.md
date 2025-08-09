# 生鮮食品需要予測・分析システム

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)

## 概要

生鮮食品の需要予測・分析を行う包括的な Python システムです。時刻・混雑度・外気温を統合した高精度な需要予測モデルを構築し、商品の最適価格設定、需要パターンの理解、在庫管理の最適化を支援します。

### 🎯 主要機能

- **高精度需要予測**: RandomForest + 交差検証による堅牢なモデル構築
- **価格最適化**: 需要曲線分析による収益最大化価格の算出
- **段階的特徴量エンジニアリング**: ベースライン → 時間 → 気象特徴量の段階的追加
- **品質評価システム**: Premium/Standard/Basic/Rejected の 4 段階自動評価
- **Want 風可視化**: 直感的で美しいダッシュボードとグラフ
- **包括的レポート**: Markdown + CSV 形式での詳細分析結果
- **完全テストカバレッジ**: ユニット・統合・E2E テスト完備

## 🚀 クイックスタート

### インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd pos_Aer

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係のインストール
pip install -r requirements.txt
```

### 基本的な使用方法

```bash
# データが data/raw/ にある場合の基本実行
python src/main.py

# 設定ファイルを指定して実行
python src/main.py --config config/config.yaml

# 特定商品のみ分析
python src/main.py --products りんご キャベツ --max-products 5

# 詳細ログ出力
python src/main.py --verbose
```

### 📊 出力例

実行後、以下のファイルが生成されます：

- **レポート**: `reports/demand_forecasting_report_YYYYMMDD_HHMMSS.md`
- **CSV 分析結果**: `reports/model_performance.csv`, `reports/feature_importance.csv`
- **可視化**: `output/visualizations/` 配下に PNG ファイル
- **ログ**: `logs/demand_forecasting.log`

## 📁 プロジェクト構造

```
pos_Aer/
├── 📋 .kiro/specs/demand-forecasting-analysis/  # システム仕様書
│   ├── requirements.md                          # 要件定義
│   ├── design.md                               # 設計書
│   └── tasks.md                                # 実装タスクリスト
├── 📊 data/
│   ├── raw/                                    # 生データ（Shift-JIS CSV）
│   ├── processed/                              # 前処理済みデータ
│   ├── interim/                                # 中間処理データ
│   └── external/                               # 外部データ（気象データ等）
├── 🔧 src/                                     # メインソースコード
│   ├── main.py                                 # エントリーポイント
│   └── demand_forecasting/                     # 需要予測パッケージ
│       ├── core/                               # コア機能
│       │   ├── data_processor.py               # データ処理
│       │   ├── feature_engineer.py             # 特徴量エンジニアリング
│       │   ├── model_builder.py                # ML モデル構築
│       │   └── demand_analyzer.py              # 需要曲線分析
│       ├── utils/                              # ユーティリティ
│       │   ├── config.py                       # 設定管理
│       │   ├── logger.py                       # ログ管理
│       │   ├── exceptions.py                   # 例外定義
│       │   └── quality_evaluator.py            # 品質評価
│       ├── visualization/                      # 可視化
│       │   └── want_plotter.py                 # Want風プロッター
│       └── reports/                            # レポート生成
│           └── report_generator.py             # レポート生成器
├── ⚙️ config/                                  # 設定ファイル
│   └── config.yaml                             # メイン設定
├── 🧪 tests/                                   # テストスイート
│   ├── conftest.py                             # テスト設定
│   ├── test_*.py                               # 各種テスト
│   ├── test_integration.py                     # 統合テスト
│   └── test_end_to_end.py                      # E2Eテスト
├── 📈 output/                                  # 出力ファイル
│   └── visualizations/                         # 生成された可視化
├── 📄 reports/                                 # 生成されたレポート
├── 💾 models/                                  # 学習済みモデル
├── 📝 logs/                                    # ログファイル
├── 📚 notebooks/                               # Jupyter ノートブック
│   └── 完全版_時刻混雑度外気温統合_需要曲線分析.ipynb
├── 📖 docs/                                    # ドキュメント
└── 🎨 want_style_plotter.py                    # 可視化スタイル定義
```

## ✅ 実装状況

### 完了済み機能

- [x] **プロジェクト基盤構築** - 設定管理・ログ・例外処理
- [x] **データ処理モジュール** - Shift-JIS 対応・外れ値除去・データクリーニング
- [x] **特徴量エンジニアリング** - ベースライン・時間・気象特徴量の統合
- [x] **ML モデル構築** - RandomForest・交差検証・SHAP 解釈性
- [x] **需要曲線分析** - 価格最適化・価格弾力性・収益最大化
- [x] **品質評価システム** - 4 段階品質レベル・実用化準備状況評価
- [x] **Want 風可視化システム** - 需要曲線・特徴量重要度・品質ダッシュボード
- [x] **レポート生成** - Markdown・CSV 形式での包括的レポート
- [x] **メインワークフロー** - 統合された分析パイプライン
- [x] **包括的テストスイート** - ユニット・統合・E2E テスト（73+ テスト）

### 📈 システム品質指標

- **テストカバレッジ**: 包括的（ユニット・統合・E2E）
- **コード品質**: モジュラー設計・エラーハンドリング完備
- **文書化**: 要件定義・設計書・API ドキュメント
- **保守性**: 設定ファイルベース・ログ出力・例外階層

## 🛠 技術スタック

### コア技術

- **Python 3.11+** - メイン言語
- **pandas** - データ処理・操作
- **numpy** - 数値計算・配列操作
- **scikit-learn** - 機械学習（RandomForest）
- **matplotlib/seaborn** - Want 風可視化
- **PyYAML** - 設定ファイル管理

### 高度な機能

- **SHAP** - モデル解釈性・特徴量重要度（任意）
- **scipy** - Savitzky-Golay 平滑化・最適化
- **Open-Meteo API** - 外部気象データ取得
- **requests** - HTTP API 通信

### 開発・テスト

- **pytest** - テストフレームワーク
- **logging** - 構造化ログ出力
- **pathlib** - パス操作
- **tempfile** - テスト用一時ファイル

### 任意機能の有効化（XGBoost / ベイズ最適化）

アンサンブルやベイズ最適化を利用する場合、以下を追加インストールしてください。

```bash
pip install xgboost scikit-optimize
```

未インストールでもエラーにはなりません（警告ログのみに抑制され、RandomForest のみで動作します）。

注記: これらはデフォルトの `requirements.txt` には含まれていません（任意機能）。必要時のみ個別にインストールしてください。

## 💡 使用例

### Python スクリプトでの基本使用

```python
from src.main import DemandForecastingPipeline

# パイプライン初期化
pipeline = DemandForecastingPipeline("config/config.yaml")

# 全体分析実行
results = pipeline.run_full_analysis(max_products=10)

# 結果の確認
print(f"分析商品数: {results['summary']['total_products_analyzed']}")
print(f"成功率: {results['summary']['success_rate']*100:.1f}%")
print(f"平均R²スコア: {results['summary']['average_r2']:.3f}")
```

### 個別コンポーネントの使用

```python
from src.demand_forecasting.core.data_processor import DataProcessor
from src.demand_forecasting.core.feature_engineer import FeatureEngineer
from src.demand_forecasting.utils.config import Config

# 設定読み込み
config = Config("config/config.yaml")

# データ処理
processor = DataProcessor(config)
data = processor.load_raw_data()
clean_data = processor.clean_data(data)

# 特徴量エンジニアリング
engineer = FeatureEngineer(config)
features = engineer.create_baseline_features(clean_data)
features = engineer.integrate_weather_features(features)
```

### コマンドライン実行

```bash
# 基本実行（全商品・デフォルト設定）
python src/main.py

# 特定商品のみ分析
python src/main.py --products りんご キャベツ 牛肉

# 商品数制限・詳細ログ
python src/main.py --max-products 5 --verbose

# カスタム設定ファイル使用
python src/main.py --config my_config.yaml
```

## 📊 データ仕様

### 入力データ形式（CSV）

| 列名       | データ型 | 説明           | 例                   |
| ---------- | -------- | -------------- | -------------------- |
| 商品コード | String   | 商品識別コード | "001", "ABC123"      |
| 商品名称   | String   | 商品名         | "りんご", "キャベツ" |
| 年月日     | String   | 販売日付       | "2024-01-15"         |
| 金額       | Integer  | 販売金額（円） | 300, 1500            |
| 数量       | Integer  | 販売数量       | 3, 5                 |
| 平均価格   | Integer  | 単価（円）     | 100, 300             |

**重要事項**:

- **文字エンコーディング**: Shift-JIS（自動検出・変換対応）
- **データ期間**: 推奨は 3 ヶ月以上の日次データ
- **最小データ量**: 商品あたり 10 レコード以上

### 出力ファイル構成

#### レポート（`reports/`）

- **`demand_forecasting_report_YYYYMMDD_HHMMSS.md`** - 包括的分析レポート
- **`model_performance.csv`** - モデル性能メトリクス
- **`feature_importance.csv`** - 特徴量重要度ランキング
- **`demand_analysis.csv`** - 需要曲線分析結果

#### 可視化（`output/visualizations/`）

- **`quality_dashboard.png`** - 品質評価ダッシュボード
- **`demand_curve_[商品名].png`** - 需要曲線・収益曲線
- **`feature_importance_[商品名].png`** - 特徴量重要度

#### その他

- **`logs/demand_forecasting.log`** - 実行ログ
- **`models/`** - 学習済みモデル（将来拡張用）

## 🎯 品質評価システム

### 品質レベル定義

| レベル          | R² スコア | 実用化状況 | 推奨アクション   | 説明                           |
| --------------- | --------- | ---------- | ---------------- | ------------------------------ |
| 🟢 **Premium**  | ≥ 0.7     | 即座実行   | 即座に本番導入   | 高精度・安定性・信頼性を兼備   |
| 🟡 **Standard** | 0.5-0.7   | 慎重実行   | A/B テスト後導入 | 中精度・検証後の段階的導入     |
| 🟠 **Basic**    | 0.3-0.5   | 要考慮     | 改善後に検討     | 低精度・特徴量見直しが必要     |
| 🔴 **Rejected** | < 0.3     | 改善必要   | 全面的な見直し   | 精度不足・アルゴリズム変更検討 |

### 実用化準備状況

システムは各商品について以下の実用化準備状況を自動判定します：

- **即座実行** - Premium 品質 + 過学習なし
- **慎重実行** - Standard 品質 + 過学習なし
- **要考慮** - Basic 品質または軽微な過学習
- **改善必要** - Rejected 品質または重度の過学習

### 品質指標の詳細

- **R² スコア** - モデルの説明力（1 に近いほど高精度）
- **RMSE/MAE** - 予測誤差の大きさ
- **交差検証安定性** - 異なるデータでの性能一貫性
- **過学習スコア** - 訓練データとテストデータの性能差

## 🧪 テスト実行

### 全テスト実行

```bash
# 全テストスイート実行
pytest tests/ -v

# カバレッジ付きテスト
pytest tests/ --cov=src --cov-report=html

# 特定カテゴリのテストのみ
pytest tests/ -m "not slow"  # 高速テストのみ
pytest tests/ -m "integration"  # 統合テストのみ
pytest tests/ -m "e2e"  # E2Eテストのみ
```

### 個別コンポーネントテスト

```bash
pytest tests/test_data_processor.py -v
pytest tests/test_quality_evaluator.py -v
pytest tests/test_want_plotter.py -v
```

## 🔧 設定カスタマイズ

### config.yaml 主要設定項目

```yaml
data:
  raw_data_path: "data/raw/your_data.csv" # データファイルパス
  encoding: "shift_jis" # ファイルエンコーディング

feature_engineering:
  weather:
    api_endpoint: "https://api.open-meteo.com/v1/forecast"
    fallback_enabled: true # API失敗時の代替データ生成

model:
  algorithm: "RandomForest"
  n_estimators: 100 # 決定木の数
  max_depth: 10 # 最大深度
  cv_folds: 5 # 交差検証のフォールド数

quality:
  thresholds:
    premium: 0.7 # Premium品質の閾値
    standard: 0.5 # Standard品質の閾値
    basic: 0.3 # Basic品質の閾値

visualization:
  output_dir: "output/visualizations"
  dpi: 300 # 画像解像度
```

## 🤝 開発・貢献

### 開発環境セットアップ

```bash
# 開発用依存関係
pip install pytest matplotlib seaborn pandas scikit-learn

# テスト実行
pytest tests/

# 新機能開発時の推奨フロー
1. tests/ に テストケース作成
2. src/ に実装
3. pytest でテスト通過確認
4. 統合テスト実行
```

### コントリビューション

1. **Issue 作成** - バグ報告・機能要望
2. **Fork & Pull Request** - コード改善
3. **テスト必須** - 新機能には対応テストを追加
4. **ドキュメント更新** - README.md・仕様書の更新

## 📝 ライセンス・クレジット

**MIT License** - 商用・非商用問わず自由に利用可能

### 作成者

- **システム設計・機械学習**: Claude (Anthropic)
- **可視化スタイル**: want_style_plotter.py 準拠
- **プロジェクト管理**: 仕様書ベース段階的開発

### 技術的貢献

- **Open-Meteo**: 気象データ API 提供
- **scikit-learn**: 機械学習フレームワーク
- **matplotlib/seaborn**: 可視化ライブラリ

## 📈 更新履歴

### v1.0.0 (2024 年)

- ✅ **全 11 タスク完了** - 要件定義から実装・テストまで
- ✅ **包括的テストスイート** - 73+ テストケース
- ✅ **Want 風可視化システム** - 美しいダッシュボード
- ✅ **段階的品質評価** - Premium/Standard/Basic/Rejected
- ✅ **価格最適化機能** - 需要曲線分析・収益最大化
- ✅ **完全ドキュメント** - README・API・仕様書

---

### ⚠️ 重要事項

- **対象**: 生鮮食品の需要予測に特化設計
- **データ要件**: 商品あたり最低 10 レコード推奨
- **実用化**: Premium/Standard 品質の商品は実運用可能
- **拡張性**: モジュラー設計により機能追加・改修が容易

**サポート**: 技術的質問は Issues よりお気軽にお問合せください 🚀
