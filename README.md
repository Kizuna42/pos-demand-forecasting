# 生鮮食品需要予測・分析システム

## 概要

生鮮食品の需要予測・分析を行う Python システムです。時刻・混雑度・外気温を統合した高精度な需要予測モデルを構築し、商品の最適価格設定、需要パターンの理解、在庫管理の最適化を支援します。

## 主な機能

- **段階的特徴量エンジニアリング**: ベースライン → 時間特徴量 → 気象特徴量の段階的追加
- **堅牢なモデル構築**: RandomForest による予測モデル + 交差検証
- **需要曲線分析**: 価格最適化・価格弾力性の定量化
- **品質評価システム**: Premium/Standard/Basic/Rejected の 4 段階品質レベル
- **Want 風可視化**: 直感的で美しいグラフ・ダッシュボード
- **包括的レポート**: Markdown + CSV 形式での詳細分析結果

## プロジェクト構造

```
├── .kiro/specs/demand-forecasting-analysis/  # 仕様書
│   ├── requirements.md                       # 要件定義
│   ├── design.md                            # 設計書
│   └── tasks.md                             # 実装タスク
├── notebooks/                               # 分析ノートブック
│   └── 完全版_時刻混雑度外気温統合_需要曲線分析.ipynb
├── data/
│   └── raw/                                 # 生データ
│       └── 納品用_20240101_20241231_生鮮全品data.csv
├── want_style_plotter.py                   # 可視化スタイル定義
├── src/                                     # ソースコード（実装予定）
├── config/                                  # 設定ファイル
├── tests/                                   # テストコード
└── docs/                                    # ドキュメント
```

## 開発状況

### ✅ 完了

- [x] 要件定義（8 要件）
- [x] システム設計
- [x] 実装タスク定義（11 タスク）
- [x] プロトタイプ分析（Jupyter notebook）

### 🚧 実装中

- [ ] プロジェクト基盤構築
- [ ] データ処理モジュール
- [ ] 特徴量エンジニアリング
- [ ] ML モデル構築
- [ ] 需要曲線分析
- [ ] 品質評価システム
- [ ] 可視化システム
- [ ] レポート生成

## 技術スタック

- **Python 3.11+**
- **データ処理**: pandas, numpy
- **機械学習**: scikit-learn, RandomForest
- **可視化**: matplotlib, seaborn
- **外部 API**: Open-Meteo (気象データ)
- **平滑化**: scipy (Savitzky-Golay)
- **解釈可能性**: SHAP

## インストール

```bash
# リポジトリのクローン
git clone <repository-url>
cd pos_Aer

# 依存関係のインストール
pip install -r requirements.txt

# 設定ファイルの準備
cp config/config.yaml.example config/config.yaml
```

## 使用方法

### 1. 基本的な分析実行

```bash
python main.py --config config/config.yaml
```

### 2. Jupyter notebook での分析

```bash
jupyter notebook notebooks/完全版_時刻混雑度外気温統合_需要曲線分析.ipynb
```

### 3. 可視化のみ実行

```python
from src.visualization.want_plotter import WantStylePlotter

plotter = WantStylePlotter()
plotter.create_all_want_visualizations(analysis_results)
```

## データ形式

### 入力データ（CSV）

- **文字エンコーディング**: Shift-JIS
- **必須列**: 商品コード, 商品名称, 年月日, 金額, 数量, 平均価格
- **期間**: 2024 年 1 月〜12 月の日次データ

### 出力データ

- **可視化**: `output/visualizations/` (PNG 形式)
- **モデル**: `models/` (pickle 形式)
- **レポート**: `reports/` (Markdown + CSV)

## 品質レベル

| レベル   | R² スコア | 実用化状況 | 説明                       |
| -------- | --------- | ---------- | -------------------------- |
| Premium  | ≥ 0.7     | 即座実行   | 高精度、即座に実用化可能   |
| Standard | 0.5-0.7   | 慎重実行   | 中精度、慎重な検証後実用化 |
| Basic    | 0.3-0.5   | 要考慮     | 低精度、改善検討が必要     |
| Rejected | < 0.3     | 改善必要   | 精度不足、大幅な改善が必要 |

## 開発・貢献

### 開発環境のセットアップ

```bash
# 開発用依存関係のインストール
pip install -r requirements-dev.txt

# テストの実行
pytest tests/

# コード品質チェック
flake8 src/
black src/
```

### タスクの実行

1. `.kiro/specs/demand-forecasting-analysis/tasks.md` を確認
2. 実行したいタスクを選択
3. 対応する実装を行う
4. テストを作成・実行
5. プルリクエストを作成

## ライセンス

MIT License

## 作成者

- データサイエンス・機械学習: [Your Name]
- 可視化・UI: want_style_plotter.py 準拠

## 更新履歴

### v0.1.0 (2024-XX-XX)

- 初期仕様作成
- プロトタイプ分析完了
- プロジェクト基盤構築

---

**注意**: このシステムは生鮮食品の需要予測に特化して設計されています。他の商品カテゴリでの使用には追加の調整が必要な場合があります。
