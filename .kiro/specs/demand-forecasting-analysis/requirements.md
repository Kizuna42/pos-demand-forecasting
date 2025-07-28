# Requirements Document

## Introduction

生鮮食品の需要予測・分析システムを構築します。既存の notebook と生鮮食品データを活用し、実用的で解釈可能な需要予測モデルを開発し、want_style_plotter.py で定義されたスタイルに従った可視化を提供します。このシステムは商品の最適価格設定、需要パターンの理解、在庫管理の最適化を支援します。

## Requirements

### Requirement 1

**User Story:** As a データアナリスト, I want 生鮮食品データを効率的に前処理できる, so that 高品質な分析基盤を構築できる

#### Acceptance Criteria

1. WHEN CSV データを読み込む THEN システム SHALL 文字エンコーディングを自動検出し適切に処理する
2. WHEN データクリーニングを実行する THEN システム SHALL 欠損値・重複・異常値を検出し適切に処理する
3. WHEN 日付データを処理する THEN システム SHALL 年月日を標準 datetime 形式に変換する
4. WHEN 基本特徴量を生成する THEN システム SHALL 売上単価・月・曜日・週末フラグを自動計算する

### Requirement 2

**User Story:** As a データサイエンティスト, I want 段階的な特徴量エンジニアリングを実行できる, so that モデル性能を段階的に改善できる

#### Acceptance Criteria

1. WHEN ベースライン特徴量を作成する THEN システム SHALL 基本的な時系列・価格特徴量を生成する
2. WHEN 時間特徴量を追加する THEN システム SHALL 時刻・混雑度・時間帯ダミー変数を生成する
3. WHEN 気象特徴量を統合する THEN システム SHALL 外気温データを取得し気温関連特徴量を作成する
4. WHEN 特徴量を選択する THEN システム SHALL 相関分析・重要度分析による特徴量選択を実行する

### Requirement 3

**User Story:** As a 機械学習エンジニア, I want 堅牢な需要予測モデルを構築できる, so that 実用的な予測精度を実現できる

#### Acceptance Criteria

1. WHEN モデルを訓練する THEN システム SHALL RandomForestRegressor を使用し交差検証で評価する
2. WHEN モデル性能を評価する THEN システム SHALL R² スコア・RMSE・MAE による多角的評価を行う
3. WHEN 過学習を防ぐ THEN システム SHALL 訓練・検証・テストデータの適切な分割を実行する
4. WHEN モデルを解釈する THEN システム SHALL 特徴量重要度・SHAP 値による解釈可能性を提供する

### Requirement 4

**User Story:** As a ビジネスアナリスト, I want 需要曲線分析と価格最適化を実行できる, so that 収益最大化戦略を立案できる

#### Acceptance Criteria

1. WHEN 需要曲線を分析する THEN システム SHALL 価格-需要関係を非線形回帰で分析する
2. WHEN 最適価格を計算する THEN システム SHALL 収益最大化ポイントを数値的に求める
3. WHEN 価格弾力性を計算する THEN システム SHALL 各商品の価格感応度を定量化する
4. WHEN 外れ値を処理する THEN システム SHALL IQR 法と Savitzky-Golay 平滑化を適用する

### Requirement 5

**User Story:** As a マネージャー, I want 直感的で美しい可視化レポートを生成できる, so that 意思決定に必要な洞察を得られる

#### Acceptance Criteria

1. WHEN 需要曲線を可視化する THEN システム SHALL want_style_plotter の需要曲線プロットを生成する
2. WHEN 特徴量重要度を可視化する THEN システム SHALL 重要度ランキングと分布を可視化する
3. WHEN 品質ダッシュボードを生成する THEN システム SHALL Premium/Standard/Basic/Rejected 品質レベルを可視化する
4. WHEN 可視化を保存する THEN システム SHALL 高解像度 PNG 形式で output/visualizations に保存する

### Requirement 6

**User Story:** As a 品質管理者, I want モデル品質を自動評価できる, so that 実用化可能なモデルを識別できる

#### Acceptance Criteria

1. WHEN モデル品質を評価する THEN システム SHALL R² スコアに基づく 4 段階品質レベル分類を実行する
2. WHEN 過学習を検出する THEN システム SHALL 訓練・テストデータの性能差を監視する
3. WHEN 実用化準備状況を判定する THEN システム SHALL 即座実行/慎重実行/要考慮/改善必要の 4 段階評価を行う
4. WHEN 商品カテゴリ別分析を実行する THEN システム SHALL 野菜/肉類/魚類/その他の成功率を算出する

### Requirement 7

**User Story:** As a レポート作成者, I want 包括的な分析レポートを自動生成できる, so that ステークホルダーに結果を効果的に伝達できる

#### Acceptance Criteria

1. WHEN 分析レポートを生成する THEN システム SHALL Markdown 形式で構造化された詳細レポートを作成する
2. WHEN 結果を保存する THEN システム SHALL CSV 形式でモデル性能・特徴量重要度・品質評価結果を保存する
3. WHEN 改善効果を定量化する THEN システム SHALL 特徴量追加による性能改善を定量的に分析する
4. WHEN 実用化提案を作成する THEN システム SHALL 段階的導入計画と期待効果を提示する

### Requirement 8

**User Story:** As a システム運用者, I want 堅牢で保守可能なシステムを利用できる, so that 継続的な分析業務を効率化できる

#### Acceptance Criteria

1. WHEN エラーが発生する THEN システム SHALL 適切なログメッセージを出力し処理を継続する
2. WHEN 設定を管理する THEN システム SHALL 設定ファイルによるパラメータ管理を提供する
3. WHEN 結果を管理する THEN システム SHALL data/processed、models、reports ディレクトリに適切に保存する
4. WHEN システムを拡張する THEN システム SHALL モジュール化された設計により新機能追加を容易にする
