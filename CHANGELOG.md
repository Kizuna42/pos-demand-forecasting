# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 実装タスクの段階的実行
- ユニットテスト・インテグレーションテスト
- API ドキュメント生成
- パフォーマンス最適化

### Changed

- TBD

### Fixed

- TBD

## [0.1.0] - 2024-12-XX

### Added

- 初期プロジェクト構造作成
- 要件定義書（8 つの主要要件）
  - データ前処理・クリーニング
  - 段階的特徴量エンジニアリング
  - 堅牢な ML モデル構築
  - 需要曲線分析・価格最適化
  - Want 風可視化システム
  - モデル品質評価システム
  - 包括的レポート生成
  - システム運用・保守性
- システム設計書
  - モジュール化されたアーキテクチャ
  - DataProcessor, FeatureEngineer, ModelBuilder, DemandCurveAnalyzer, QualityEvaluator
  - データモデル定義（ProductData, ModelResult, DemandCurveResult, AnalysisReport）
  - エラーハンドリング戦略
  - テスト戦略・設定管理・パフォーマンス・セキュリティ考慮
- 実装タスクリスト（11 の段階的タスク）
- プロトタイプ分析ノートブック
  - 完全版時刻混雑度外気温統合需要曲線分析
  - 5 軸比較分析システム
  - Savitzky-Golay 平滑化による外れ値処理
  - 外気温統合効果の定量化
- want_style_plotter.py
  - 需要曲線分析プロット
  - 特徴量重要度分析プロット
  - 品質ダッシュボードプロット
- 生鮮食品データ（2024 年 1 年分）
- GitHub リポジトリ・CI/CD 設定
- 包括的な README・ドキュメント

### Technical Details

- **データ**: Shift-JIS エンコーディング対応
- **機械学習**: RandomForest + 交差検証
- **特徴量**: 時刻・混雑度・外気温統合
- **可視化**: matplotlib + seaborn (want 風スタイル)
- **品質評価**: Premium/Standard/Basic/Rejected 4 段階
- **外れ値処理**: IQR 法 + Savitzky-Golay 平滑化
- **解釈可能性**: 特徴量重要度 + SHAP 値（予定）
