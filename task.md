### 計画概要

- 目的: 需要予測システムの短期的な安定稼働と評価精度の正常化（設定処理の堅牢化、品質評価入力の整合、I/O 安定化、最小限の TDD ワークフロー導入）
- 適用範囲: `src/`、`config/`、`tests/`、`README.md`、`task.md`（長期改善は含めない）

### タスクリスト（短期・重要度順）

- [x] 設定処理の堅牢化（空設定のデフォルトマージ／存在しない設定は例外）

  - **課題**: 空設定で起動不可、存在しない設定パスで挙動が不定
  - **修正案**: デフォルト`config.yaml`とユーザー設定をディープマージ。存在しない設定ファイルは明示的に`FileNotFoundError`
    - [x] 実装: `src/demand_forecasting/utils/config.py`に`_load_yaml`/`_deep_merge`/`_load_and_merge_configs`を追加
    - [x] 検証: `pytest tests/test_end_to_end.py::TestEndToEnd::test_pipeline_error_handling -q`

- [x] 品質評価入力の整合（過学習・CV 指標の受け渡し）

  - **課題**: `assess_implementation_readiness`に`overfitting_score`や`cv_mean_r2`が渡らず、過学習判定が過剰に厳格化
  - **修正案**: `src/main.py`から評価入力に`overfitting_score`・`cv_mean_r2`・`cv_std_r2`を明示的に含める
    - [x] 実装: `src/main.py`で`implementation_metrics={r2, overfitting_score, cv_mean_r2, cv_std_r2}`を渡す
    - [x] 検証: `pytest tests/test_quality_evaluator.py -q`

- [x] 過学習閾値の一貫性（短期はテスト準拠の 0.01）

  - **課題**: `ModelBuilder.detect_overfitting`と`QualityEvaluator`で閾値のデフォルトが不一致（0.10 vs 0.01）
  - **修正案**: `quality.overfitting_threshold`は 0.01 に統一し、参照も同一キーに一本化
    - [x] 実装: `config/config.yaml`、`QualityEvaluator`、`ModelBuilder.detect_overfitting`のデフォルトを 0.01 に整合
    - [x] 検証: `pytest tests/test_model_builder.py::TestModelBuilder::test_detect_overfitting -q`

- [x] 依存関係（オプショナル依存）の明確化

  - **課題**: `xgboost`/`scikit-optimize`の任意依存が運用時に分かりにくい
  - **修正案**: インストールガイドを明記し、必要に応じて`extras`や追加の要件ファイルに分離
    - [x] `requirements.txt`を見直し、README に「任意機能（XGBoost/ベイズ最適化）導入方法」を追記（例: `pip install xgboost scikit-optimize`）
    - [x] 検証: 任意依存なしでもエラーにならず警告ログのみ、任意依存導入時にアンサンブル/最適化が有効化

- [x] 気象 API の簡易リトライで I/O 安定化（短期版）

  - **課題**: ネットワーク起因の一時的失敗に脆弱（単回リクエスト/30 秒タイムアウト）
  - **修正案**: 簡易リトライ（例: 最大 3 回、指数バックオフ）を`_fetch_weather_data`に追加
    - [x] 実装: `src/demand_forecasting/core/feature_engineer.py`の`requests.get`をリトライ
    - [x] 検証: 全テスト通過（I/O 警告のみ）

- [x] 未使用テンプレートの整理（混乱防止）
  - **課題**: `pos2/`はテンプレ状態で本システムと独立しており、利用者を混乱させる
  - **修正案**: 非使用であれば削除、残す場合は README に用途（サンプル/テンプレ）を明記
    - [x] `pos2/`の現利用状況を確認し、不要なら削除または`docs/`へ移動。残す場合は README に注記を追加
    - [x] 検証: README 更新で利用者の導線が明確になっていることを確認

### テスト駆動開発（TDD）を取り入れた短期ワークフロー

- [ ] 失敗するテストを先に書く（仕様化）
  - 例: 「空設定を与えた場合でもデフォルト設定で初期化できる」テストを`tests/test_end_to_end.py`へ追加
- [ ] 最小実装でテストを通す
  - 例: `Config`のデフォルトマージ実装、`main`の初期ファイル警告化
- [ ] リファクタリング（重複の除去・命名改善・責務分離）
  - 例: 閾値参照の一本化、`_validate_configuration`の責務明確化
- [ ] 回帰防止のためテストを拡充
  - 例: 結果ゼロ時の Markdown 生成テスト、気象 API リトライのフォールバック確認テスト
- [ ] CI の高速検証フェーズを分離
  - 例: `-m "not slow"`タグでユニット・軽量統合テスト →PR 必須、E2E は夜間実行

### 実行コマンド（検証）

- [ ] 単体/重点テスト
  - `pytest tests/test_quality_evaluator.py -q`
  - `pytest tests/test_model_builder.py::TestModelBuilder::test_detect_overfitting -q`
- [ ] E2E/統合テスト
  - `pytest tests/test_end_to_end.py -q`
- [ ] 全体テスト
  - `pytest tests -q`
