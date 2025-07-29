"""
生鮮食品需要予測・分析システム メインエントリーポイント

このスクリプトは全体の分析ワークフローを実行し、
データ処理から最終レポート生成まで統合的に処理します。

Requirements: 8.1, 8.2, 8.3, 8.4
- 8.1: 適切なエラーハンドリングとログ出力機能
- 8.2: 設定ファイルによるパラメータ管理機能
- 8.3: 適切なディレクトリへの結果保存機能
- 8.4: 包括的なエラーハンドリングとログ出力機能
"""

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.demand_forecasting.core.data_processor import DataProcessor
from src.demand_forecasting.core.demand_analyzer import DemandCurveAnalyzer
from src.demand_forecasting.core.feature_engineer import FeatureEngineer
from src.demand_forecasting.core.model_builder import ModelBuilder
from src.demand_forecasting.reports.report_generator import ReportGenerator
from src.demand_forecasting.utils.config import Config
from src.demand_forecasting.utils.exceptions import (
    ConfigurationError,
    DataProcessingError,
    DemandForecastingError,
    FeatureEngineeringError,
    ModelBuildingError,
    ReportGenerationError,
    VisualizationError,
)
from src.demand_forecasting.utils.logger import Logger
from src.demand_forecasting.utils.quality_evaluator import QualityEvaluator
from src.demand_forecasting.visualization.want_plotter import WantPlotter


class DemandForecastingPipeline:
    """
    需要予測分析パイプライン

    全体の分析ワークフローを統合し、設定ファイルによるパラメータ管理、
    適切なディレクトリへの結果保存、包括的なエラーハンドリングを提供します。
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初期化

        Args:
            config_path: 設定ファイルパス

        Raises:
            ConfigurationError: 設定ファイルの読み込みに失敗した場合
        """
        try:
            # 設定管理の初期化
            self.config = Config(config_path)
            self.logger = Logger(self.config.get_logging_config()).get_logger("main")

            # 実行開始時刻を記録
            self.start_time = datetime.now()
            self.logger.info(f"需要予測分析パイプライン初期化開始: {self.start_time}")

            # 必要なディレクトリの作成
            self._ensure_directories()

            # 設定値の検証
            self._validate_configuration()

            # コンポーネント初期化
            self._initialize_components()

            self.logger.info("需要予測分析パイプライン初期化完了")

        except Exception as e:
            error_msg = f"パイプライン初期化エラー: {e}"
            if hasattr(self, "logger"):
                self.logger.error(error_msg)
                self.logger.error(f"詳細: {traceback.format_exc()}")
            else:
                print(f"❌ {error_msg}")
                print(f"詳細: {traceback.format_exc()}")
            raise ConfigurationError(error_msg) from e

    def _ensure_directories(self):
        """必要なディレクトリを作成"""
        try:
            # 基本ディレクトリ
            required_dirs = [
                "data/processed",
                "models",
                "reports",
                "output/visualizations",
                "logs",
            ]

            # 設定ファイルから追加ディレクトリを取得
            viz_config = self.config.get_visualization_config()
            if viz_config.get("output_dir"):
                required_dirs.append(viz_config["output_dir"])

            processed_path = self.config.get("data.processed_data_path")
            if processed_path:
                required_dirs.append(processed_path)

            # ディレクトリ作成
            for dir_path in required_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

            self.logger.info(f"必要なディレクトリを確認/作成しました: {required_dirs}")

        except Exception as e:
            raise ConfigurationError(f"ディレクトリ作成エラー: {e}") from e

    def _validate_configuration(self):
        """設定値の検証"""
        try:
            # 必須設定項目の確認
            required_configs = [
                ("data.raw_data_path", "生データファイルパス"),
                ("data.encoding", "文字エンコーディング"),
                ("model.algorithm", "モデルアルゴリズム"),
                ("quality.thresholds", "品質閾値"),
            ]

            missing_configs = []
            for config_key, description in required_configs:
                if self.config.get(config_key) is None:
                    missing_configs.append(f"{config_key} ({description})")

            if missing_configs:
                raise ConfigurationError(f"必須設定項目が不足: {', '.join(missing_configs)}")

            # データファイルの存在確認
            raw_data_path = self.config.get("data.raw_data_path")
            if not Path(raw_data_path).exists():
                raise ConfigurationError(f"生データファイルが見つかりません: {raw_data_path}")

            self.logger.info("設定値の検証が完了しました")

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"設定値検証エラー: {e}") from e

    def _initialize_components(self):
        """コンポーネントの初期化"""
        try:
            self.logger.info("コンポーネント初期化開始")

            # 各コンポーネントを順次初期化
            components = [
                ("data_processor", DataProcessor),
                ("feature_engineer", FeatureEngineer),
                ("model_builder", ModelBuilder),
                ("demand_analyzer", DemandCurveAnalyzer),
                ("quality_evaluator", QualityEvaluator),
                ("plotter", WantPlotter),
                ("report_generator", ReportGenerator),
            ]

            for component_name, component_class in components:
                try:
                    setattr(self, component_name, component_class(self.config))
                    self.logger.debug(f"{component_name} 初期化完了")
                except Exception as e:
                    raise ConfigurationError(f"{component_name} 初期化エラー: {e}") from e

            self.logger.info("全コンポーネント初期化完了")

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"コンポーネント初期化エラー: {e}") from e

    def run_full_analysis(
        self, target_products: Optional[List[str]] = None, max_products: int = 10
    ) -> Dict[str, Any]:
        """
        全体分析を実行

        設定ファイルによるパラメータ管理、適切なディレクトリへの結果保存、
        包括的なエラーハンドリングを提供します。

        Args:
            target_products: 対象商品リスト
            max_products: 最大処理商品数

        Returns:
            分析結果辞書

        Raises:
            DemandForecastingError: 分析処理中にエラーが発生した場合
        """
        analysis_start_time = datetime.now()
        self.logger.info(f"需要予測分析パイプライン開始: {analysis_start_time}")

        # 分析結果を格納する変数を初期化
        analysis_results = []
        quality_report = {}
        visualization_files = []
        report_files = []

        try:
            # ステップ1: データ読み込みと前処理
            self.logger.info("=" * 50)
            self.logger.info("ステップ1: データ読み込みと前処理")
            self.logger.info("=" * 50)

            try:
                raw_data = self.data_processor.load_raw_data()
                self.logger.info(f"生データ読み込み完了: {len(raw_data)} レコード")

                clean_data = self.data_processor.clean_data(raw_data)
                self.logger.info(f"データクリーニング完了: {len(clean_data)} レコード")

                # 前処理済みデータを保存
                processed_data_path = Path(
                    self.config.get("data.processed_data_path", "data/processed")
                )
                processed_file = (
                    processed_data_path
                    / f"cleaned_data_{analysis_start_time.strftime('%Y%m%d_%H%M%S')}.csv"
                )
                clean_data.to_csv(processed_file, index=False, encoding="utf-8")
                self.logger.info(f"前処理済みデータ保存: {processed_file}")

            except Exception as e:
                raise DataProcessingError(f"データ処理ステップでエラー: {e}") from e

            # ステップ2: 特徴量エンジニアリング
            self.logger.info("=" * 50)
            self.logger.info("ステップ2: 特徴量エンジニアリング")
            self.logger.info("=" * 50)

            try:
                baseline_features = self.feature_engineer.create_baseline_features(clean_data)
                self.logger.info(
                    f"ベースライン特徴量生成完了: {baseline_features.shape[1]} 特徴量"
                )

                time_features = self.feature_engineer.add_time_features(baseline_features)
                self.logger.info(f"時間特徴量追加完了: {time_features.shape[1]} 特徴量")

                weather_features = self.feature_engineer.integrate_weather_features(time_features)
                self.logger.info(f"気象特徴量統合完了: {weather_features.shape[1]} 特徴量")

                # 高度な時系列特徴量追加
                final_features = self.feature_engineer.add_advanced_time_series_features(
                    weather_features
                )
                self.logger.info(f"最終特徴量セット完成: {final_features.shape[1]} 特徴量")

                # 特徴量データを保存
                features_file = (
                    processed_data_path
                    / f"features_{analysis_start_time.strftime('%Y%m%d_%H%M%S')}.csv"
                )
                final_features.to_csv(features_file, index=False, encoding="utf-8")
                self.logger.info(f"特徴量データ保存: {features_file}")

            except Exception as e:
                raise FeatureEngineeringError(
                    f"特徴量エンジニアリングステップでエラー: {e}"
                ) from e

            # ステップ3: 分析対象商品の決定
            self.logger.info("=" * 50)
            self.logger.info("ステップ3: 分析対象商品の決定")
            self.logger.info("=" * 50)

            try:
                if target_products is None:
                    target_products = self.data_processor.stratified_product_sampling(
                        final_features, max_products=max_products
                    )

                self.logger.info(f"分析対象商品決定: {len(target_products)} 商品")
                for i, product in enumerate(target_products, 1):
                    self.logger.info(f"  {i:2d}. {product}")

            except Exception as e:
                raise DataProcessingError(f"商品選択ステップでエラー: {e}") from e

            # ステップ4: 商品別分析実行
            self.logger.info("=" * 50)
            self.logger.info("ステップ4: 商品別分析実行")
            self.logger.info("=" * 50)

            successful_analyses = 0
            failed_analyses = 0

            for i, product in enumerate(target_products, 1):
                self.logger.info(f"商品分析 {i}/{len(target_products)}: {product}")

                try:
                    # 商品データを抽出
                    product_data = final_features[final_features["商品名称"] == product].copy()

                    if len(product_data) < 100:
                        self.logger.warning(
                            f"データ不足により{product}をスキップ (レコード数: {len(product_data)})"
                        )
                        failed_analyses += 1
                        continue

                    # 時系列品質チェック
                    if not self._validate_time_series_quality(product_data, product):
                        self.logger.warning(f"時系列品質不良により{product}をスキップ")
                        failed_analyses += 1
                        continue

                    result = self._analyze_single_product(product, product_data, final_features)
                    if result:
                        analysis_results.append(result)
                        successful_analyses += 1
                        self.logger.info(
                            f"  ✅ {product}: R²={result.get('test_metrics', {}).get('r2_score', 0):.3f}"
                        )
                    else:
                        failed_analyses += 1
                        self.logger.warning(f"  ❌ {product}: 分析結果なし")

                except Exception as e:
                    failed_analyses += 1
                    self.logger.error(f"  ❌ {product}: 分析エラー - {e}")
                    self.logger.debug(f"詳細エラー: {traceback.format_exc()}")
                    continue

            self.logger.info(f"商品別分析完了: 成功={successful_analyses}, 失敗={failed_analyses}")

            # ステップ5: 品質評価
            self.logger.info("=" * 50)
            self.logger.info("ステップ5: 品質評価")
            self.logger.info("=" * 50)

            try:
                quality_report = self.quality_evaluator.create_quality_report(analysis_results)
                self.logger.info("品質評価完了")

            except Exception as e:
                self.logger.error(f"品質評価エラー: {e}")
                # 品質評価が失敗しても分析は継続
                quality_report = {"summary": {"success_rate": 0.0, "average_r2": 0.0}}

            # ステップ6: 可視化生成
            self.logger.info("=" * 50)
            self.logger.info("ステップ6: 可視化生成")
            self.logger.info("=" * 50)

            try:
                visualization_files = self._generate_visualizations(
                    analysis_results, quality_report
                )
                self.logger.info(f"可視化生成完了: {len(visualization_files)} ファイル")

            except Exception as e:
                self.logger.error(f"可視化生成エラー: {e}")
                self.logger.debug(f"詳細エラー: {traceback.format_exc()}")
                # 可視化が失敗しても分析は継続
                visualization_files = []

            # ステップ7: レポート生成
            self.logger.info("=" * 50)
            self.logger.info("ステップ7: レポート生成")
            self.logger.info("=" * 50)

            try:
                report_files = self._generate_reports(analysis_results, quality_report)
                self.logger.info(f"レポート生成完了: {len(report_files)} ファイル")

            except Exception as e:
                self.logger.error(f"レポート生成エラー: {e}")
                self.logger.debug(f"詳細エラー: {traceback.format_exc()}")
                # レポート生成が失敗しても分析は継続
                report_files = []

            # 結果統合
            analysis_end_time = datetime.now()
            execution_time = (analysis_end_time - analysis_start_time).total_seconds()

            final_results = {
                "analysis_results": analysis_results,
                "quality_report": quality_report,
                "visualization_files": visualization_files,
                "report_files": report_files,
                "execution_info": {
                    "start_time": analysis_start_time.isoformat(),
                    "end_time": analysis_end_time.isoformat(),
                    "execution_time_seconds": execution_time,
                    "successful_analyses": successful_analyses,
                    "failed_analyses": failed_analyses,
                },
                "summary": {
                    "total_products_analyzed": len(analysis_results),
                    "success_rate": quality_report.get("summary", {}).get("success_rate", 0.0),
                    "average_r2": quality_report.get("summary", {}).get("average_r2", 0.0),
                },
            }

            self.logger.info("=" * 50)
            self.logger.info("需要予測分析パイプライン完了")
            self.logger.info("=" * 50)
            self._log_summary(final_results)

            return final_results

        except Exception as e:
            # 包括的なエラーハンドリング
            error_msg = f"パイプライン実行エラー: {e}"
            self.logger.error(error_msg)
            self.logger.error(f"詳細エラー: {traceback.format_exc()}")

            # 部分的な結果でも返却
            partial_results = {
                "analysis_results": analysis_results,
                "quality_report": quality_report,
                "visualization_files": visualization_files,
                "report_files": report_files,
                "error": error_msg,
                "summary": {
                    "total_products_analyzed": len(analysis_results),
                    "success_rate": 0.0,
                    "average_r2": 0.0,
                },
            }

            # エラーでも可能な限り結果を保存
            try:
                if analysis_results:
                    self.logger.info("エラー発生時の部分的結果を保存中...")
                    self._generate_reports(analysis_results, quality_report or {})
            except Exception as save_error:
                self.logger.error(f"部分的結果保存エラー: {save_error}")

            raise DemandForecastingError(error_msg) from e

    def _analyze_single_product(
        self, product: str, product_data: pd.DataFrame, full_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """単一商品の分析"""
        try:
            # 特徴量選択
            target_column = "数量"
            if target_column not in product_data.columns:
                self.logger.warning(f"{product}: ターゲット列'{target_column}'が見つかりません")
                return None

            # 強化された特徴量選択を実行
            selected_features = self.feature_engineer.select_features(
                product_data, target_column, max_features=25
            )

            if len(selected_features) < 3:
                self.logger.warning(
                    f"{product}: 特徴量選択後の数が不足 ({len(selected_features)}個)"
                )
                return None

            X = product_data[selected_features]
            y = product_data[target_column]

            self.logger.info(f"{product}: 選択特徴量数={len(selected_features)}")

            # Phase 3: アンサンブル機械学習モデル構築
            model_results = self.model_builder.train_with_cv(X, y, model_type="ensemble")

            # 品質レベル評価
            quality_level = self.quality_evaluator.evaluate_quality_level(
                model_results["test_metrics"]["r2_score"]
            )

            # 実用化準備状況評価
            implementation_readiness = self.quality_evaluator.assess_implementation_readiness(
                model_results["test_metrics"]
            )

            # 需要曲線分析
            demand_results = None
            try:
                demand_results = self.demand_analyzer.analyze_demand_curve(product_data, product)
            except Exception as e:
                self.logger.warning(f"{product}: 需要曲線分析失敗: {e}")

            # 商品カテゴリ取得
            category = (
                product_data["商品カテゴリ"].iloc[0]
                if "商品カテゴリ" in product_data.columns
                else "その他"
            )

            # 結果統合
            result = {
                "product_name": product,
                "category": category,
                "quality_level": quality_level,
                "implementation_readiness": implementation_readiness,
                "test_metrics": model_results["test_metrics"],
                "cv_scores": model_results["cv_scores"],
                "overfitting_score": model_results["overfitting_score"],
                "feature_importance": model_results["feature_importance"],
                "model": model_results["model"],
                "feature_names": model_results["feature_names"],
            }

            if demand_results:
                result["demand_results"] = demand_results

            return result

        except Exception as e:
            self.logger.error(f"商品{product}の分析エラー: {e}")
            return None

    def _generate_visualizations(
        self, analysis_results: List[Dict[str, Any]], quality_report: Dict[str, Any]
    ) -> List[str]:
        """
        可視化を生成

        適切なディレクトリへの結果保存とエラーハンドリングを提供します。
        """
        visualization_files = []

        if not analysis_results:
            self.logger.warning("分析結果がないため可視化をスキップします")
            return visualization_files

        try:
            # 可視化出力ディレクトリの確保
            viz_config = self.config.get_visualization_config()
            output_dir = Path(viz_config.get("output_dir", "output/visualizations"))
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 品質ダッシュボード
            try:
                quality_dashboard_path = self.plotter.create_quality_dashboard(
                    quality_report.get("summary", {})
                )
                if quality_dashboard_path:
                    visualization_files.append(quality_dashboard_path)
                    self.logger.info(f"品質ダッシュボード生成: {quality_dashboard_path}")
            except Exception as e:
                self.logger.error(f"品質ダッシュボード生成エラー: {e}")

            # 個別商品のプロット（上位5商品のみ）
            sorted_results = sorted(
                analysis_results,
                key=lambda x: x.get("test_metrics", {}).get("r2_score", 0),
                reverse=True,
            )[:5]

            self.logger.info(f"上位{len(sorted_results)}商品の個別プロット生成中...")

            for i, result in enumerate(sorted_results, 1):
                product_name = result.get("product_name", f"product_{i}")

                try:
                    # 需要曲線プロット
                    if "demand_results" in result:
                        demand_plot_path = self.plotter.create_demand_curve_plot(
                            result["demand_results"]
                        )
                        if demand_plot_path:
                            visualization_files.append(demand_plot_path)
                            self.logger.info(f"需要曲線プロット生成: {product_name}")
                except Exception as e:
                    self.logger.error(f"需要曲線プロット生成エラー ({product_name}): {e}")

                try:
                    # 特徴量重要度プロット
                    if "feature_importance" in result:
                        importance_plot_path = self.plotter.create_feature_importance_plot(
                            result["feature_importance"], product_name
                        )
                        if importance_plot_path:
                            visualization_files.append(importance_plot_path)
                            self.logger.info(f"特徴量重要度プロット生成: {product_name}")
                except Exception as e:
                    self.logger.error(f"特徴量重要度プロット生成エラー ({product_name}): {e}")

            # 可視化ファイルリストを保存
            if visualization_files:
                viz_list_file = output_dir / f"visualization_files_{timestamp}.txt"
                with open(viz_list_file, "w", encoding="utf-8") as f:
                    f.write(f"可視化ファイル一覧 - {timestamp}\n")
                    f.write("=" * 50 + "\n")
                    for i, file_path in enumerate(visualization_files, 1):
                        f.write(f"{i:2d}. {file_path}\n")

                self.logger.info(f"可視化ファイル一覧保存: {viz_list_file}")

        except Exception as e:
            error_msg = f"可視化生成エラー: {e}"
            self.logger.error(error_msg)
            self.logger.debug(f"詳細エラー: {traceback.format_exc()}")
            raise VisualizationError(error_msg) from e

        return visualization_files

    def _generate_reports(
        self, analysis_results: List[Dict[str, Any]], quality_report: Dict[str, Any]
    ) -> List[str]:
        """
        レポートを生成

        適切なディレクトリへの結果保存とエラーハンドリングを提供します。
        """
        report_files = []

        if not analysis_results:
            self.logger.warning("分析結果がないためレポート生成をスキップします")
            return report_files

        try:
            # レポート出力ディレクトリの確保
            reports_dir = Path("reports")
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Markdownレポート生成
            try:
                self.logger.info("Markdownレポート生成中...")
                markdown_report = self.report_generator.generate_markdown_report(
                    analysis_results, quality_report
                )
                markdown_path = self.report_generator.save_markdown_report(markdown_report)
                if markdown_path:
                    report_files.append(markdown_path)
                    self.logger.info(f"Markdownレポート生成完了: {markdown_path}")
            except Exception as e:
                self.logger.error(f"Markdownレポート生成エラー: {e}")

            # CSVレポート生成
            try:
                self.logger.info("CSVレポート生成中...")
                csv_paths = self.report_generator.generate_csv_reports(analysis_results)
                if csv_paths:
                    report_files.extend(csv_paths)
                    self.logger.info(f"CSVレポート生成完了: {len(csv_paths)} ファイル")
                    for csv_path in csv_paths:
                        self.logger.info(f"  - {csv_path}")
            except Exception as e:
                self.logger.error(f"CSVレポート生成エラー: {e}")

            # モデルファイルの保存
            try:
                models_dir = Path("models")
                models_dir.mkdir(parents=True, exist_ok=True)

                model_files = []
                for i, result in enumerate(analysis_results):
                    if "model" in result and result["model"] is not None:
                        product_name = result.get("product_name", f"product_{i}")
                        # ファイル名に使用できない文字を置換
                        safe_name = "".join(
                            c if c.isalnum() or c in "._-" else "_" for c in product_name
                        )
                        model_file = models_dir / f"model_{safe_name}_{timestamp}.pkl"

                        try:
                            import pickle

                            with open(model_file, "wb") as f:
                                pickle.dump(result["model"], f)
                            model_files.append(str(model_file))
                            self.logger.debug(f"モデル保存: {model_file}")
                        except Exception as e:
                            self.logger.warning(f"モデル保存エラー ({product_name}): {e}")

                if model_files:
                    self.logger.info(f"モデルファイル保存完了: {len(model_files)} ファイル")
                    report_files.extend(model_files)

            except Exception as e:
                self.logger.error(f"モデル保存エラー: {e}")

            # レポートファイルリストを保存
            if report_files:
                report_list_file = reports_dir / f"report_files_{timestamp}.txt"
                with open(report_list_file, "w", encoding="utf-8") as f:
                    f.write(f"レポートファイル一覧 - {timestamp}\n")
                    f.write("=" * 50 + "\n")
                    for i, file_path in enumerate(report_files, 1):
                        f.write(f"{i:2d}. {file_path}\n")

                self.logger.info(f"レポートファイル一覧保存: {report_list_file}")

        except Exception as e:
            error_msg = f"レポート生成エラー: {e}"
            self.logger.error(error_msg)
            self.logger.debug(f"詳細エラー: {traceback.format_exc()}")
            raise ReportGenerationError(error_msg) from e

        return report_files

    def _validate_time_series_quality(self, product_data: pd.DataFrame, product_name: str) -> bool:
        """
        時系列データの品質を検証

        Args:
            product_data: 商品データ
            product_name: 商品名

        Returns:
            品質が十分ならTrue、不十分ならFalse
        """
        try:
            # 日付列の存在確認
            if "年月日" not in product_data.columns:
                return False

            # 日付順にソート
            product_data_sorted = product_data.sort_values("年月日")

            # 1. 時系列連続性チェック：データの期間をチェック
            date_range = (
                product_data_sorted["年月日"].max() - product_data_sorted["年月日"].min()
            ).days
            if date_range < 90:  # 3ヶ月未満のデータは除外
                self.logger.warning(f"{product_name}: データ期間が短すぎます ({date_range}日)")
                return False

            # 2. データ密度チェック：期間に対するレコード数の密度
            expected_records = date_range * 0.3  # 3日に1回程度の売上を期待
            if len(product_data) < expected_records:
                self.logger.warning(
                    f"{product_name}: データ密度が低すぎます (期間:{date_range}日, レコード数:{len(product_data)})"
                )
                return False

            # 3. 季節性パターンチェック：月次売上の分散をチェック
            if "month" in product_data.columns or "月" in product_data.columns:
                month_col = "month" if "month" in product_data.columns else "月"
                monthly_sales = product_data.groupby(month_col)["数量"].sum()

                # 月次売上の変動係数（CV）をチェック
                cv = (
                    monthly_sales.std() / monthly_sales.mean()
                    if monthly_sales.mean() > 0
                    else float("inf")
                )
                if cv > 2.0:  # 変動係数が2.0を超える場合は不安定すぎる
                    self.logger.warning(
                        f"{product_name}: 月次売上の不安定性が高すぎます (CV: {cv:.2f})"
                    )
                    return False

            # 4. ゼロ売上期間チェック：連続するゼロ売上の期間をチェック
            zero_sales_ratio = (product_data["数量"] == 0).sum() / len(product_data)
            if zero_sales_ratio > 0.5:  # ゼロ売上が50%を超える場合
                self.logger.warning(
                    f"{product_name}: ゼロ売上の比率が高すぎます ({zero_sales_ratio:.1%})"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"{product_name}: 時系列品質検証中にエラー: {e}")
            return False

    def _log_summary(self, results: Dict[str, Any]):
        """
        結果サマリーをログ出力

        包括的な実行結果の要約を提供します。
        """
        summary = results["summary"]
        execution_info = results.get("execution_info", {})

        self.logger.info("=" * 60)
        self.logger.info("🎯 需要予測分析パイプライン 実行結果サマリー")
        self.logger.info("=" * 60)

        # 実行情報
        if execution_info:
            self.logger.info("📅 実行情報:")
            if "start_time" in execution_info:
                self.logger.info(f"   開始時刻: {execution_info['start_time']}")
            if "end_time" in execution_info:
                self.logger.info(f"   終了時刻: {execution_info['end_time']}")
            if "execution_time_seconds" in execution_info:
                exec_time = execution_info["execution_time_seconds"]
                hours = int(exec_time // 3600)
                minutes = int((exec_time % 3600) // 60)
                seconds = int(exec_time % 60)
                self.logger.info(f"   実行時間: {hours:02d}:{minutes:02d}:{seconds:02d}")

            if "successful_analyses" in execution_info and "failed_analyses" in execution_info:
                total = execution_info["successful_analyses"] + execution_info["failed_analyses"]
                success_rate = (
                    execution_info["successful_analyses"] / total * 100 if total > 0 else 0
                )
                self.logger.info(
                    f"   成功/失敗: {execution_info['successful_analyses']}/{execution_info['failed_analyses']} ({success_rate:.1f}%)"
                )

        # 分析結果
        self.logger.info("📊 分析結果:")
        self.logger.info(f"   分析商品数: {summary['total_products_analyzed']}")
        self.logger.info(f"   成功率: {summary['success_rate']*100:.1f}%")
        self.logger.info(f"   平均R²スコア: {summary['average_r2']:.3f}")

        # 品質レベル分布
        quality_report = results.get("quality_report", {})
        if "quality_distribution" in quality_report:
            self.logger.info("   品質レベル分布:")
            for level, count in quality_report["quality_distribution"].items():
                self.logger.info(f"     {level}: {count} 商品")

        # 出力ファイル
        self.logger.info("📁 出力ファイル:")
        self.logger.info(f"   可視化ファイル数: {len(results['visualization_files'])}")
        self.logger.info(f"   レポートファイル数: {len(results['report_files'])}")

        # 設定情報
        self.logger.info("⚙️  使用設定:")
        model_config = self.config.get_model_config()
        self.logger.info(f"   モデルアルゴリズム: {model_config.get('algorithm', 'N/A')}")
        self.logger.info(f"   交差検証フォールド数: {model_config.get('cv_folds', 'N/A')}")

        quality_config = self.config.get_quality_config()
        thresholds = quality_config.get("thresholds", {})
        self.logger.info(
            f"   品質閾値: Premium≥{thresholds.get('premium', 'N/A')}, Standard≥{thresholds.get('standard', 'N/A')}, Basic≥{thresholds.get('basic', 'N/A')}"
        )

        # エラー情報
        if "error" in results:
            self.logger.error("❌ エラー情報:")
            self.logger.error(f"   {results['error']}")

        self.logger.info("=" * 60)


def main():
    """
    メイン関数

    コマンドライン引数を解析し、需要予測分析パイプラインを実行します。
    包括的なエラーハンドリングとログ出力を提供します。
    """
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="生鮮食品需要予測・分析システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python src/main.py                                    # デフォルト設定で実行
  python src/main.py --config config/custom.yaml       # カスタム設定で実行
  python src/main.py --products "商品A" "商品B"         # 特定商品のみ分析
  python src/main.py --max-products 20 --verbose       # 詳細ログで20商品まで分析
        """,
    )

    parser.add_argument(
        "--config", type=str, help="設定ファイルパス (デフォルト: config/config.yaml)"
    )
    parser.add_argument(
        "--products", type=str, nargs="+", help="対象商品リスト (指定しない場合は自動選択)"
    )
    parser.add_argument(
        "--max-products", type=int, default=10, help="最大処理商品数 (デフォルト: 10)"
    )
    parser.add_argument("--verbose", action="store_true", help="詳細ログ出力を有効にする")
    parser.add_argument(
        "--dry-run", action="store_true", help="設定確認のみ実行（実際の分析は行わない）"
    )

    args = parser.parse_args()

    # 実行開始時刻
    start_time = datetime.now()

    # 初期化段階のエラーハンドリング
    pipeline = None
    try:
        print("🚀 生鮮食品需要予測・分析システム")
        print("=" * 50)
        print(f"実行開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if args.config:
            print(f"設定ファイル: {args.config}")
        else:
            print("設定ファイル: config/config.yaml (デフォルト)")

        if args.products:
            print(f"対象商品: {', '.join(args.products)}")
        else:
            print(f"対象商品: 自動選択 (最大{args.max_products}商品)")

        print("=" * 50)

        # パイプライン初期化
        print("🔧 システム初期化中...")
        pipeline = DemandForecastingPipeline(args.config)

        # 詳細ログ設定
        if args.verbose:
            pipeline.logger.setLevel("DEBUG")
            print("📝 詳細ログモードを有効にしました")

        # Dry run モード
        if args.dry_run:
            print("🔍 設定確認モード (実際の分析は実行されません)")
            print("\n✅ 設定確認完了!")
            print("設定に問題がないことを確認しました。")
            return

        print("✅ システム初期化完了")
        print()

        # 分析実行
        print("🔄 分析実行中...")
        results = pipeline.run_full_analysis(
            target_products=args.products, max_products=args.max_products
        )

        # 実行時間計算
        end_time = datetime.now()
        execution_time = end_time - start_time

        # 結果表示
        print("\n" + "=" * 60)
        print("🎉 分析完了!")
        print("=" * 60)

        summary = results["summary"]
        execution_info = results.get("execution_info", {})

        print(f"📊 分析商品数: {summary['total_products_analyzed']}")
        print(f"📈 成功率: {summary['success_rate']*100:.1f}%")
        print(f"🎯 平均R²スコア: {summary['average_r2']:.3f}")
        print(f"⏱️  実行時間: {execution_time}")
        print(f"📁 レポートファイル: {len(results['report_files'])}件")
        print(f"📊 可視化ファイル: {len(results['visualization_files'])}件")

        # 品質レベル分布表示
        quality_report = results.get("quality_report", {})
        if "quality_distribution" in quality_report:
            print("\n📋 品質レベル分布:")
            for level, count in quality_report["quality_distribution"].items():
                print(f"   {level}: {count} 商品")

        # 生成ファイル一覧
        all_files = results["report_files"] + results["visualization_files"]
        if all_files:
            print(f"\n📄 生成されたファイル ({len(all_files)}件):")
            for i, file_path in enumerate(all_files, 1):
                print(f"  {i:2d}. {file_path}")

        # 成功メッセージ
        print("\n🎊 すべての処理が正常に完了しました!")

        # 終了コード
        if summary["success_rate"] > 0.5:
            sys.exit(0)  # 成功
        else:
            print("⚠️  成功率が低いため、設定やデータを確認してください。")
            sys.exit(2)  # 警告

    except KeyboardInterrupt:
        print("\n\n⏹️  ユーザーによって処理が中断されました")
        sys.exit(130)

    except ConfigurationError as e:
        print(f"\n❌ 設定エラー: {e}")
        print("💡 設定ファイルの内容を確認してください。")
        sys.exit(1)

    except (DataProcessingError, FeatureEngineeringError, ModelBuildingError) as e:
        print(f"\n❌ 分析処理エラー: {e}")
        print("💡 データファイルの内容や形式を確認してください。")
        if pipeline and hasattr(pipeline, "logger"):
            pipeline.logger.error(f"分析処理エラー: {e}")
        sys.exit(1)

    except (VisualizationError, ReportGenerationError) as e:
        print(f"\n⚠️  出力生成エラー: {e}")
        print("💡 分析は完了しましたが、一部の出力生成に失敗しました。")
        if pipeline and hasattr(pipeline, "logger"):
            pipeline.logger.warning(f"出力生成エラー: {e}")
        sys.exit(2)

    except DemandForecastingError as e:
        print(f"\n❌ システムエラー: {e}")
        if pipeline and hasattr(pipeline, "logger"):
            pipeline.logger.error(f"システムエラー: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n💥 予期しないエラーが発生しました: {e}")
        print(f"詳細: {traceback.format_exc()}")
        if pipeline and hasattr(pipeline, "logger"):
            pipeline.logger.critical(f"予期しないエラー: {e}")
            pipeline.logger.critical(f"詳細: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
