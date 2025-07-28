"""
生鮮食品需要予測・分析システム メインエントリーポイント

このスクリプトは全体の分析ワークフローを実行し、
データ処理から最終レポート生成まで統合的に処理します。
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from src.demand_forecasting.core.data_processor import DataProcessor
from src.demand_forecasting.core.demand_analyzer import DemandCurveAnalyzer
from src.demand_forecasting.core.feature_engineer import FeatureEngineer
from src.demand_forecasting.core.model_builder import ModelBuilder
from src.demand_forecasting.reports.report_generator import ReportGenerator
from src.demand_forecasting.utils.config import Config
from src.demand_forecasting.utils.logger import Logger
from src.demand_forecasting.utils.quality_evaluator import QualityEvaluator
from src.demand_forecasting.visualization.want_plotter import WantPlotter


class DemandForecastingPipeline:
    """需要予測分析パイプライン"""

    def __init__(self, config_path: str = None):
        """
        初期化

        Args:
            config_path: 設定ファイルパス
        """
        self.config = Config(config_path)
        self.logger = Logger(self.config.get_logging_config()).get_logger("main")

        # コンポーネント初期化
        self.data_processor = DataProcessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_builder = ModelBuilder(self.config)
        self.demand_analyzer = DemandCurveAnalyzer(self.config)
        self.quality_evaluator = QualityEvaluator(self.config)
        self.plotter = WantPlotter(self.config)
        self.report_generator = ReportGenerator(self.config)

    def run_full_analysis(
        self, target_products: List[str] = None, max_products: int = 10
    ) -> Dict[str, Any]:
        """
        全体分析を実行

        Args:
            target_products: 対象商品リスト
            max_products: 最大処理商品数

        Returns:
            分析結果辞書
        """
        self.logger.info("需要予測分析パイプライン開始")

        try:
            # 1. データ読み込みと前処理
            self.logger.info("ステップ1: データ読み込みと前処理")
            raw_data = self.data_processor.load_raw_data()
            clean_data = self.data_processor.clean_data(raw_data)

            # 2. 特徴量エンジニアリング
            self.logger.info("ステップ2: 特徴量エンジニアリング")
            baseline_features = self.feature_engineer.create_baseline_features(clean_data)
            time_features = self.feature_engineer.add_time_features(baseline_features)
            final_features = self.feature_engineer.integrate_weather_features(time_features)

            # 3. 分析対象商品の決定
            if target_products is None:
                # Phase 2: 層化サンプリングで代表商品を選択
                target_products = self.data_processor.stratified_product_sampling(
                    final_features, max_products=max_products
                )

            self.logger.info(f"分析対象商品: {len(target_products)}商品")

            # 4. 商品別分析実行
            analysis_results = []

            for i, product in enumerate(target_products, 1):
                self.logger.info(f"商品分析 {i}/{len(target_products)}: {product}")

                try:
                    # 商品データを抽出
                    product_data = final_features[final_features["商品名称"] == product].copy()

                    if len(product_data) < 100:
                        self.logger.warning(
                            f"データ不足により{product}をスキップ (レコード数: {len(product_data)})"
                        )
                        continue

                    # 時系列連続性・季節性チェック
                    if not self._validate_time_series_quality(product_data, product):
                        self.logger.warning(f"時系列品質不良により{product}をスキップ")
                        continue

                    result = self._analyze_single_product(product, product_data, final_features)
                    if result:
                        analysis_results.append(result)

                except Exception as e:
                    self.logger.error(f"商品{product}の分析に失敗: {e}")
                    continue

            # 5. 品質評価
            self.logger.info("ステップ3: 品質評価")
            quality_report = self.quality_evaluator.create_quality_report(analysis_results)

            # 6. 可視化生成
            self.logger.info("ステップ4: 可視化生成")
            visualization_files = self._generate_visualizations(analysis_results, quality_report)

            # 7. レポート生成
            self.logger.info("ステップ5: レポート生成")
            report_files = self._generate_reports(analysis_results, quality_report)

            # 結果統合
            final_results = {
                "analysis_results": analysis_results,
                "quality_report": quality_report,
                "visualization_files": visualization_files,
                "report_files": report_files,
                "summary": {
                    "total_products_analyzed": len(analysis_results),
                    "success_rate": quality_report.get("summary", {}).get("success_rate", 0.0),
                    "average_r2": quality_report.get("summary", {}).get("average_r2", 0.0),
                },
            }

            self.logger.info("需要予測分析パイプライン完了")
            self._log_summary(final_results)

            return final_results

        except Exception as e:
            self.logger.error(f"パイプライン実行エラー: {e}")
            raise

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

            # 特徴量とターゲットを分離
            feature_columns = product_data.select_dtypes(include=["number"]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)

            if len(feature_columns) < 3:
                self.logger.warning(f"{product}: 特徴量不足")
                return None

            X = product_data[feature_columns]
            y = product_data[target_column]

            # 機械学習モデル構築
            model_results = self.model_builder.train_with_cv(X, y)

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
        """可視化を生成"""
        visualization_files = []

        try:
            # 品質ダッシュボード
            quality_dashboard_path = self.plotter.create_quality_dashboard(
                quality_report.get("summary", {})
            )
            visualization_files.append(quality_dashboard_path)

            # 個別商品のプロット（上位5商品のみ）
            sorted_results = sorted(
                analysis_results,
                key=lambda x: x.get("test_metrics", {}).get("r2_score", 0),
                reverse=True,
            )[:5]

            for result in sorted_results:
                # 需要曲線プロット
                if "demand_results" in result:
                    demand_plot_path = self.plotter.create_demand_curve_plot(
                        result["demand_results"]
                    )
                    visualization_files.append(demand_plot_path)

                # 特徴量重要度プロット
                if "feature_importance" in result:
                    importance_plot_path = self.plotter.create_feature_importance_plot(
                        result["feature_importance"], result["product_name"]
                    )
                    visualization_files.append(importance_plot_path)

        except Exception as e:
            self.logger.error(f"可視化生成エラー: {e}")

        return visualization_files

    def _generate_reports(
        self, analysis_results: List[Dict[str, Any]], quality_report: Dict[str, Any]
    ) -> List[str]:
        """レポートを生成"""
        report_files = []

        try:
            # Markdownレポート
            markdown_report = self.report_generator.generate_markdown_report(
                analysis_results, quality_report
            )
            markdown_path = self.report_generator.save_markdown_report(markdown_report)
            report_files.append(markdown_path)

            # CSVレポート
            csv_paths = self.report_generator.generate_csv_reports(analysis_results)
            report_files.extend(csv_paths)

        except Exception as e:
            self.logger.error(f"レポート生成エラー: {e}")

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
        """結果サマリーをログ出力"""
        summary = results["summary"]

        self.logger.info("=== 分析結果サマリー ===")
        self.logger.info(f"分析商品数: {summary['total_products_analyzed']}")
        self.logger.info(f"成功率: {summary['success_rate']*100:.1f}%")
        self.logger.info(f"平均R²スコア: {summary['average_r2']:.3f}")
        self.logger.info(f"生成可視化ファイル数: {len(results['visualization_files'])}")
        self.logger.info(f"生成レポートファイル数: {len(results['report_files'])}")
        self.logger.info("========================")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="生鮮食品需要予測・分析システム")
    parser.add_argument("--config", type=str, help="設定ファイルパス")
    parser.add_argument("--products", type=str, nargs="+", help="対象商品リスト")
    parser.add_argument("--max-products", type=int, default=10, help="最大処理商品数")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ出力")

    args = parser.parse_args()

    try:
        # パイプライン実行
        pipeline = DemandForecastingPipeline(args.config)

        if args.verbose:
            pipeline.logger.setLevel("DEBUG")

        results = pipeline.run_full_analysis(
            target_products=args.products, max_products=args.max_products
        )

        print("\n✅ 分析完了!")
        print(f"📊 分析商品数: {results['summary']['total_products_analyzed']}")
        print(f"📈 成功率: {results['summary']['success_rate']*100:.1f}%")
        print(f"🎯 平均R²スコア: {results['summary']['average_r2']:.3f}")
        print(f"📁 レポートファイル: {len(results['report_files'])}件")
        print(f"📊 可視化ファイル: {len(results['visualization_files'])}件")

        print("\n📄 生成されたファイル:")
        for file_path in results["report_files"] + results["visualization_files"]:
            print(f"  - {file_path}")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
