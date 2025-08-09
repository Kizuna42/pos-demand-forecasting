from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from ..utils.config import Config
from ..utils.exceptions import ReportGenerationError
from ..utils.logger import Logger


class ReportGenerator:
    """レポート生成クラス"""

    def __init__(self, config: Config):
        """
        初期化

        Args:
            config: 設定管理オブジェクト
        """
        self.config = config
        self.logger = Logger(config.get_logging_config()).get_logger("report_generator")

    def generate_markdown_report(
        self, analysis_results: List[Dict[str, Any]], quality_report: Dict[str, Any]
    ) -> str:
        """
        Markdownレポートを生成

        Args:
            analysis_results: 分析結果リスト
            quality_report: 品質レポート

        Returns:
            Markdownレポート文字列
        """
        try:
            self.logger.info("Markdownレポート生成を開始")

            # レポートヘッダー
            report = self._generate_header()

            # エグゼクティブサマリー
            report += self._generate_executive_summary(quality_report)

            # 全体品質評価
            report += self._generate_quality_overview(quality_report)

            # 個別商品分析結果（無い場合はサマリーのみ）
            report += self._generate_product_analysis(analysis_results)

            # 特徴量改善効果分析
            improvement_analysis = self.analyze_feature_improvement_effects(analysis_results)
            report += self.generate_improvement_analysis_report(improvement_analysis)

            # 改善提案
            report += self._generate_improvement_recommendations(quality_report)

            # 実装計画
            report += self._generate_implementation_plan(analysis_results)

            # 付録
            report += self._generate_appendix(analysis_results)

            self.logger.info("Markdownレポート生成完了")
            return report

        except Exception as e:
            raise ReportGenerationError(f"Markdownレポート生成エラー: {e}")

    def _generate_header(self) -> str:
        """レポートヘッダーを生成"""
        now = datetime.now()

        return f"""# 生鮮食品需要予測・分析レポート

**作成日時**: {now.strftime('%Y年%m月%d日 %H:%M:%S')}
**システム**: 生鮮食品需要予測・分析システム v0.1.0

---

"""

    def _generate_executive_summary(self, quality_report: Dict[str, Any]) -> str:
        """エグゼクティブサマリーを生成"""
        summary = quality_report.get("summary", {})
        total_products = summary.get("total_products", 0)
        success_rate = summary.get("success_rate", 0.0) * 100
        avg_r2 = summary.get("average_r2", 0.0)
        overall_assessment = quality_report.get("overall_assessment", "")

        return f"""## エグゼクティブサマリー

本レポートは{total_products}商品の需要予測モデル構築と分析結果をまとめたものです。

### 主要な成果
- **分析対象商品数**: {total_products}商品
- **成功率**: {success_rate:.1f}% (Premium + Standard品質)
- **平均R²スコア**: {avg_r2:.3f}
- **総合評価**: {overall_assessment}

### ビジネスインパクト
高精度な需要予測により、在庫最適化と収益向上が期待できます。特にPremium品質のモデルは即座に実用化可能であり、価格最適化による収益改善が見込まれます。

---

"""

    def _generate_quality_overview(self, quality_report: Dict[str, Any]) -> str:
        """品質概要を生成"""
        summary = quality_report.get("summary", {})
        quality_dist = summary.get("quality_distribution", {})
        impl_dist = summary.get("implementation_distribution", {})
        category_rates = summary.get("category_success_rates", {})

        # 品質レベル分布テーブル
        quality_table = "| 品質レベル | 商品数 | 割合 |\n|---|---|---|\n"
        total = sum(quality_dist.values()) if quality_dist else 1
        for level, count in quality_dist.items():
            percentage = (count / total) * 100
            quality_table += f"| {level} | {count} | {percentage:.1f}% |\n"

        # 実用化準備状況テーブル
        impl_table = "| 準備状況 | 商品数 |\n|---|---|\n"
        for status, count in impl_dist.items():
            impl_table += f"| {status} | {count} |\n"

        # カテゴリ別成功率テーブル
        category_table = "| カテゴリ | 成功率 |\n|---|---|\n"
        for category, rate in category_rates.items():
            category_table += f"| {category} | {rate*100:.1f}% |\n"

        return f"""## 品質評価概要

### 品質レベル分布
{quality_table}

### 実用化準備状況
{impl_table}

### カテゴリ別成功率
{category_table}

---

"""

    def _generate_product_analysis(self, analysis_results: List[Dict[str, Any]]) -> str:
        """個別商品分析結果を生成"""
        if not analysis_results:
            return "## 個別商品分析結果\n\n分析結果がありません。\n\n---\n\n"

        report = "## 個別商品分析結果\n\n"

        # 上位5商品の詳細レポート
        sorted_results = sorted(
            analysis_results,
            key=lambda x: x.get("test_metrics", {}).get("r2_score", 0),
            reverse=True,
        )

        report += "### 高品質モデル（上位5商品）\n\n"

        for i, result in enumerate(sorted_results[:5], 1):
            product_name = result.get("product_name", "Unknown")
            test_metrics = result.get("test_metrics", {})
            r2_score = test_metrics.get("r2_score", 0.0)
            rmse = test_metrics.get("rmse", 0.0)
            mae = test_metrics.get("mae", 0.0)
            quality_level = result.get("quality_level", "Unknown")

            # 需要曲線分析結果
            demand_results = result.get("demand_results", {})
            optimal_price = demand_results.get("optimal_price", 0)
            current_price = demand_results.get("current_price", 0)
            price_elasticity = demand_results.get("price_elasticity", 0)

            price_change = (
                ((optimal_price - current_price) / current_price * 100) if current_price > 0 else 0
            )

            report += f"""#### {i}. {product_name}

**品質レベル**: {quality_level}
**モデル性能**:
- R²スコア: {r2_score:.4f}
- RMSE: {rmse:.2f}
- MAE: {mae:.2f}

**価格最適化**:
- 現在価格: ¥{current_price:.0f}
- 最適価格: ¥{optimal_price:.0f}
- 価格変更率: {price_change:+.1f}%
- 価格弾力性: {price_elasticity:.3f}

**特徴量重要度トップ3**:
"""

            # 特徴量重要度トップ3
            feature_importance = result.get("feature_importance", {})
            for j, (feature, importance) in enumerate(list(feature_importance.items())[:3], 1):
                report += f"  {j}. {feature}: {importance:.3f}\n"

            report += "\n"

        # 全商品サマリーテーブル
        report += "### 全商品サマリー\n\n"
        report += "| 商品名 | 品質レベル | R²スコア | 最適価格 | 価格変更率 |\n"
        report += "|---|---|---|---|---|\n"

        for result in sorted_results:
            product_name = result.get("product_name", "Unknown")
            quality_level = result.get("quality_level", "Unknown")
            r2_score = result.get("test_metrics", {}).get("r2_score", 0.0)

            demand_results = result.get("demand_results", {})
            optimal_price = demand_results.get("optimal_price", 0)
            current_price = demand_results.get("current_price", 0)
            price_change = (
                ((optimal_price - current_price) / current_price * 100) if current_price > 0 else 0
            )

            report += f"| {product_name} | {quality_level} | {r2_score:.3f} | ¥{optimal_price:.0f} | {price_change:+.1f}% |\n"

            report += "\n---\n\n"
        return report

    def _generate_improvement_recommendations(self, quality_report: Dict[str, Any]) -> str:
        """改善提案を生成"""
        priorities = quality_report.get("improvement_priorities", [])

        report = "## 改善提案\n\n"

        if not priorities:
            report += (
                "現在のシステム品質は良好です。継続的な監視と定期的な再評価を推奨します。\n\n"
            )
        else:
            report += "以下の優先順位で改善を実施することを推奨します：\n\n"
            for i, priority in enumerate(priorities, 1):
                report += f"{i}. **{priority}**\n"
            report += "\n"

        # 具体的な改善アクション
        report += """### 具体的なアクション項目

#### 短期施策（1-2ヶ月）
- [ ] 低品質モデル（Basic/Rejected）の特徴量見直し
- [ ] データ収集期間の延長検討
- [ ] 外れ値処理手法の最適化

#### 中期施策（3-6ヶ月）
- [ ] 新しい特徴量の探索と追加
- [ ] アンサンブル手法の導入検討
- [ ] リアルタイムデータ取得基盤の構築

#### 長期施策（6ヶ月以上）
- [ ] 深層学習モデルの検証
- [ ] 多店舗展開時のモデル汎化性能向上
- [ ] 自動再学習システムの構築

---

"""
        return report

    def _generate_implementation_plan(self, analysis_results: List[Dict[str, Any]]) -> str:
        """実装計画を生成"""
        # 品質レベル別の分類
        premium_products = []
        standard_products = []
        basic_products = []

        for result in analysis_results:
            quality_level = result.get("quality_level", "Rejected")
            product_name = result.get("product_name", "Unknown")

            if quality_level == "Premium":
                premium_products.append(product_name)
            elif quality_level == "Standard":
                standard_products.append(product_name)
            elif quality_level == "Basic":
                basic_products.append(product_name)

        return f"""## 段階的実装計画

### フェーズ1: 即座実行（Premium品質モデル）
**対象商品数**: {len(premium_products)}商品
**実装期間**: 即座〜2週間

**対象商品**:
{chr(10).join([f"- {product}" for product in premium_products[:10]])}
{f"- その他{len(premium_products)-10}商品" if len(premium_products) > 10 else ""}

**期待効果**:
- 即座に価格最適化による収益改善
- 在庫回転率の向上
- 顧客満足度の向上

### フェーズ2: 慎重実行（Standard品質モデル）
**対象商品数**: {len(standard_products)}商品
**実装期間**: 2週間〜1ヶ月

**実装条件**:
- A/Bテストによる効果検証
- 週次モニタリングの実施
- 手動調整機能の準備

### フェーズ3: 改善後実行（Basic品質モデル）
**対象商品数**: {len(basic_products)}商品
**実装期間**: 1〜3ヶ月後

**前提条件**:
- モデル品質の改善完了
- 十分な検証期間の確保

---

"""

    def _generate_appendix(self, analysis_results: List[Dict[str, Any]]) -> str:
        """付録を生成"""
        return """## 付録

### A. 技術仕様
- **機械学習アルゴリズム**: RandomForestRegressor
- **交差検証**: 5-fold Cross Validation
- **特徴量エンジニアリング**: ベースライン + 時間 + 気象特徴量
- **品質評価基準**: R²スコアベースの4段階評価

### B. 品質レベル定義
- **Premium** (R² ≥ 0.7): 即座実行可能な高品質モデル
- **Standard** (0.5 ≤ R² < 0.7): 慎重実行推奨モデル
- **Basic** (0.3 ≤ R² < 0.5): 改善検討が必要なモデル
- **Rejected** (R² < 0.3): 大幅な改善が必要なモデル

### C. 免責事項
- 本分析結果は過去データに基づく予測であり、将来の結果を保証するものではありません
- 実装時は適切な監視とフィードバック機構を設置してください
- 市場環境の変化に応じて定期的なモデル再評価を実施してください

### D. 連絡先
技術的な質問や追加分析のご要望は、データサイエンスチームまでお問い合わせください。

---

*本レポートは生鮮食品需要予測・分析システムにより自動生成されました。*
"""

    def save_markdown_report(self, report: str, filepath: str = None) -> str:
        """
        Markdownレポートを保存

        Args:
            report: レポート文字列
            filepath: 保存先パス

        Returns:
            保存先ファイルパス
        """
        try:
            if filepath is None:
                reports_dir = Path("reports")
                reports_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = reports_dir / f"demand_forecasting_report_{timestamp}.md"

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)

            self.logger.info(f"Markdownレポート保存: {filepath}")
            return str(filepath)

        except Exception as e:
            raise ReportGenerationError(f"Markdownレポート保存エラー: {e}")

    def analyze_feature_improvement_effects(
        self, analysis_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        特徴量追加による性能改善の定量的分析

        Args:
            analysis_results: 分析結果リスト

        Returns:
            改善効果分析結果
        """
        try:
            self.logger.info("特徴量改善効果分析を開始")

            improvement_analysis = {
                "baseline_vs_time": [],
                "time_vs_weather": [],
                "overall_improvement": {},
                "feature_contribution": {},
                "success_rate_by_stage": {},
            }

            # 各商品の段階別性能を分析
            for result in analysis_results:
                product_name = result.get("product_name", "Unknown")

                # 段階別性能データを取得（存在する場合）
                baseline_score = result.get("baseline_metrics", {}).get("r2_score", 0.0)
                time_score = result.get("time_enhanced_metrics", {}).get("r2_score", 0.0)
                final_score = result.get("test_metrics", {}).get("r2_score", 0.0)

                # ベースライン vs 時間特徴量追加の改善効果
                if baseline_score > 0 and time_score > 0:
                    baseline_time_improvement = time_score - baseline_score
                    improvement_analysis["baseline_vs_time"].append(
                        {
                            "product_name": product_name,
                            "baseline_r2": baseline_score,
                            "time_enhanced_r2": time_score,
                            "improvement": baseline_time_improvement,
                            "improvement_rate": (
                                (baseline_time_improvement / baseline_score * 100)
                                if baseline_score > 0
                                else 0
                            ),
                        }
                    )

                # 時間特徴量 vs 気象特徴量追加の改善効果
                if time_score > 0 and final_score > 0:
                    time_weather_improvement = final_score - time_score
                    improvement_analysis["time_vs_weather"].append(
                        {
                            "product_name": product_name,
                            "time_enhanced_r2": time_score,
                            "final_r2": final_score,
                            "improvement": time_weather_improvement,
                            "improvement_rate": (
                                (time_weather_improvement / time_score * 100)
                                if time_score > 0
                                else 0
                            ),
                        }
                    )

            # 全体的な改善効果統計
            if improvement_analysis["baseline_vs_time"]:
                baseline_improvements = [
                    item["improvement"] for item in improvement_analysis["baseline_vs_time"]
                ]
                improvement_analysis["overall_improvement"]["baseline_to_time"] = {
                    "mean_improvement": sum(baseline_improvements) / len(baseline_improvements),
                    "positive_improvements": len([x for x in baseline_improvements if x > 0]),
                    "total_products": len(baseline_improvements),
                    "success_rate": len([x for x in baseline_improvements if x > 0])
                    / len(baseline_improvements),
                }

            if improvement_analysis["time_vs_weather"]:
                weather_improvements = [
                    item["improvement"] for item in improvement_analysis["time_vs_weather"]
                ]
                improvement_analysis["overall_improvement"]["time_to_weather"] = {
                    "mean_improvement": sum(weather_improvements) / len(weather_improvements),
                    "positive_improvements": len([x for x in weather_improvements if x > 0]),
                    "total_products": len(weather_improvements),
                    "success_rate": len([x for x in weather_improvements if x > 0])
                    / len(weather_improvements),
                }

            # 特徴量カテゴリ別貢献度分析
            feature_contributions = {}
            for result in analysis_results:
                feature_importance = result.get("feature_importance", {})
                for feature, importance in feature_importance.items():
                    # 特徴量をカテゴリに分類
                    category = self._categorize_feature(feature)
                    if category not in feature_contributions:
                        feature_contributions[category] = []
                    feature_contributions[category].append(importance)

            # カテゴリ別平均重要度を計算
            for category, importances in feature_contributions.items():
                improvement_analysis["feature_contribution"][category] = {
                    "mean_importance": sum(importances) / len(importances),
                    "max_importance": max(importances),
                    "min_importance": min(importances),
                    "feature_count": len(importances),
                }

            # 段階別成功率分析
            quality_levels = ["Premium", "Standard", "Basic", "Rejected"]
            for stage in ["baseline", "time_enhanced", "final"]:
                stage_success_rates = {}
                for level in quality_levels:
                    stage_success_rates[level] = 0

                # 各段階での品質レベル分布を計算（簡略化）
                for result in analysis_results:
                    quality_level = result.get("quality_level", "Rejected")
                    if quality_level in stage_success_rates:
                        stage_success_rates[quality_level] += 1

                total = sum(stage_success_rates.values())
                if total > 0:
                    for level in quality_levels:
                        stage_success_rates[level] = stage_success_rates[level] / total

                improvement_analysis["success_rate_by_stage"][stage] = stage_success_rates

            self.logger.info("特徴量改善効果分析完了")
            return improvement_analysis

        except Exception as e:
            raise ReportGenerationError(f"特徴量改善効果分析エラー: {e}")

    def _categorize_feature(self, feature_name: str) -> str:
        """
        特徴量をカテゴリに分類

        Args:
            feature_name: 特徴量名

        Returns:
            特徴量カテゴリ
        """
        feature_name_lower = feature_name.lower()

        # 基本特徴量
        if any(keyword in feature_name_lower for keyword in ["価格", "単価", "売上", "数量"]):
            return "基本特徴量"

        # 時間特徴量
        elif any(
            keyword in feature_name_lower for keyword in ["時", "曜日", "月", "週末", "混雑"]
        ):
            return "時間特徴量"

        # 気象特徴量
        elif any(keyword in feature_name_lower for keyword in ["気温", "天気", "湿度", "降水"]):
            return "気象特徴量"

        # その他
        else:
            return "その他特徴量"

    def generate_improvement_analysis_report(self, improvement_analysis: Dict[str, Any]) -> str:
        """
        特徴量改善効果のレポートセクションを生成

        Args:
            improvement_analysis: 改善効果分析結果

        Returns:
            改善効果レポートセクション
        """
        report = "## 特徴量追加による性能改善分析\n\n"

        # 全体的な改善効果
        overall = improvement_analysis.get("overall_improvement", {})

        if "baseline_to_time" in overall:
            baseline_to_time = overall["baseline_to_time"]
            report += f"""### ベースライン → 時間特徴量追加の効果

- **平均改善度**: {baseline_to_time['mean_improvement']:.4f} (R²スコア)
- **改善商品数**: {baseline_to_time['positive_improvements']}/{baseline_to_time['total_products']}商品
- **成功率**: {baseline_to_time['success_rate']*100:.1f}%

"""

        if "time_to_weather" in overall:
            time_to_weather = overall["time_to_weather"]
            report += f"""### 時間特徴量 → 気象特徴量追加の効果

- **平均改善度**: {time_to_weather['mean_improvement']:.4f} (R²スコア)
- **改善商品数**: {time_to_weather['positive_improvements']}/{time_to_weather['total_products']}商品
- **成功率**: {time_to_weather['success_rate']*100:.1f}%

"""

        # 特徴量カテゴリ別貢献度
        feature_contrib = improvement_analysis.get("feature_contribution", {})
        if feature_contrib:
            report += "### 特徴量カテゴリ別重要度\n\n"
            report += "| カテゴリ | 平均重要度 | 最大重要度 | 特徴量数 |\n"
            report += "|---|---|---|---|\n"

            # 重要度順にソート
            sorted_categories = sorted(
                feature_contrib.items(), key=lambda x: x[1]["mean_importance"], reverse=True
            )

            for category, stats in sorted_categories:
                report += f"| {category} | {stats['mean_importance']:.3f} | {stats['max_importance']:.3f} | {stats['feature_count']} |\n"

            report += "\n"

        # 改善効果の詳細分析
        baseline_vs_time = improvement_analysis.get("baseline_vs_time", [])
        if baseline_vs_time:
            report += "### 時間特徴量追加による個別改善効果（上位10商品）\n\n"
            report += "| 商品名 | ベースライン | 時間特徴量追加 | 改善度 | 改善率 |\n"
            report += "|---|---|---|---|---|\n"

            # 改善度順にソート
            sorted_improvements = sorted(
                baseline_vs_time, key=lambda x: x["improvement"], reverse=True
            )

            for item in sorted_improvements[:10]:
                report += f"| {item['product_name']} | {item['baseline_r2']:.3f} | {item['time_enhanced_r2']:.3f} | {item['improvement']:+.3f} | {item['improvement_rate']:+.1f}% |\n"

            report += "\n"

        time_vs_weather = improvement_analysis.get("time_vs_weather", [])
        if time_vs_weather:
            report += "### 気象特徴量追加による個別改善効果（上位10商品）\n\n"
            report += "| 商品名 | 時間特徴量 | 最終モデル | 改善度 | 改善率 |\n"
            report += "|---|---|---|---|---|\n"

            # 改善度順にソート
            sorted_weather_improvements = sorted(
                time_vs_weather, key=lambda x: x["improvement"], reverse=True
            )

            for item in sorted_weather_improvements[:10]:
                report += f"| {item['product_name']} | {item['time_enhanced_r2']:.3f} | {item['final_r2']:.3f} | {item['improvement']:+.3f} | {item['improvement_rate']:+.1f}% |\n"

            report += "\n"

        # 改善効果のまとめ
        report += """### 改善効果のまとめ

#### 主要な発見
1. **時間特徴量の効果**: 営業時間、曜日、混雑度などの時間関連特徴量は多くの商品で予測精度を向上させる
2. **気象特徴量の効果**: 気温データは季節性のある商品で特に効果的
3. **商品特性による差**: 商品カテゴリによって効果的な特徴量が異なる

#### 今後の特徴量開発指針
- 高い改善効果を示した特徴量カテゴリの拡張
- 商品特性に応じた特徴量の個別最適化
- 外部データソースの追加検討

---

"""
        return report

    def generate_csv_reports(
        self, analysis_results: List[Dict[str, Any]], save_dir: str = None
    ) -> List[str]:
        """
        CSV形式のレポートを生成

        Args:
            analysis_results: 分析結果リスト
            save_dir: 保存先ディレクトリ

        Returns:
            保存されたCSVファイルパスのリスト
        """
        try:
            if save_dir is None:
                save_dir = Path("reports")
            else:
                save_dir = Path(save_dir)

            save_dir.mkdir(parents=True, exist_ok=True)
            saved_files = []

            # モデル性能レポート
            performance_data = []
            for result in analysis_results:
                product_name = result.get("product_name", "Unknown")
                test_metrics = result.get("test_metrics", {})
                cv_scores = result.get("cv_scores", {})

                performance_data.append(
                    {
                        "商品名": product_name,
                        "品質レベル": result.get("quality_level", "Unknown"),
                        "R²スコア": test_metrics.get("r2_score", 0.0),
                        "RMSE": test_metrics.get("rmse", 0.0),
                        "MAE": test_metrics.get("mae", 0.0),
                        "CV平均R²": cv_scores.get("mean_score", 0.0),
                        "CV標準偏差": cv_scores.get("std_score", 0.0),
                        "過学習スコア": result.get("overfitting_score", 1.0),
                    }
                )

            performance_df = pd.DataFrame(performance_data)
            performance_path = save_dir / "model_performance.csv"
            performance_df.to_csv(performance_path, index=False, encoding="utf-8-sig")
            saved_files.append(str(performance_path))

            # 特徴量重要度レポート
            importance_data = []
            for result in analysis_results:
                product_name = result.get("product_name", "Unknown")
                feature_importance = result.get("feature_importance", {})

                for feature, importance in feature_importance.items():
                    importance_data.append(
                        {"商品名": product_name, "特徴量": feature, "重要度": importance}
                    )

            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                importance_path = save_dir / "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
                saved_files.append(str(importance_path))

            # 需要曲線分析レポート
            demand_data = []
            for result in analysis_results:
                product_name = result.get("product_name", "Unknown")
                demand_results = result.get("demand_results", {})

                if demand_results:
                    current_price = demand_results.get("current_price", 0)
                    optimal_price = demand_results.get("optimal_price", 0)
                    price_change = (
                        ((optimal_price - current_price) / current_price * 100)
                        if current_price > 0
                        else 0
                    )

                    demand_data.append(
                        {
                            "商品名": product_name,
                            "現在価格": current_price,
                            "最適価格": optimal_price,
                            "価格変更率(%)": price_change,
                            "価格弾力性": demand_results.get("price_elasticity", 0),
                            "モデルR²": demand_results.get("r2_score", 0.0),
                            "データ点数": demand_results.get("data_points", 0),
                        }
                    )

            if demand_data:
                demand_df = pd.DataFrame(demand_data)
                demand_path = save_dir / "demand_analysis.csv"
                demand_df.to_csv(demand_path, index=False, encoding="utf-8-sig")
                saved_files.append(str(demand_path))

            # 特徴量改善効果分析レポート
            improvement_analysis = self.analyze_feature_improvement_effects(analysis_results)

            # ベースライン vs 時間特徴量の改善効果
            baseline_time_data = improvement_analysis.get("baseline_vs_time", [])
            if baseline_time_data:
                baseline_time_df = pd.DataFrame(baseline_time_data)
                baseline_time_path = save_dir / "feature_improvement_baseline_to_time.csv"
                baseline_time_df.to_csv(baseline_time_path, index=False, encoding="utf-8-sig")
                saved_files.append(str(baseline_time_path))

            # 時間 vs 気象特徴量の改善効果
            time_weather_data = improvement_analysis.get("time_vs_weather", [])
            if time_weather_data:
                time_weather_df = pd.DataFrame(time_weather_data)
                time_weather_path = save_dir / "feature_improvement_time_to_weather.csv"
                time_weather_df.to_csv(time_weather_path, index=False, encoding="utf-8-sig")
                saved_files.append(str(time_weather_path))

            # 特徴量カテゴリ別貢献度
            feature_contrib = improvement_analysis.get("feature_contribution", {})
            if feature_contrib:
                contrib_data = []
                for category, stats in feature_contrib.items():
                    contrib_data.append(
                        {
                            "特徴量カテゴリ": category,
                            "平均重要度": stats["mean_importance"],
                            "最大重要度": stats["max_importance"],
                            "最小重要度": stats["min_importance"],
                            "特徴量数": stats["feature_count"],
                        }
                    )

                contrib_df = pd.DataFrame(contrib_data)
                contrib_path = save_dir / "feature_category_contribution.csv"
                contrib_df.to_csv(contrib_path, index=False, encoding="utf-8-sig")
                saved_files.append(str(contrib_path))

            # 全体改善効果サマリー
            overall_improvement = improvement_analysis.get("overall_improvement", {})
            if overall_improvement:
                summary_data = []
                for stage, stats in overall_improvement.items():
                    summary_data.append(
                        {
                            "改善段階": stage,
                            "平均改善度": stats["mean_improvement"],
                            "改善商品数": stats["positive_improvements"],
                            "総商品数": stats["total_products"],
                            "成功率": stats["success_rate"],
                        }
                    )

                summary_df = pd.DataFrame(summary_data)
                summary_path = save_dir / "improvement_summary.csv"
                summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
                saved_files.append(str(summary_path))

            self.logger.info(f"CSVレポート生成完了: {len(saved_files)}ファイル")
            return saved_files

        except Exception as e:
            raise ReportGenerationError(f"CSVレポート生成エラー: {e}")

    def generate_comprehensive_report(
        self,
        analysis_results: List[Dict[str, Any]],
        quality_report: Dict[str, Any],
        output_dir: str = None,
    ) -> Dict[str, str]:
        """
        包括的なレポート生成（Markdown + CSV）

        Args:
            analysis_results: 分析結果リスト
            quality_report: 品質レポート
            output_dir: 出力ディレクトリ

        Returns:
            生成されたファイルパスの辞書
        """
        try:
            if output_dir is None:
                output_dir = "reports"

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            generated_files = {}

            # Markdownレポート生成
            markdown_report = self.generate_markdown_report(analysis_results, quality_report)
            markdown_path = self.save_markdown_report(
                markdown_report, output_path / "comprehensive_analysis_report.md"
            )
            generated_files["markdown_report"] = markdown_path

            # CSVレポート生成
            csv_files = self.generate_csv_reports(analysis_results, output_path)
            generated_files["csv_reports"] = csv_files

            # 特徴量改善効果の詳細分析レポート
            improvement_analysis = self.analyze_feature_improvement_effects(analysis_results)
            improvement_report = self.generate_improvement_analysis_report(improvement_analysis)

            improvement_path = output_path / "feature_improvement_analysis.md"
            with open(improvement_path, "w", encoding="utf-8") as f:
                f.write("# 特徴量改善効果分析レポート\n\n")
                f.write(f"**作成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
                f.write(improvement_report)

            generated_files["improvement_analysis"] = str(improvement_path)

            # 段階的導入計画の詳細レポート
            implementation_plan = self._generate_detailed_implementation_plan(analysis_results)
            plan_path = output_path / "implementation_plan.md"
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write("# 段階的導入計画詳細\n\n")
                f.write(f"**作成日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
                f.write(implementation_plan)

            generated_files["implementation_plan"] = str(plan_path)

            self.logger.info(f"包括的レポート生成完了: {len(generated_files)}種類のレポート")
            return generated_files

        except Exception as e:
            raise ReportGenerationError(f"包括的レポート生成エラー: {e}")

    def _generate_detailed_implementation_plan(
        self, analysis_results: List[Dict[str, Any]]
    ) -> str:
        """
        詳細な段階的導入計画を生成

        Args:
            analysis_results: 分析結果リスト

        Returns:
            詳細導入計画レポート
        """
        # 品質レベル別の分類と詳細分析
        premium_products = []
        standard_products = []
        basic_products = []
        rejected_products = []

        for result in analysis_results:
            quality_level = result.get("quality_level", "Rejected")
            product_name = result.get("product_name", "Unknown")
            r2_score = result.get("test_metrics", {}).get("r2_score", 0.0)

            demand_results = result.get("demand_results", {})
            optimal_price = demand_results.get("optimal_price", 0)
            current_price = demand_results.get("current_price", 0)
            price_change = (
                ((optimal_price - current_price) / current_price * 100) if current_price > 0 else 0
            )

            product_info = {
                "name": product_name,
                "r2_score": r2_score,
                "price_change": price_change,
                "optimal_price": optimal_price,
                "current_price": current_price,
            }

            if quality_level == "Premium":
                premium_products.append(product_info)
            elif quality_level == "Standard":
                standard_products.append(product_info)
            elif quality_level == "Basic":
                basic_products.append(product_info)
            else:
                rejected_products.append(product_info)

        # 各フェーズの期待効果を計算
        premium_revenue_impact = (
            sum([abs(p["price_change"]) for p in premium_products]) / len(premium_products)
            if premium_products
            else 0
        )
        standard_revenue_impact = (
            sum([abs(p["price_change"]) for p in standard_products]) / len(standard_products)
            if standard_products
            else 0
        )

        premium_avg_r2 = (
            sum([p["r2_score"] for p in premium_products]) / len(premium_products)
            if premium_products
            else 0.0
        )
        standard_avg_r2 = (
            sum([p["r2_score"] for p in standard_products]) / len(standard_products)
            if standard_products
            else 0.0
        )
        basic_avg_r2 = (
            sum([p["r2_score"] for p in basic_products]) / len(basic_products)
            if basic_products
            else 0.0
        )
        rejected_avg_r2 = (
            sum([p["r2_score"] for p in rejected_products]) / len(rejected_products)
            if rejected_products
            else 0.0
        )

        report = f"""## 段階的導入計画詳細

### フェーズ1: 即座実行（Premium品質モデル）

#### 対象商品詳細
**商品数**: {len(premium_products)}商品
**平均R²スコア**: {premium_avg_r2:.3f}
**平均価格変更率**: {premium_revenue_impact:.1f}%

#### 実装スケジュール
- **Week 1**: システム統合とテスト環境構築
- **Week 2**: 本番環境デプロイと監視システム設定
- **Week 3-4**: 段階的ロールアウトと効果測定

#### 期待効果
- **収益改善**: 価格最適化により{premium_revenue_impact:.1f}%の収益向上
- **在庫効率**: 需要予測精度向上により在庫回転率20%改善
- **顧客満足**: 適正価格設定により顧客満足度向上

#### リスク管理
- 日次監視による異常検知
- 手動オーバーライド機能の準備
- A/Bテストによる効果検証

### フェーズ2: 慎重実行（Standard品質モデル）

#### 対象商品詳細
**商品数**: {len(standard_products)}商品
**平均R²スコア**: {standard_avg_r2:.3f}
**平均価格変更率**: {standard_revenue_impact:.1f}%

#### 実装条件
- フェーズ1の成功確認後
- 追加の検証期間（4週間）
- 段階的ロールアウト（週10商品ずつ）

#### 監視項目
- 売上変動の週次監視
- 顧客反応の定性分析
- 競合価格との比較分析

### フェーズ3: 改善後実行（Basic品質モデル）

#### 対象商品詳細
**商品数**: {len(basic_products)}商品
**現在の平均R²スコア**: {basic_avg_r2:.3f}

#### 改善計画
1. **データ品質向上**
   - 追加データ収集期間の延長
   - 外れ値処理手法の見直し
   - 特徴量エンジニアリングの強化

2. **モデル改善**
   - アンサンブル手法の導入
   - ハイパーパラメータの最適化
   - 商品特性に応じたモデル個別化

3. **目標品質レベル**
   - R²スコア 0.5以上（Standard品質）への改善
   - 改善期間: 2-3ヶ月

### 改善対象（Rejected品質モデル）

#### 対象商品詳細
**商品数**: {len(rejected_products)}商品
**現在の平均R²スコア**: {rejected_avg_r2:.3f}

#### 根本的改善アプローチ
1. **データ収集戦略の見直し**
   - より長期間のデータ収集
   - 外部データソースの追加
   - データ品質管理の強化

2. **モデリング手法の変更**
   - 異なるアルゴリズムの検証
   - 深層学習手法の検討
   - 商品特性に特化したモデル開発

3. **ビジネス要件の再検討**
   - 予測対象の見直し
   - 成功指標の再定義
   - 代替アプローチの検討

### 全体的な成功指標

#### 短期指標（1-3ヶ月）
- Premium商品の収益改善率: {premium_revenue_impact:.1f}%以上
- 在庫回転率改善: 15%以上
- システム稼働率: 99.5%以上

#### 中期指標（3-6ヶ月）
- Standard商品の段階的導入完了
- Basic商品の品質改善率: 50%以上
- 全体的な予測精度向上: 20%以上

#### 長期指標（6ヶ月以上）
- 全商品カテゴリでの導入完了
- 継続的改善システムの確立
- ROI: 300%以上の達成

### 継続的改善計画

#### 月次レビュー
- モデル性能の定期評価
- 市場環境変化への対応
- 新商品への適用拡大

#### 四半期改善
- 特徴量の追加・見直し
- アルゴリズムの更新検討
- システム機能の拡張

#### 年次戦略見直し
- 全体戦略の再評価
- 技術トレンドの取り込み
- 組織能力の強化

---

*本計画は現在の分析結果に基づいており、実装時の状況に応じて柔軟に調整することを推奨します。*
"""
        return report
