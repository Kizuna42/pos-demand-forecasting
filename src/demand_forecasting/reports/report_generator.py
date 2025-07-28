from typing import Dict, Any, List
import pandas as pd
from pathlib import Path
from datetime import datetime

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.exceptions import ReportGenerationError


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

            # 個別商品分析結果
            report += self._generate_product_analysis(analysis_results)

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

            self.logger.info(f"CSVレポート生成完了: {len(saved_files)}ファイル")
            return saved_files

        except Exception as e:
            raise ReportGenerationError(f"CSVレポート生成エラー: {e}")
