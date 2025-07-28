from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.exceptions import VisualizationError


class WantPlotter:
    """want_style_plotter統合可視化クラス"""

    def __init__(self, config: Config):
        """
        初期化

        Args:
            config: 設定管理オブジェクト
        """
        self.config = config
        self.logger = Logger(config.get_logging_config()).get_logger("want_plotter")
        self.viz_config = config.get_visualization_config()
        self._setup_style()

    def _setup_style(self):
        """Wantスタイルの設定"""
        # Wantスタイルのカラーパレット
        self.want_colors = {
            "primary": "#FF6B35",  # オレンジ
            "secondary": "#004E89",  # ネイビー
            "accent": "#1A936F",  # グリーン
            "background": "#F7F9FA",  # ライトグレー
            "text": "#2C3E50",  # ダークグレー
            "success": "#27AE60",  # 成功色
            "warning": "#F39C12",  # 警告色
            "error": "#E74C3C",  # エラー色
        }

        # matplotlibスタイル設定
        plt.style.use("default")
        sns.set_palette(
            [
                self.want_colors["primary"],
                self.want_colors["secondary"],
                self.want_colors["accent"],
                self.want_colors["success"],
            ]
        )

        # フォント設定
        plt.rcParams["font.family"] = [
            "DejaVu Sans",
            "Hiragino Sans",
            "Yu Gothic",
            "Meiryo",
            "sans-serif",
        ]
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 16
        plt.rcParams["axes.labelsize"] = 14

    def create_demand_curve_plot(
        self, demand_results: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """
        需要曲線プロットを作成

        Args:
            demand_results: 需要曲線分析結果
            save_path: 保存先パス

        Returns:
            保存先ファイルパス
        """
        try:
            self.logger.info(f"需要曲線プロット作成開始: {demand_results['product_name']}")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # データ取得
            price_demand_data = demand_results["price_demand_data"]
            demand_func = demand_results["demand_curve_function"]
            optimal_price = demand_results["optimal_price"]
            current_price = demand_results["current_price"]

            # 左側: 需要曲線
            self._plot_demand_curve(
                ax1, price_demand_data, demand_func, optimal_price, current_price
            )
            ax1.set_title(f"{demand_results['product_name']} - 需要曲線", fontweight="bold")

            # 右側: 収益曲線
            self._plot_revenue_curve(
                ax2, demand_func, optimal_price, current_price, demand_results["price_range"]
            )
            ax2.set_title(f"{demand_results['product_name']} - 収益曲線", fontweight="bold")

            plt.tight_layout()

            # 保存
            if save_path is None:
                output_dir = Path(self.viz_config.get("output_dir", "output/visualizations"))
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"demand_curve_{demand_results['product_name']}.png"

            plt.savefig(
                save_path,
                dpi=self.viz_config.get("dpi", 300),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

            self.logger.info(f"需要曲線プロット保存: {save_path}")
            return str(save_path)

        except Exception as e:
            raise VisualizationError(f"需要曲線プロット作成エラー: {e}")

    def _plot_demand_curve(self, ax, price_demand_data, demand_func, optimal_price, current_price):
        """需要曲線をプロット"""
        # 実データ点をプロット
        ax.scatter(
            price_demand_data["price"],
            price_demand_data["quantity"],
            color=self.want_colors["secondary"],
            alpha=0.6,
            s=50,
            label="実データ",
        )

        # 需要曲線をプロット
        price_range = np.linspace(
            price_demand_data["price"].min(), price_demand_data["price"].max(), 100
        )
        demand_curve = [max(0, demand_func(p)) for p in price_range]

        ax.plot(
            price_range,
            demand_curve,
            color=self.want_colors["primary"],
            linewidth=3,
            label="需要曲線",
        )

        # 最適価格と現在価格を表示
        if optimal_price:
            optimal_quantity = max(0, demand_func(optimal_price))
            ax.axvline(
                optimal_price,
                color=self.want_colors["success"],
                linestyle="--",
                alpha=0.8,
                label=f"最適価格: ¥{optimal_price:.0f}",
            )
            ax.scatter(
                [optimal_price],
                [optimal_quantity],
                color=self.want_colors["success"],
                s=100,
                zorder=5,
            )

        if current_price:
            current_quantity = max(0, demand_func(current_price))
            ax.axvline(
                current_price,
                color=self.want_colors["warning"],
                linestyle="--",
                alpha=0.8,
                label=f"現在価格: ¥{current_price:.0f}",
            )
            ax.scatter(
                [current_price],
                [current_quantity],
                color=self.want_colors["warning"],
                s=100,
                zorder=5,
            )

        ax.set_xlabel("価格 (¥)", fontweight="bold")
        ax.set_ylabel("需要量", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_revenue_curve(self, ax, demand_func, optimal_price, current_price, price_range):
        """収益曲線をプロット"""
        prices = np.linspace(price_range[0], price_range[1], 100)
        revenues = []

        for price in prices:
            quantity = max(0, demand_func(price))
            revenue = price * quantity
            revenues.append(revenue)

        ax.plot(prices, revenues, color=self.want_colors["accent"], linewidth=3, label="収益曲線")

        # 最適価格での収益
        if optimal_price:
            optimal_revenue = optimal_price * max(0, demand_func(optimal_price))
            ax.axvline(
                optimal_price,
                color=self.want_colors["success"],
                linestyle="--",
                alpha=0.8,
                label=f"最適価格: ¥{optimal_price:.0f}",
            )
            ax.scatter(
                [optimal_price],
                [optimal_revenue],
                color=self.want_colors["success"],
                s=100,
                zorder=5,
            )

        # 現在価格での収益
        if current_price:
            current_revenue = current_price * max(0, demand_func(current_price))
            ax.axvline(
                current_price,
                color=self.want_colors["warning"],
                linestyle="--",
                alpha=0.8,
                label=f"現在価格: ¥{current_price:.0f}",
            )
            ax.scatter(
                [current_price],
                [current_revenue],
                color=self.want_colors["warning"],
                s=100,
                zorder=5,
            )

        ax.set_xlabel("価格 (¥)", fontweight="bold")
        ax.set_ylabel("収益 (¥)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_feature_importance_plot(
        self,
        feature_importance: Dict[str, float],
        product_name: str,
        save_path: Optional[str] = None,
    ) -> str:
        """
        特徴量重要度プロットを作成

        Args:
            feature_importance: 特徴量重要度辞書
            product_name: 商品名
            save_path: 保存先パス

        Returns:
            保存先ファイルパス
        """
        try:
            # 上位10特徴量を選択
            top_features = dict(list(feature_importance.items())[:10])

            fig, ax = plt.subplots(figsize=(12, 8))

            # 横棒グラフで表示
            features = list(top_features.keys())
            importances = list(top_features.values())

            bars = ax.barh(features, importances, color=self.want_colors["primary"])

            # 値をバーに表示
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                ax.text(
                    bar.get_width() + max(importances) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{imp:.3f}",
                    ha="left",
                    va="center",
                    fontweight="bold",
                )

            ax.set_xlabel("重要度", fontweight="bold")
            ax.set_title(f"{product_name} - 特徴量重要度", fontsize=16, fontweight="bold")
            ax.grid(True, axis="x", alpha=0.3)

            # 色のグラデーション
            for i, bar in enumerate(bars):
                bar.set_color(plt.cm.YlOrRd(0.3 + 0.7 * (len(bars) - i) / len(bars)))

            plt.tight_layout()

            # 保存
            if save_path is None:
                output_dir = Path(self.viz_config.get("output_dir", "output/visualizations"))
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / f"feature_importance_{product_name}.png"

            plt.savefig(
                save_path,
                dpi=self.viz_config.get("dpi", 300),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

            self.logger.info(f"特徴量重要度プロット保存: {save_path}")
            return str(save_path)

        except Exception as e:
            raise VisualizationError(f"特徴量重要度プロット作成エラー: {e}")

    def create_quality_dashboard(
        self, quality_data: Dict[str, Any], save_path: Optional[str] = None
    ) -> str:
        """
        品質ダッシュボードを作成

        Args:
            quality_data: 品質データ
            save_path: 保存先パス

        Returns:
            保存先ファイルパス
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # 品質レベル分布
            self._plot_quality_distribution(ax1, quality_data.get("quality_distribution", {}))

            # 実用化準備状況分布
            self._plot_implementation_distribution(
                ax2, quality_data.get("implementation_distribution", {})
            )

            # カテゴリ別成功率
            self._plot_category_success_rates(ax3, quality_data.get("category_success_rates", {}))

            # R²スコア分布
            self._plot_r2_distribution(ax4, quality_data)

            plt.suptitle("品質評価ダッシュボード", fontsize=20, fontweight="bold", y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # 保存
            if save_path is None:
                output_dir = Path(self.viz_config.get("output_dir", "output/visualizations"))
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / "quality_dashboard.png"

            plt.savefig(
                save_path,
                dpi=self.viz_config.get("dpi", 300),
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

            self.logger.info(f"品質ダッシュボード保存: {save_path}")
            return str(save_path)

        except Exception as e:
            raise VisualizationError(f"品質ダッシュボード作成エラー: {e}")

    def _plot_quality_distribution(self, ax, quality_dist):
        """品質レベル分布をプロット"""
        if not quality_dist:
            ax.text(0.5, 0.5, "データなし", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("品質レベル分布")
            return

        levels = list(quality_dist.keys())
        counts = list(quality_dist.values())

        colors = [
            self.want_colors["success"],
            self.want_colors["accent"],
            self.want_colors["warning"],
            self.want_colors["error"],
        ][: len(levels)]

        wedges, texts, autotexts = ax.pie(
            counts, labels=levels, colors=colors, autopct="%1.1f%%", startangle=90
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax.set_title("品質レベル分布", fontweight="bold")

    def _plot_implementation_distribution(self, ax, impl_dist):
        """実用化準備状況分布をプロット"""
        if not impl_dist:
            ax.text(0.5, 0.5, "データなし", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("実用化準備状況")
            return

        statuses = list(impl_dist.keys())
        counts = list(impl_dist.values())

        bars = ax.bar(statuses, counts, color=self.want_colors["secondary"])

        # 値をバーに表示
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("実用化準備状況", fontweight="bold")
        ax.set_ylabel("商品数", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def _plot_category_success_rates(self, ax, category_rates):
        """カテゴリ別成功率をプロット"""
        if not category_rates:
            ax.text(0.5, 0.5, "データなし", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("カテゴリ別成功率")
            return

        categories = list(category_rates.keys())
        rates = [rate * 100 for rate in category_rates.values()]  # パーセント表示

        bars = ax.bar(categories, rates, color=self.want_colors["accent"])

        # 値をバーに表示
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title("カテゴリ別成功率", fontweight="bold")
        ax.set_ylabel("成功率 (%)", fontweight="bold")
        ax.set_ylim(0, 105)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def _plot_r2_distribution(self, ax, quality_data):
        """R²スコア分布をプロット"""
        avg_r2 = quality_data.get("average_r2", 0)
        r2_std = quality_data.get("r2_std", 0)

        # 統計値を表示
        ax.text(
            0.5,
            0.7,
            f"平均R²スコア",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
        )
        ax.text(
            0.5,
            0.5,
            f"{avg_r2:.3f}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            color=self.want_colors["primary"],
        )
        ax.text(
            0.5,
            0.3,
            f"標準偏差: {r2_std:.3f}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("R²スコア統計", fontweight="bold")

    def create_comprehensive_report_plots(
        self, analysis_results: List[Dict[str, Any]], save_dir: Optional[str] = None
    ) -> List[str]:
        """
        包括的なレポート用プロットを作成

        Args:
            analysis_results: 分析結果リスト
            save_dir: 保存先ディレクトリ

        Returns:
            保存されたファイルパスのリスト
        """
        saved_files = []

        if save_dir is None:
            save_dir = Path(self.viz_config.get("output_dir", "output/visualizations"))
        else:
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 各商品の需要曲線プロット
            for result in analysis_results:
                if "demand_results" in result:
                    demand_plot_path = self.create_demand_curve_plot(
                        result["demand_results"], save_dir / f"demand_{result['product_name']}.png"
                    )
                    saved_files.append(demand_plot_path)

                # 特徴量重要度プロット
                if "feature_importance" in result:
                    importance_plot_path = self.create_feature_importance_plot(
                        result["feature_importance"],
                        result["product_name"],
                        save_dir / f"importance_{result['product_name']}.png",
                    )
                    saved_files.append(importance_plot_path)

            self.logger.info(f"包括的レポートプロット作成完了: {len(saved_files)}ファイル")

        except Exception as e:
            raise VisualizationError(f"包括的レポートプロット作成エラー: {e}")

        return saved_files
