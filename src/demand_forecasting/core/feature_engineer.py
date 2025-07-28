from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.exceptions import FeatureEngineeringError, ExternalAPIError


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""

    def __init__(self, config: Config):
        """
        初期化

        Args:
            config: 設定管理オブジェクト
        """
        self.config = config
        self.logger = Logger(config.get_logging_config()).get_logger("feature_engineer")
        self.feature_config = config.get_feature_config()

    def create_baseline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ベースライン特徴量を作成

        Args:
            df: 入力DataFrame

        Returns:
            ベースライン特徴量追加後のDataFrame
        """
        self.logger.info("ベースライン特徴量作成を開始")

        df_features = df.copy()

        try:
            # 価格関連特徴量
            if "平均価格" in df_features.columns:
                # 価格帯分類
                df_features["価格帯"] = pd.cut(
                    df_features["平均価格"],
                    bins=[0, 100, 300, 500, float("inf")],
                    labels=["低価格", "中価格", "高価格", "超高価格"],
                )

                # 価格の対数変換（価格の分布を正規化）
                df_features["価格_log"] = np.log1p(df_features["平均価格"])

                self.logger.info("価格関連特徴量を生成: 価格帯, 価格_log")

            # 売上関連特徴量
            if "金額" in df_features.columns and "数量" in df_features.columns:
                # 売上ボリューム分類
                df_features["売上ボリューム"] = pd.cut(
                    df_features["金額"],
                    bins=[0, 500, 1000, 2000, float("inf")],
                    labels=["小", "中", "大", "特大"],
                )

                # 数量の対数変換
                df_features["数量_log"] = np.log1p(df_features["数量"])

                self.logger.info("売上関連特徴量を生成: 売上ボリューム, 数量_log")

            # 商品カテゴリのダミー変数化
            if "商品カテゴリ" in df_features.columns:
                category_dummies = pd.get_dummies(df_features["商品カテゴリ"], prefix="カテゴリ")
                df_features = pd.concat([df_features, category_dummies], axis=1)
                self.logger.info(f"商品カテゴリダミー変数を生成: {list(category_dummies.columns)}")

            self.logger.info("ベースライン特徴量作成完了")

        except Exception as e:
            raise FeatureEngineeringError(f"ベースライン特徴量作成エラー: {e}")

        return df_features

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時間特徴量を追加

        Args:
            df: 入力DataFrame

        Returns:
            時間特徴量追加後のDataFrame
        """
        self.logger.info("時間特徴量追加を開始")

        df_features = df.copy()

        try:
            if "年月日" not in df_features.columns:
                raise FeatureEngineeringError("年月日列が見つかりません")

            # 日付列がdatetime型でない場合は変換
            if not pd.api.types.is_datetime64_any_dtype(df_features["年月日"]):
                df_features["年月日"] = pd.to_datetime(df_features["年月日"])

            # 基本的な時間特徴量
            df_features["年"] = df_features["年月日"].dt.year
            df_features["四半期"] = df_features["年月日"].dt.quarter
            df_features["月初フラグ"] = (df_features["年月日"].dt.day <= 5).astype(int)
            df_features["月末フラグ"] = (df_features["年月日"].dt.day >= 25).astype(int)

            # 季節性特徴量
            df_features["季節"] = df_features["月"].map(
                {
                    12: "冬",
                    1: "冬",
                    2: "冬",
                    3: "春",
                    4: "春",
                    5: "春",
                    6: "夏",
                    7: "夏",
                    8: "夏",
                    9: "秋",
                    10: "秋",
                    11: "秋",
                }
            )

            # 祝日・イベント日フラグ
            df_features = self._add_holiday_features(df_features)

            # 時刻と混雑度の設定（仮想的な時刻割り当て）
            if self.feature_config.get("time_features", {}).get("enable_time_assignment", False):
                df_features = self._assign_virtual_time(df_features)

            # 時間帯ダミー変数
            if "時刻" in df_features.columns:
                time_dummies = pd.get_dummies(df_features["時間帯"], prefix="時間帯")
                df_features = pd.concat([df_features, time_dummies], axis=1)

            self.logger.info("時間特徴量追加完了")

        except Exception as e:
            raise FeatureEngineeringError(f"時間特徴量追加エラー: {e}")

        return df_features

    def _add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """祝日・イベント日フラグを追加"""
        df_holidays = df.copy()

        # 主要な祝日（簡易版）
        holidays_2024 = [
            "2024-01-01",  # 元日
            "2024-02-14",  # バレンタインデー
            "2024-03-20",  # 春分の日
            "2024-04-29",  # 昭和の日
            "2024-05-03",  # 憲法記念日
            "2024-05-04",  # みどりの日
            "2024-05-05",  # こどもの日
            "2024-07-15",  # 海の日
            "2024-08-11",  # 山の日
            "2024-09-16",  # 敬老の日
            "2024-09-23",  # 秋分の日
            "2024-10-14",  # スポーツの日
            "2024-11-03",  # 文化の日
            "2024-11-23",  # 勤労感謝の日
            "2024-12-25",  # クリスマス
            "2024-12-31",  # 大晦日
        ]

        holiday_dates = pd.to_datetime(holidays_2024)
        df_holidays["祝日フラグ"] = (
            df_holidays["年月日"].dt.date.isin(holiday_dates.date).astype(int)
        )

        # 月末・月初セール期間
        if "月初フラグ" in df_holidays.columns and "月末フラグ" in df_holidays.columns:
            df_holidays["セール期間フラグ"] = (
                (df_holidays["月初フラグ"] == 1) | (df_holidays["月末フラグ"] == 1)
            ).astype(int)
        else:
            # 月初・月末フラグがない場合は日付から計算
            df_holidays["セール期間フラグ"] = (
                (df_holidays["年月日"].dt.day <= 5) | (df_holidays["年月日"].dt.day >= 25)
            ).astype(int)

        return df_holidays

    def _assign_virtual_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """仮想的な時刻を割り当て"""
        df_time = df.copy()

        # 商品カテゴリに応じた購買時間帯の傾向を模擬
        category_time_mapping = {
            "野菜": {"morning": 0.3, "afternoon": 0.5, "evening": 0.2},
            "肉類": {"morning": 0.2, "afternoon": 0.3, "evening": 0.5},
            "魚類": {"morning": 0.4, "afternoon": 0.3, "evening": 0.3},
            "果物": {"morning": 0.2, "afternoon": 0.6, "evening": 0.2},
            "その他": {"morning": 0.33, "afternoon": 0.33, "evening": 0.33},
        }

        def assign_time_slot(category):
            if category in category_time_mapping:
                probs = list(category_time_mapping[category].values())
                # 確率を正規化して合計を1にする
                probs = np.array(probs)
                probs = probs / probs.sum()
                time_slots = ["morning", "afternoon", "evening"]
                return np.random.choice(time_slots, p=probs)
            return "afternoon"  # デフォルト

        # カテゴリがない場合はランダム
        if "商品カテゴリ" in df_time.columns:
            df_time["時間帯"] = df_time["商品カテゴリ"].apply(assign_time_slot)
        else:
            df_time["時間帯"] = np.random.choice(
                ["morning", "afternoon", "evening"], size=len(df_time)
            )

        # 時刻を数値に変換
        time_mapping = {"morning": 9, "afternoon": 14, "evening": 19}
        df_time["時刻"] = df_time["時間帯"].map(time_mapping)

        # 混雑度を計算（時間帯と曜日に基づく）
        df_time["混雑度"] = self._calculate_congestion(df_time)

        return df_time

    def _calculate_congestion(self, df: pd.DataFrame) -> pd.Series:
        """混雑度を計算"""
        congestion = pd.Series(index=df.index, dtype=float)

        for idx, row in df.iterrows():
            base_congestion = 0.5  # ベース混雑度

            # 時間帯による調整
            if row["時間帯"] == "morning":
                time_factor = 0.7
            elif row["時間帯"] == "afternoon":
                time_factor = 1.0
            else:  # evening
                time_factor = 1.2

            # 曜日による調整
            if row["週末フラグ"] == 1:
                day_factor = 1.3
            else:
                day_factor = 1.0

            # 祝日による調整
            holiday_factor = 1.2 if row.get("祝日フラグ", 0) == 1 else 1.0

            congestion[idx] = min(1.0, base_congestion * time_factor * day_factor * holiday_factor)

        return congestion

    def integrate_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        気象特徴量を統合

        Args:
            df: 入力DataFrame

        Returns:
            気象特徴量追加後のDataFrame
        """
        self.logger.info("気象特徴量統合を開始")

        df_weather = df.copy()

        try:
            if not self.feature_config.get("weather_features", {}).get(
                "enable_weather_api", False
            ):
                self.logger.info("気象API無効のため、気象特徴量をスキップ")
                return self._add_synthetic_weather_features(df_weather)

            # 実際の気象データ取得を試行
            try:
                df_weather = self._fetch_weather_data(df_weather)
            except ExternalAPIError as e:
                self.logger.warning(f"気象API取得失敗: {e}")
                self.logger.info("合成気象特徴量を使用")
                df_weather = self._add_synthetic_weather_features(df_weather)

            self.logger.info("気象特徴量統合完了")

        except Exception as e:
            raise FeatureEngineeringError(f"気象特徴量統合エラー: {e}")

        return df_weather

    def _fetch_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """外部APIから気象データを取得"""
        df_weather = df.copy()

        api_config = self.feature_config.get("weather_features", {})
        api_endpoint = api_config.get("api_endpoint")
        location = api_config.get("location", "Tokyo")

        if not api_endpoint:
            raise ExternalAPIError("気象API エンドポイントが設定されていません")

        # 日付範囲を取得
        min_date = df["年月日"].min().strftime("%Y-%m-%d")
        max_date = df["年月日"].max().strftime("%Y-%m-%d")

        # Open-Meteo API パラメータ
        params = {
            "latitude": 35.6762,  # 東京の緯度
            "longitude": 139.6503,  # 東京の経度
            "start_date": min_date,
            "end_date": max_date,
            "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max,precipitation_sum",
            "timezone": "Asia/Tokyo",
        }

        try:
            self.logger.info(f"気象データ取得中: {min_date} ~ {max_date}")
            response = requests.get(api_endpoint, params=params, timeout=30)
            response.raise_for_status()

            weather_data = response.json()

            # 気象データをDataFrameに変換
            weather_df = pd.DataFrame(
                {
                    "年月日": pd.to_datetime(weather_data["daily"]["time"]),
                    "平均気温": weather_data["daily"]["temperature_2m_mean"],
                    "最低気温": weather_data["daily"]["temperature_2m_min"],
                    "最高気温": weather_data["daily"]["temperature_2m_max"],
                    "降水量": weather_data["daily"]["precipitation_sum"],
                }
            )

            # 日付でマージ
            df_weather["年月日_date"] = df_weather["年月日"].dt.date
            weather_df["年月日_date"] = weather_df["年月日"].dt.date

            df_weather = df_weather.merge(
                weather_df[["年月日_date", "平均気温", "最低気温", "最高気温", "降水量"]],
                on="年月日_date",
                how="left",
            )

            # 一時列を削除
            df_weather.drop("年月日_date", axis=1, inplace=True)

            # 気象関連特徴量を追加
            df_weather = self._add_weather_features(df_weather)

            self.logger.info("気象データ取得・統合完了")

        except requests.RequestException as e:
            raise ExternalAPIError(f"気象API リクエストエラー: {e}")
        except KeyError as e:
            raise ExternalAPIError(f"気象API レスポンス形式エラー: {e}")

        return df_weather

    def _add_synthetic_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """合成気象特徴量を追加（API取得失敗時）"""
        df_synthetic = df.copy()

        # 季節に応じた合成気温データ
        def synthetic_temperature(month):
            temp_map = {
                1: 5,
                2: 7,
                3: 12,
                4: 18,
                5: 23,
                6: 26,
                7: 29,
                8: 31,
                9: 27,
                10: 21,
                11: 15,
                12: 8,
            }
            base_temp = temp_map.get(month, 20)
            # ランダムな変動を追加
            return base_temp + np.random.normal(0, 3)

        df_synthetic["平均気温"] = df_synthetic["月"].apply(synthetic_temperature)
        df_synthetic["最低気温"] = df_synthetic["平均気温"] - np.random.uniform(
            3, 8, len(df_synthetic)
        )
        df_synthetic["最高気温"] = df_synthetic["平均気温"] + np.random.uniform(
            3, 8, len(df_synthetic)
        )
        df_synthetic["降水量"] = np.random.exponential(2, len(df_synthetic))

        # 気象関連特徴量を追加
        df_synthetic = self._add_weather_features(df_synthetic)

        self.logger.info("合成気象特徴量を生成")

        return df_synthetic

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """気象関連特徴量を追加"""
        df_weather_features = df.copy()

        # 気温区分
        df_weather_features["気温区分"] = pd.cut(
            df_weather_features["平均気温"],
            bins=[-float("inf"), 5, 15, 25, float("inf")],
            labels=["寒い", "涼しい", "暖かい", "暑い"],
        )

        # 気温の移動平均（3日間）
        df_weather_features = df_weather_features.sort_values("年月日")
        df_weather_features["気温_移動平均"] = (
            df_weather_features["平均気温"].rolling(window=3, min_periods=1).mean()
        )

        # 雨の日フラグ
        df_weather_features["雨の日フラグ"] = (df_weather_features["降水量"] > 1.0).astype(int)

        # 気温変化フラグ
        df_weather_features["急激な気温変化フラグ"] = (
            abs(df_weather_features["最高気温"] - df_weather_features["最低気温"]) > 10
        ).astype(int)

        return df_weather_features

    def select_features(self, df: pd.DataFrame, target: str) -> List[str]:
        """
        特徴量選択を実行

        Args:
            df: 入力DataFrame
            target: ターゲット列名

        Returns:
            選択された特徴量リスト
        """
        self.logger.info("特徴量選択を開始")

        try:
            # 数値特徴量のみを対象
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

            # ターゲット列を除外
            if target in numeric_features:
                numeric_features.remove(target)

            # 相関分析による特徴量選択
            correlation_matrix = df[numeric_features + [target]].corr()
            target_correlation = abs(correlation_matrix[target]).sort_values(ascending=False)

            # 相関が閾値以上の特徴量を選択
            correlation_threshold = 0.1
            selected_features = target_correlation[
                target_correlation > correlation_threshold
            ].index.tolist()

            # ターゲットを除外
            if target in selected_features:
                selected_features.remove(target)

            # 高相関の特徴量ペアを除去
            selected_features = self._remove_highly_correlated_features(
                df, selected_features, threshold=0.8
            )

            self.logger.info(f"選択された特徴量数: {len(selected_features)}")
            self.logger.debug(f"選択された特徴量: {selected_features}")

            return selected_features

        except Exception as e:
            raise FeatureEngineeringError(f"特徴量選択エラー: {e}")

    def _remove_highly_correlated_features(
        self, df: pd.DataFrame, features: List[str], threshold: float = 0.8
    ) -> List[str]:
        """高相関の特徴量ペアを除去"""
        correlation_matrix = df[features].corr().abs()

        # 上三角行列を取得
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # 高相関の特徴量を特定
        to_drop = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > threshold):
                to_drop.append(column)

        # 高相関特徴量を除去
        selected_features = [f for f in features if f not in to_drop]

        if to_drop:
            self.logger.info(f"高相関により除去された特徴量: {to_drop}")

        return selected_features

    def get_feature_importance(
        self, df: pd.DataFrame, features: List[str], target: str
    ) -> Dict[str, float]:
        """
        特徴量重要度を計算

        Args:
            df: 入力DataFrame
            features: 特徴量リスト
            target: ターゲット列名

        Returns:
            特徴量重要度辞書
        """
        try:
            # 相関係数による重要度
            correlation_importance = {}
            for feature in features:
                if feature in df.columns:
                    corr = abs(df[feature].corr(df[target]))
                    correlation_importance[feature] = corr if not pd.isna(corr) else 0.0

            # 重要度でソート
            sorted_importance = dict(
                sorted(correlation_importance.items(), key=lambda x: x[1], reverse=True)
            )

            self.logger.info("特徴量重要度計算完了")

            return sorted_importance

        except Exception as e:
            raise FeatureEngineeringError(f"特徴量重要度計算エラー: {e}")
