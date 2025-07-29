from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import json
import sqlite3
import hashlib

import numpy as np
import pandas as pd
import requests

from ..utils.config import Config
from ..utils.exceptions import ExternalAPIError, FeatureEngineeringError
from ..utils.logger import Logger


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

        # 気象データキャッシュの初期化
        self._init_weather_cache()

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
        """
        確率モデルに基づく時刻付与機能

        要件A: トランザクションへの時刻付与機能
        - 平日・休日別の確率分布関数を実装
        - 確率モデルに基づくランダム時刻生成
        - トランザクションデータへの時刻フィールド追加
        """
        df_time = df.copy()

        self.logger.info("確率モデルに基づく時刻付与を開始")

        # 確率モデル設定を取得
        prob_config = self.feature_config.get("time_features", {}).get("probability_model", {})
        weekday_config = prob_config.get(
            "weekday_pattern", {"peak_time": 18, "peak_intensity": 0.8, "spread": 2.0}
        )
        holiday_config = prob_config.get(
            "holiday_pattern", {"peak_time": 12, "peak_intensity": 0.4, "spread": 4.0}
        )

        # 時刻と混雑度（確率値）を計算
        times = []
        congestion_values = []

        for idx, row in df_time.iterrows():
            is_weekend = row.get("週末フラグ", 0) == 1
            is_holiday = row.get("祝日フラグ", 0) == 1

            # 平日・休日パターンの選択
            if is_weekend or is_holiday:
                # 休日パターン: 早い時間帯に低く緩やかな丘
                config = holiday_config
                pattern_type = "holiday"
            else:
                # 平日パターン: 遅い時間帯に高く鋭い山
                config = weekday_config
                pattern_type = "weekday"

            # 確率分布関数による時刻生成
            assigned_time, probability = self._generate_time_from_probability_model(config)

            times.append(assigned_time)
            congestion_values.append(probability)

        # 結果をDataFrameに追加
        df_time["時刻"] = times
        df_time["混雑度"] = congestion_values

        # 時間帯分類を追加
        df_time["時間帯"] = df_time["時刻"].apply(self._classify_time_period)

        self.logger.info(
            f"時刻付与完了: 平均時刻={np.mean(times):.1f}, 平均混雑度={np.mean(congestion_values):.3f}"
        )

        return df_time

    def _generate_time_from_probability_model(self, config: Dict[str, float]) -> tuple:
        """
        確率モデルから時刻と確率値を生成

        Args:
            config: 確率モデル設定（peak_time, peak_intensity, spread）

        Returns:
            tuple: (時刻, 確率値)
        """
        peak_time = config["peak_time"]
        peak_intensity = config["peak_intensity"]
        spread = config["spread"]

        # 営業時間を9-21時と仮定
        time_range = np.arange(9, 22, 0.5)  # 30分刻み

        # ガウス分布ベースの確率密度関数
        probabilities = peak_intensity * np.exp(-((time_range - peak_time) ** 2) / (2 * spread**2))

        # 確率を正規化
        probabilities = probabilities / np.sum(probabilities)

        # 確率に基づいて時刻を選択
        selected_time = np.random.choice(time_range, p=probabilities)

        # 選択された時刻の確率値（混雑度として使用）
        time_idx = np.where(time_range == selected_time)[0][0]
        probability_value = probabilities[time_idx]

        return float(selected_time), float(probability_value)

    def _classify_time_period(self, time_hour: float) -> str:
        """時刻を時間帯に分類"""
        if time_hour < 11:
            return "morning"
        elif time_hour < 16:
            return "afternoon"
        else:
            return "evening"

    def _calculate_congestion(self, df: pd.DataFrame) -> pd.Series:
        """
        混雑度を計算（レガシー関数 - 新しい実装では_assign_virtual_time内で処理）

        要件B: 混雑度説明変数の実装
        - 確率モデルの縦軸値（確率値）を混雑度として活用
        - 各トランザクションへの混雑度フィールド追加
        """
        self.logger.warning(
            "レガシー混雑度計算関数が呼び出されました。新しい確率モデルベースの実装を使用してください。"
        )

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

    def _init_weather_cache(self):
        """気象データキャッシュの初期化"""
        try:
            from pathlib import Path

            # キャッシュディレクトリの作成
            cache_dir = Path("data/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            # SQLiteキャッシュデータベースの初期化
            self.cache_db_path = cache_dir / "weather_cache.db"

            with sqlite3.connect(str(self.cache_db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS weather_cache (
                        cache_key TEXT PRIMARY KEY,
                        date_range TEXT,
                        api_name TEXT,
                        data_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """
                )
                conn.commit()

            self.logger.info(f"気象データキャッシュ初期化完了: {self.cache_db_path}")

        except Exception as e:
            self.logger.warning(f"気象データキャッシュ初期化失敗: {e}")
            self.cache_db_path = None

    def _get_cache_key(self, api_name: str, params: dict) -> str:
        """キャッシュキーを生成"""
        # パラメータを正規化してハッシュ化
        param_str = json.dumps(sorted(params.items()), ensure_ascii=False)
        cache_str = f"{api_name}:{param_str}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cached_weather_data(self, cache_key: str) -> Optional[dict]:
        """キャッシュから気象データを取得"""
        if not self.cache_db_path:
            return None

        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                cursor = conn.execute(
                    """
                    SELECT data_json FROM weather_cache 
                    WHERE cache_key = ? AND expires_at > CURRENT_TIMESTAMP
                """,
                    (cache_key,),
                )

                result = cursor.fetchone()
                if result:
                    self.logger.info("キャッシュから気象データを取得")
                    return json.loads(result[0])

        except Exception as e:
            self.logger.warning(f"キャッシュ取得エラー: {e}")

        return None

    def _cache_weather_data(self, cache_key: str, api_name: str, date_range: str, data: dict):
        """気象データをキャッシュに保存"""
        if not self.cache_db_path:
            return

        try:
            # 1日間キャッシュ（履歴データなので長期間保持可能）
            expires_at = datetime.now() + timedelta(days=1)

            with sqlite3.connect(str(self.cache_db_path)) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO weather_cache 
                    (cache_key, date_range, api_name, data_json, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (cache_key, date_range, api_name, json.dumps(data), expires_at),
                )
                conn.commit()

            self.logger.info("気象データをキャッシュに保存")

        except Exception as e:
            self.logger.warning(f"キャッシュ保存エラー: {e}")

    def _cleanup_weather_cache(self):
        """期限切れキャッシュの削除"""
        if not self.cache_db_path:
            return

        try:
            with sqlite3.connect(str(self.cache_db_path)) as conn:
                cursor = conn.execute(
                    "DELETE FROM weather_cache WHERE expires_at <= CURRENT_TIMESTAMP"
                )
                deleted_count = cursor.rowcount
                conn.commit()

            if deleted_count > 0:
                self.logger.info(f"期限切れキャッシュを削除: {deleted_count}件")

        except Exception as e:
            self.logger.warning(f"キャッシュクリーンアップエラー: {e}")

    def _fetch_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """外部APIから気象データを取得（履歴データ対応版）"""
        df_weather = df.copy()

        api_config = self.feature_config.get("weather_features", {})

        # 日付範囲を取得
        min_date = df["年月日"].min().strftime("%Y-%m-%d")
        max_date = df["年月日"].max().strftime("%Y-%m-%d")

        # 複数APIのフォールバック設定
        weather_apis = [
            {
                "name": "open-meteo-historical",
                "endpoint": "https://archive-api.open-meteo.com/v1/archive",
                "params": {
                    "latitude": 35.6762,  # 東京の緯度
                    "longitude": 139.6503,  # 東京の経度
                    "start_date": min_date,
                    "end_date": max_date,
                    "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_mean",
                    "timezone": "Asia/Tokyo",
                },
            },
            {
                "name": "open-meteo-forecast",
                "endpoint": "https://api.open-meteo.com/v1/forecast",
                "params": {
                    "latitude": 35.6762,
                    "longitude": 139.6503,
                    "start_date": min_date,
                    "end_date": max_date,
                    "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max,precipitation_sum",
                    "timezone": "Asia/Tokyo",
                },
            },
        ]

        # 期限切れキャッシュの削除
        self._cleanup_weather_cache()

        for api_info in weather_apis:
            try:
                # キャッシュキーの生成
                cache_key = self._get_cache_key(api_info["name"], api_info["params"])

                # キャッシュから取得を試行
                cached_data = self._get_cached_weather_data(cache_key)
                if cached_data:
                    weather_data = cached_data
                else:
                    # APIから取得
                    self.logger.info(
                        f"気象データ取得中 ({api_info['name']}): {min_date} ~ {max_date}"
                    )

                    response = requests.get(
                        api_info["endpoint"],
                        params=api_info["params"],
                        timeout=30,
                        headers={"User-Agent": "DemandForecasting/1.0"},
                    )
                    response.raise_for_status()

                    weather_data = response.json()

                    # キャッシュに保存
                    date_range = f"{min_date} ~ {max_date}"
                    self._cache_weather_data(cache_key, api_info["name"], date_range, weather_data)

                # APIレスポンスの検証
                if "daily" not in weather_data or "time" not in weather_data["daily"]:
                    raise KeyError("不正なAPIレスポンス形式")

                # 基本気象データをDataFrameに変換
                weather_dict = {
                    "年月日": pd.to_datetime(weather_data["daily"]["time"]),
                    "平均気温": weather_data["daily"]["temperature_2m_mean"],
                    "最低気温": weather_data["daily"]["temperature_2m_min"],
                    "最高気温": weather_data["daily"]["temperature_2m_max"],
                    "降水量": weather_data["daily"]["precipitation_sum"],
                }

                # 追加データがあれば取得（湿度、風速）
                if "relative_humidity_2m_mean" in weather_data["daily"]:
                    weather_dict["湿度"] = weather_data["daily"]["relative_humidity_2m_mean"]
                if "wind_speed_10m_mean" in weather_data["daily"]:
                    weather_dict["風速"] = weather_data["daily"]["wind_speed_10m_mean"]

                weather_df = pd.DataFrame(weather_dict)

                # 日付でマージ
                df_weather["年月日_date"] = df_weather["年月日"].dt.date
                weather_df["年月日_date"] = weather_df["年月日"].dt.date

                merge_columns = ["年月日_date", "平均気温", "最低気温", "最高気温", "降水量"]
                if "湿度" in weather_df.columns:
                    merge_columns.append("湿度")
                if "風速" in weather_df.columns:
                    merge_columns.append("風速")

                df_weather = df_weather.merge(
                    weather_df[merge_columns], on="年月日_date", how="left"
                )

                # 一時列を削除
                df_weather.drop("年月日_date", axis=1, inplace=True)

                # 気象関連特徴量を追加
                df_weather = self._add_advanced_weather_features(df_weather)

                self.logger.info(f"気象データ取得・統合完了 ({api_info['name']})")
                return df_weather

            except requests.RequestException as e:
                self.logger.warning(f"気象API取得失敗 ({api_info['name']}): {e}")
                continue
            except (KeyError, ValueError) as e:
                self.logger.warning(f"気象APIレスポンス解析失敗 ({api_info['name']}): {e}")
                continue

        # 全てのAPIが失敗した場合
        raise ExternalAPIError("全ての気象APIからのデータ取得に失敗しました")

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

    def _add_advanced_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度な気象関連特徴量を追加"""
        df_weather_features = df.copy()

        # 基本的な気象特徴量
        self.logger.info("基本気象特徴量を生成中...")

        # 気温区分（より詳細）
        df_weather_features["気温区分"] = pd.cut(
            df_weather_features["平均気温"],
            bins=[-float("inf"), 0, 10, 20, 30, float("inf")],
            labels=["極寒", "寒い", "涼しい", "暖かい", "暑い"],
        )

        # 雨の日フラグ（強度別）
        df_weather_features["雨の日フラグ"] = (df_weather_features["降水量"] > 1.0).astype(int)
        df_weather_features["大雨フラグ"] = (df_weather_features["降水量"] > 10.0).astype(int)

        # 気温変化フラグ
        df_weather_features["急激な気温変化フラグ"] = (
            abs(df_weather_features["最高気温"] - df_weather_features["最低気温"]) > 10
        ).astype(int)

        # 時系列気象特徴量の生成
        if len(df_weather_features) > 1:
            df_weather_features = df_weather_features.sort_values("年月日")

            # 気温の移動平均（3日、7日、14日）
            for window in [3, 7, 14]:
                df_weather_features[f"気温_移動平均_{window}日"] = (
                    df_weather_features["平均気温"].rolling(window=window, min_periods=1).mean()
                )
                df_weather_features[f"降水量_移動平均_{window}日"] = (
                    df_weather_features["降水量"].rolling(window=window, min_periods=1).mean()
                )

            # 気温変化率（前日比）
            df_weather_features["気温_前日比変化"] = df_weather_features["平均気温"].diff()
            df_weather_features["気温_前日比変化率"] = (
                df_weather_features["気温_前日比変化"] / df_weather_features["平均気温"].shift(1)
            ).fillna(0)

            # 気温トレンド（7日間の傾き）
            df_weather_features["気温_7日トレンド"] = (
                df_weather_features["平均気温"]
                .rolling(window=7, min_periods=3)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0)
            ).fillna(0)

        # 複合指標の計算
        self.logger.info("複合気象指標を生成中...")

        # 体感温度（湿度考慮、簡易版）
        if "湿度" in df_weather_features.columns:
            df_weather_features["体感温度"] = (
                df_weather_features["平均気温"] + (df_weather_features["湿度"] - 50) * 0.1
            )
        else:
            df_weather_features["体感温度"] = df_weather_features["平均気温"]

        # 不快指数（温度と湿度から）
        if "湿度" in df_weather_features.columns:
            df_weather_features["不快指数"] = (
                0.81 * df_weather_features["平均気温"]
                + 0.01
                * df_weather_features["湿度"]
                * (0.99 * df_weather_features["平均気温"] - 14.3)
                + 46.3
            )
            df_weather_features["不快指数_高フラグ"] = (
                df_weather_features["不快指数"] > 75
            ).astype(int)

        # 風冷却効果（風速考慮）
        if "風速" in df_weather_features.columns:
            # 風冷指数（Wind Chill Index）の簡易版
            df_weather_features["風冷指数"] = (
                df_weather_features["平均気温"] - df_weather_features["風速"] * 0.5
            )

        # 季節調整済み気温偏差
        if "月" in df_weather_features.columns:
            # 月別平均気温を計算
            monthly_avg_temp = df_weather_features.groupby("月")["平均気温"].transform("mean")
            df_weather_features["気温_季節偏差"] = (
                df_weather_features["平均気温"] - monthly_avg_temp
            )

        # 商品カテゴリ別影響フラグ
        self.logger.info("商品カテゴリ別気象影響フラグを生成中...")

        # 冷たい商品需要（高温時）
        df_weather_features["冷商品需要_高フラグ"] = (df_weather_features["平均気温"] > 25).astype(
            int
        )
        df_weather_features["冷商品需要_中フラグ"] = (
            (df_weather_features["平均気温"] > 20) & (df_weather_features["平均気温"] <= 25)
        ).astype(int)

        # 温かい商品需要（低温時）
        df_weather_features["温商品需要_高フラグ"] = (df_weather_features["平均気温"] < 10).astype(
            int
        )
        df_weather_features["温商品需要_中フラグ"] = (
            (df_weather_features["平均気温"] >= 10) & (df_weather_features["平均気温"] < 20)
        ).astype(int)

        # 雨の日商品需要
        df_weather_features["雨日商品需要フラグ"] = (df_weather_features["降水量"] > 0.5).astype(
            int
        )

        # 気象ストレス指標（極端な気象条件）
        df_weather_features["気象ストレス指標"] = (
            (df_weather_features["平均気温"] < 5).astype(int) * 2  # 極寒
            + (df_weather_features["平均気温"] > 30).astype(int) * 2  # 猛暑
            + (df_weather_features["降水量"] > 20).astype(int)  # 大雨
            + df_weather_features["急激な気温変化フラグ"]  # 気温変化
        )

        self.logger.info(
            f"高度気象特徴量生成完了: {len([c for c in df_weather_features.columns if '気温' in c or '降水' in c or '湿度' in c or '風' in c])}個の気象関連特徴量"
        )

        return df_weather_features

    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """従来の気象関連特徴量を追加（後方互換性）"""
        return self._add_advanced_weather_features(df)

    def select_features(self, df: pd.DataFrame, target: str, max_features: int = 25) -> List[str]:
        """
        特徴量選択を実行

        Args:
            df: 入力DataFrame
            target: ターゲット列名
            max_features: 最大特徴量数（デフォルト25個）

        Returns:
            選択された特徴量リスト
        """
        self.logger.info(f"特徴量選択を開始 (最大{max_features}個)")

        try:
            # 数値特徴量のみを対象
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

            # ターゲット列を除外
            if target in numeric_features:
                numeric_features.remove(target)

            self.logger.info(f"初期特徴量数: {len(numeric_features)}")

            # 相関分析による特徴量選択
            correlation_matrix = df[numeric_features + [target]].corr()
            target_correlation = abs(correlation_matrix[target]).sort_values(ascending=False)

            # 統一された相関閾値
            correlation_threshold = 0.1

            selected_features = []

            # 全特徴量を同じ条件で選択
            for feature in numeric_features:
                if target_correlation.get(feature, 0) > correlation_threshold:
                    selected_features.append(feature)

            self.logger.info(f"相関選択後: {len(selected_features)}個")

            # 高相関の特徴量ペアを除去
            selected_features = self._remove_highly_correlated_features(
                df, selected_features, threshold=0.75
            )

            self.logger.info(f"高相関除去後: {len(selected_features)}個")

            # 最大特徴量数制限（相関順）
            if len(selected_features) > max_features:
                selected_features_with_corr = [
                    (f, target_correlation.get(f, 0)) for f in selected_features
                ]
                selected_features_with_corr.sort(key=lambda x: x[1], reverse=True)
                selected_features = [f for f, _ in selected_features_with_corr[:max_features]]
                self.logger.info(f"最大数制限適用: {max_features}個に削減")

            self.logger.info(f"最終選択特徴量数: {len(selected_features)}個")

            return selected_features

        except Exception as e:
            raise FeatureEngineeringError(f"特徴量選択エラー: {e}")

    def _remove_highly_correlated_features_with_weather_protection(
        self,
        df: pd.DataFrame,
        features: List[str],
        threshold: float = 0.75,
        weather_features: List[str] = None,
    ) -> List[str]:
        """高相関特徴量除去（気象特徴量保護版）"""
        if len(features) <= 1:
            return features

        if weather_features is None:
            weather_features = []

        correlation_matrix = df[features].corr().abs()

        # 上三角行列を取得
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # 高相関ペアを特定し、気象特徴量を保護しながら除去
        to_drop = set()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if upper_triangle.iloc[i, j] > threshold:
                    feature_i, feature_j = features[i], features[j]

                    # 気象特徴量を優先保持
                    if feature_i in weather_features and feature_j not in weather_features:
                        to_drop.add(feature_j)
                    elif feature_j in weather_features and feature_i not in weather_features:
                        to_drop.add(feature_i)
                    else:
                        # 両方とも気象特徴量 or 両方とも非気象特徴量の場合は分散で判定
                        var_i = df[feature_i].var()
                        var_j = df[feature_j].var()
                        if var_i < var_j:
                            to_drop.add(feature_i)
                        else:
                            to_drop.add(feature_j)

        # 高相関特徴量を除去
        selected_features = [f for f in features if f not in to_drop]

        if to_drop:
            protected_weather = [f for f in weather_features if f in features and f not in to_drop]
            self.logger.info(
                f"高相関除去: {len(to_drop)}個除去, 気象特徴量{len(protected_weather)}個保護"
            )

        return selected_features

    def _prioritize_weather_features_in_selection(
        self,
        features: List[str],
        weather_features: List[str],
        target_correlation: pd.Series,
        max_features: int,
    ) -> List[str]:
        """気象特徴量を優先した特徴量選択"""
        weather_in_features = [f for f in features if f in weather_features]
        non_weather_in_features = [f for f in features if f not in weather_features]

        # 気象特徴量を相関順にソート
        weather_sorted = sorted(
            weather_in_features, key=lambda x: target_correlation.get(x, 0), reverse=True
        )

        # 非気象特徴量を相関順にソート
        non_weather_sorted = sorted(
            non_weather_in_features, key=lambda x: target_correlation.get(x, 0), reverse=True
        )

        # 気象特徴量を最低30%は確保（ただし利用可能数が上限）
        min_weather_features = min(len(weather_sorted), max(1, int(max_features * 0.3)))

        selected = []

        # 気象特徴量から選択
        selected.extend(weather_sorted[:min_weather_features])

        # 残り枠を非気象特徴量で埋める
        remaining_slots = max_features - len(selected)
        selected.extend(non_weather_sorted[:remaining_slots])

        # まだ枠があれば残りの気象特徴量を追加
        if len(selected) < max_features:
            remaining_weather = weather_sorted[min_weather_features:]
            remaining_slots = max_features - len(selected)
            selected.extend(remaining_weather[:remaining_slots])

        self.logger.info(
            f"優先選択結果: 気象{len([f for f in selected if f in weather_features])}個, "
            f"その他{len([f for f in selected if f not in weather_features])}個"
        )

        return selected

    def _remove_highly_correlated_features(
        self, df: pd.DataFrame, features: List[str], threshold: float = 0.75
    ) -> List[str]:
        """高相関の特徴量ペアを除去（強化版）"""
        if len(features) <= 1:
            return features

        correlation_matrix = df[features].corr().abs()

        # 上三角行列を取得
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # 高相関ペアを特定し、より情報量の少ない特徴量を除去
        to_drop = set()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if upper_triangle.iloc[i, j] > threshold:
                    # より分散の小さい特徴量を除去（情報量が少ない）
                    var_i = df[features[i]].var()
                    var_j = df[features[j]].var()
                    if var_i < var_j:
                        to_drop.add(features[i])
                    else:
                        to_drop.add(features[j])

        # 高相関特徴量を除去
        selected_features = [f for f in features if f not in to_drop]

        if to_drop:
            self.logger.info(f"高相関により除去された特徴量 (閾値{threshold}): {list(to_drop)}")
            self.logger.info(f"残存特徴量数: {len(selected_features)}")

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

    def add_lag_features(
        self, df: pd.DataFrame, target_column: str = "数量", lag_days: List[int] = [1, 2, 3, 7]
    ) -> pd.DataFrame:
        """
        ラグ特徴量（過去の売上）を追加

        Args:
            df: 入力DataFrame
            target_column: ターゲット列名
            lag_days: ラグ日数リスト

        Returns:
            ラグ特徴量追加後のDataFrame
        """
        self.logger.info(f"ラグ特徴量追加開始: {lag_days}日")

        df_lag = df.copy()

        try:
            # 日付でソート
            if "年月日" in df_lag.columns:
                df_lag = df_lag.sort_values("年月日")

            # 商品別にグループ化してラグ特徴量を作成
            if "商品名称" in df_lag.columns:
                for lag in lag_days:
                    lag_col_name = f"{target_column}_lag{lag}日"
                    df_lag[lag_col_name] = df_lag.groupby("商品名称")[target_column].shift(lag)

                    # 欠損値は0で埋める（商品の販売開始初期の場合）
                    df_lag[lag_col_name] = df_lag[lag_col_name].fillna(0)

                    self.logger.info(f"追加: {lag_col_name}")
            else:
                # 商品名称がない場合は全体でラグ特徴量を作成
                for lag in lag_days:
                    lag_col_name = f"{target_column}_lag{lag}日"
                    df_lag[lag_col_name] = df_lag[target_column].shift(lag).fillna(0)

            self.logger.info("ラグ特徴量追加完了")

        except Exception as e:
            raise FeatureEngineeringError(f"ラグ特徴量追加エラー: {e}")

        return df_lag

    def add_moving_average_features(
        self, df: pd.DataFrame, target_column: str = "数量", windows: List[int] = [3, 7, 14]
    ) -> pd.DataFrame:
        """
        移動平均特徴量を追加

        Args:
            df: 入力DataFrame
            target_column: ターゲット列名
            windows: 移動平均ウィンドウサイズリスト

        Returns:
            移動平均特徴量追加後のDataFrame
        """
        self.logger.info(f"移動平均特徴量追加開始: {windows}日")

        df_ma = df.copy()

        try:
            # 日付でソート
            if "年月日" in df_ma.columns:
                df_ma = df_ma.sort_values("年月日")

            # 商品別にグループ化して移動平均を作成
            if "商品名称" in df_ma.columns:
                for window in windows:
                    ma_col_name = f"{target_column}_移動平均{window}日"
                    df_ma[ma_col_name] = (
                        df_ma.groupby("商品名称")[target_column]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )

                    # 移動平均の変化率も追加
                    change_col_name = f"{target_column}_移動平均{window}日_変化率"
                    df_ma[change_col_name] = (
                        df_ma.groupby("商品名称")[ma_col_name].pct_change().fillna(0)
                    )

                    self.logger.info(f"追加: {ma_col_name}, {change_col_name}")
            else:
                # 商品名称がない場合は全体で移動平均を作成
                for window in windows:
                    ma_col_name = f"{target_column}_移動平均{window}日"
                    df_ma[ma_col_name] = (
                        df_ma[target_column].rolling(window=window, min_periods=1).mean()
                    )

                    change_col_name = f"{target_column}_移動平均{window}日_変化率"
                    df_ma[change_col_name] = df_ma[ma_col_name].pct_change().fillna(0)

            self.logger.info("移動平均特徴量追加完了")

        except Exception as e:
            raise FeatureEngineeringError(f"移動平均特徴量追加エラー: {e}")

        return df_ma

    def add_advanced_time_series_features(
        self, df: pd.DataFrame, target_column: str = "数量"
    ) -> pd.DataFrame:
        """
        Phase 3: 高度な時系列特徴量を統合して追加

        Args:
            df: 入力DataFrame
            target_column: ターゲット列名

        Returns:
            高度な時系列特徴量追加後のDataFrame
        """
        self.logger.info("高度な時系列特徴量の統合開始")

        df_advanced = df.copy()

        try:
            # 1. ラグ特徴量追加（1-7日前の売上）
            df_advanced = self.add_lag_features(df_advanced, target_column, lag_days=[1, 2, 3, 7])

            # 2. 移動平均特徴量追加（3, 7, 14日）
            df_advanced = self.add_moving_average_features(
                df_advanced, target_column, windows=[3, 7, 14]
            )

            # 3. 季節性特徴量（周期性パターン）
            if "年月日" in df_advanced.columns:
                df_advanced["年間通算日"] = df_advanced["年月日"].dt.dayofyear
                df_advanced["月通算日"] = df_advanced["年月日"].dt.day

                # 季節性の正弦・余弦変換（周期性を捉える）
                df_advanced["年間周期_sin"] = np.sin(
                    2 * np.pi * df_advanced["年間通算日"] / 365.25
                )
                df_advanced["年間周期_cos"] = np.cos(
                    2 * np.pi * df_advanced["年間通算日"] / 365.25
                )
                df_advanced["月間周期_sin"] = np.sin(2 * np.pi * df_advanced["月通算日"] / 30.44)
                df_advanced["月間周期_cos"] = np.cos(2 * np.pi * df_advanced["月通算日"] / 30.44)

            # 4. 商品別集計特徴量
            if "商品名称" in df_advanced.columns:
                # 過去7日間の売上合計・平均・標準偏差
                for window in [7, 14]:
                    sum_col = f"{target_column}_過去{window}日間合計"
                    std_col = f"{target_column}_過去{window}日間標準偏差"

                    df_advanced[sum_col] = (
                        df_advanced.groupby("商品名称")[target_column]
                        .rolling(window=window, min_periods=1)
                        .sum()
                        .reset_index(0, drop=True)
                    )

                    df_advanced[std_col] = (
                        df_advanced.groupby("商品名称")[target_column]
                        .rolling(window=window, min_periods=1)
                        .std()
                        .fillna(0)
                        .reset_index(0, drop=True)
                    )

            # 5. トレンド特徴量
            if "商品名称" in df_advanced.columns:
                # 短期（3日）と長期（14日）のトレンド比較
                short_ma = f"{target_column}_移動平均3日"
                long_ma = f"{target_column}_移動平均14日"

                if short_ma in df_advanced.columns and long_ma in df_advanced.columns:
                    df_advanced["トレンド指標"] = df_advanced[short_ma] / (
                        df_advanced[long_ma] + 1e-8
                    )  # ゼロ除算回避

            self.logger.info("高度な時系列特徴量の統合完了")

        except Exception as e:
            raise FeatureEngineeringError(f"高度な時系列特徴量追加エラー: {e}")

        return df_advanced
