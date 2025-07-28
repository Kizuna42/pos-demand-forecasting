from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import chardet
import numpy as np
import pandas as pd

from ..utils.config import Config
from ..utils.exceptions import DataProcessingError
from ..utils.logger import Logger


class DataProcessor:
    """データ処理クラス"""

    def __init__(self, config: Config):
        """
        初期化

        Args:
            config: 設定管理オブジェクト
        """
        self.config = config
        self.logger = Logger(config.get_logging_config()).get_logger("data_processor")
        self.data_config = config.get_data_config()

    def detect_encoding(self, file_path: str) -> str:
        """
        ファイルの文字エンコーディングを自動検出

        Args:
            file_path: ファイルパス

        Returns:
            検出されたエンコーディング
        """
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # 最初の10KBを読み込み

            result = chardet.detect(raw_data)
            encoding = result["encoding"]
            confidence = result["confidence"]

            self.logger.info(f"文字エンコーディングを検出: {encoding} (信頼度: {confidence:.2f})")

            # 信頼度が低い場合はデフォルトのShift-JISを使用
            if confidence < 0.7:
                encoding = self.data_config.get("encoding", "shift_jis")
                self.logger.warning(
                    f"検出信頼度が低いため、デフォルトエンコーディングを使用: {encoding}"
                )

            return encoding

        except Exception as e:
            self.logger.error(f"エンコーディング検出エラー: {e}")
            default_encoding = self.data_config.get("encoding", "shift_jis")
            self.logger.info(f"デフォルトエンコーディングを使用: {default_encoding}")
            return default_encoding

    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        生データを読み込む

        Args:
            file_path: ファイルパス（指定しない場合は設定ファイルから取得）

        Returns:
            読み込まれたDataFrame
        """
        if file_path is None:
            file_path = self.data_config.get("raw_data_path")

        if not file_path:
            raise DataProcessingError("データファイルパスが指定されていません")

        file_path = Path(file_path)
        if not file_path.exists():
            raise DataProcessingError(f"データファイルが見つかりません: {file_path}")

        try:
            # エンコーディングを自動検出
            encoding = self.detect_encoding(str(file_path))

            # CSVファイルを読み込み
            self.logger.info(f"データファイルを読み込み: {file_path}")
            df = pd.read_csv(file_path, encoding=encoding)

            self.logger.info(f"データ読み込み完了: {len(df)} 行, {len(df.columns)} 列")
            self.logger.debug(f"列名: {list(df.columns)}")

            return df

        except Exception as e:
            raise DataProcessingError(f"データ読み込みエラー: {e}")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値を処理する

        Args:
            df: 入力DataFrame

        Returns:
            欠損値処理後のDataFrame
        """
        self.logger.info("欠損値処理を開始")

        # 欠損値の状況を確認
        missing_info = df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]

        if len(missing_cols) > 0:
            self.logger.info(f"欠損値が見つかりました:")
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"  {col}: {count} 件 ({percentage:.1f}%)")
        else:
            self.logger.info("欠損値は見つかりませんでした")
            return df

        df_processed = df.copy()

        # 数値列の欠損値を中央値で補完
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in missing_cols:
                median_val = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(median_val)
                self.logger.info(f"{col} の欠損値を中央値 {median_val} で補完")

        # カテゴリ列の欠損値を最頻値で補完
        categorical_cols = df_processed.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col in missing_cols:
                mode_val = df_processed[col].mode()
                if len(mode_val) > 0:
                    df_processed[col] = df_processed[col].fillna(mode_val[0])
                    self.logger.info(f"{col} の欠損値を最頻値 '{mode_val[0]}' で補完")

        self.logger.info("欠損値処理完了")
        return df_processed

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        重複データを除去する

        Args:
            df: 入力DataFrame

        Returns:
            重複除去後のDataFrame
        """
        self.logger.info("重複データ除去を開始")

        initial_count = len(df)
        df_deduplicated = df.drop_duplicates()
        final_count = len(df_deduplicated)
        removed_count = initial_count - final_count

        if removed_count > 0:
            self.logger.info(f"重複データを除去: {removed_count} 件")
        else:
            self.logger.info("重複データはありませんでした")

        return df_deduplicated

    def remove_outliers(
        self, df: pd.DataFrame, method: str = "iqr", columns: list = None
    ) -> pd.DataFrame:
        """
        外れ値を除去する

        Args:
            df: 入力DataFrame
            method: 外れ値検出方法 ('iqr' または 'zscore')
            columns: 対象列（指定しない場合は数値列すべて）

        Returns:
            外れ値除去後のDataFrame
        """
        self.logger.info(f"外れ値除去を開始 (方法: {method})")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df_processed = df.copy()
        initial_count = len(df_processed)

        if method == "iqr":
            for col in columns:
                if col in df_processed.columns:
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    before_count = len(df_processed)
                    df_processed = df_processed[
                        (df_processed[col] >= lower_bound) & (df_processed[col] <= upper_bound)
                    ]
                    after_count = len(df_processed)
                    removed = before_count - after_count

                    if removed > 0:
                        self.logger.info(f"{col}: {removed} 件の外れ値を除去")

        elif method == "zscore":
            for col in columns:
                if col in df_processed.columns:
                    z_scores = np.abs(
                        (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
                    )

                    before_count = len(df_processed)
                    df_processed = df_processed[z_scores < 3]
                    after_count = len(df_processed)
                    removed = before_count - after_count

                    if removed > 0:
                        self.logger.info(f"{col}: {removed} 件の外れ値を除去")

        total_removed = initial_count - len(df_processed)
        self.logger.info(f"外れ値除去完了: 合計 {total_removed} 件を除去")

        return df_processed

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本特徴量を生成する

        Args:
            df: 入力DataFrame

        Returns:
            基本特徴量追加後のDataFrame
        """
        self.logger.info("基本特徴量生成を開始")

        df_features = df.copy()

        try:
            # 日付列の処理
            if "年月日" in df_features.columns:
                # 日本語形式の日付を適切に解析
                try:
                    # まず標準的な解析を試行
                    df_features["年月日"] = pd.to_datetime(df_features["年月日"])
                except (ValueError, pd.errors.ParserError):
                    # 日本語形式の日付を変換（例: "2024年1月2日" -> "2024-01-02"）
                    self.logger.info("日本語形式の日付を検出、変換処理を実行")
                    date_series = df_features["年月日"].astype(str)

                    # 正規表現で日本語日付を抽出・変換
                    import re

                    def parse_japanese_date(date_str):
                        try:
                            # "2024年1月2日" のような形式をマッチ
                            match = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", str(date_str))
                            if match:
                                year, month, day = match.groups()
                                return f"{year}-{int(month):02d}-{int(day):02d}"
                            return date_str
                        except:
                            return date_str

                    converted_dates = date_series.apply(parse_japanese_date)
                    df_features["年月日"] = pd.to_datetime(converted_dates)

                # 月、曜日、週末フラグを作成
                df_features["月"] = df_features["年月日"].dt.month
                df_features["曜日"] = df_features["年月日"].dt.dayofweek
                df_features["週末フラグ"] = (df_features["曜日"] >= 5).astype(int)

                self.logger.info("日付関連特徴量を生成: 月, 曜日, 週末フラグ")

            # 売上単価の計算
            if "金額" in df_features.columns and "数量" in df_features.columns:
                # ゼロ除算を避ける
                df_features["売上単価"] = df_features["金額"] / df_features["数量"].replace(
                    0, np.nan
                )
                df_features["売上単価"] = df_features["売上単価"].fillna(0)

                self.logger.info("売上単価を計算")

            # 商品カテゴリの処理（商品名称から推定）
            if "商品名称" in df_features.columns:
                df_features = self._categorize_products(df_features)

            self.logger.info("基本特徴量生成完了")

        except Exception as e:
            raise DataProcessingError(f"基本特徴量生成エラー: {e}")

        return df_features

    def _categorize_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        商品名称から商品カテゴリを推定

        Args:
            df: 入力DataFrame

        Returns:
            カテゴリ情報追加後のDataFrame
        """
        df_categorized = df.copy()

        # 生鮮食品のカテゴリ定義
        category_keywords = {
            "野菜": [
                "キャベツ",
                "レタス",
                "トマト",
                "きゅうり",
                "玉ねぎ",
                "じゃがいも",
                "にんじん",
                "ほうれん草",
                "白菜",
                "もやし",
                "ピーマン",
                "なす",
            ],
            "肉類": ["牛肉", "豚肉", "鶏肉", "ひき肉", "ソーセージ", "ハム", "ベーコン"],
            "魚類": ["まぐろ", "さけ", "あじ", "さば", "いわし", "海老", "いか", "たこ"],
            "果物": ["りんご", "みかん", "バナナ", "いちご", "ぶどう", "もも", "なし"],
        }

        def categorize_product(product_name):
            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if keyword in product_name:
                        return category
            return "その他"

        df_categorized["商品カテゴリ"] = df_categorized["商品名称"].apply(categorize_product)

        category_counts = df_categorized["商品カテゴリ"].value_counts()
        self.logger.info(f"商品カテゴリ分布: {dict(category_counts)}")

        return df_categorized

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データクリーニングの統合処理

        Args:
            df: 入力DataFrame

        Returns:
            クリーニング後のDataFrame
        """
        self.logger.info("データクリーニングを開始")

        # 1. 重複除去
        df_clean = self.remove_duplicates(df)

        # 2. 欠損値処理
        df_clean = self.handle_missing_values(df_clean)

        # 3. 外れ値除去
        df_clean = self.remove_outliers(df_clean, method="iqr")

        # 4. 基本特徴量生成
        df_clean = self.create_basic_features(df_clean)

        self.logger.info("データクリーニング完了")
        return df_clean

    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv") -> str:
        """
        処理済みデータを保存

        Args:
            df: 保存するDataFrame
            filename: ファイル名

        Returns:
            保存先ファイルパス
        """
        processed_dir = Path(self.data_config.get("processed_data_path", "data/processed"))
        processed_dir.mkdir(parents=True, exist_ok=True)

        output_path = processed_dir / filename

        try:
            df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"処理済みデータを保存: {output_path}")
            return str(output_path)

        except Exception as e:
            raise DataProcessingError(f"データ保存エラー: {e}")
