from pathlib import Path
from typing import List, Optional

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
            self.logger.info("欠損値が見つかりました:")
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

    def remove_outliers_advanced(
        self,
        df: pd.DataFrame,
        method: str = "iqr_enhanced",
        columns: list = None,
        iqr_multiplier: float = 1.5,
        group_by_column: str = None,
    ) -> pd.DataFrame:
        """
        Phase 3: 強化された外れ値除去

        Args:
            df: 入力DataFrame
            method: 外れ値検出方法 ('iqr_enhanced', 'zscore_robust', 'isolation_forest')
            columns: 対象列（指定しない場合は数値列すべて）
            iqr_multiplier: IQR法の倍率（デフォルト1.5、厳しくするなら1.0-1.2）
            group_by_column: グループ別外れ値検出用の列名（商品名称など）

        Returns:
            外れ値除去後のDataFrame
        """
        self.logger.info(f"強化外れ値除去を開始 (方法: {method})")

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        df_processed = df.copy()
        initial_count = len(df_processed)

        if method == "iqr_enhanced":
            # グループ別IQR法（商品ごとに外れ値を検出）
            if group_by_column and group_by_column in df_processed.columns:
                outlier_mask = pd.Series([False] * len(df_processed), index=df_processed.index)

                for group_name, group_data in df_processed.groupby(group_by_column):
                    group_outlier_mask = pd.Series(
                        [False] * len(group_data), index=group_data.index
                    )

                    for col in columns:
                        if (
                            col in group_data.columns and len(group_data) > 4
                        ):  # 最低5個のデータが必要
                            Q1 = group_data[col].quantile(0.25)
                            Q3 = group_data[col].quantile(0.75)
                            IQR = Q3 - Q1

                            if IQR > 0:  # IQRが0より大きい場合のみ実行
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR

                                col_outliers = (group_data[col] < lower_bound) | (
                                    group_data[col] > upper_bound
                                )
                                group_outlier_mask |= col_outliers

                                outlier_count = col_outliers.sum()
                                if outlier_count > 0:
                                    self.logger.info(
                                        f"{group_name} - {col}: {outlier_count} 件の外れ値を検出"
                                    )

                    outlier_mask[group_data.index] = group_outlier_mask

                # 外れ値を除去
                df_processed = df_processed[~outlier_mask]

            else:
                # 全体でのIQR法（従来の方法だが倍率調整可能）
                for col in columns:
                    if col in df_processed.columns:
                        Q1 = df_processed[col].quantile(0.25)
                        Q3 = df_processed[col].quantile(0.75)
                        IQR = Q3 - Q1

                        if IQR > 0:
                            lower_bound = Q1 - iqr_multiplier * IQR
                            upper_bound = Q3 + iqr_multiplier * IQR

                            before_count = len(df_processed)
                            df_processed = df_processed[
                                (df_processed[col] >= lower_bound)
                                & (df_processed[col] <= upper_bound)
                            ]
                            after_count = len(df_processed)
                            removed = before_count - after_count

                            if removed > 0:
                                self.logger.info(f"{col}: {removed} 件の外れ値を除去")

        elif method == "zscore_robust":
            # より堅牢なZ-Score法（中央値とMADを使用）
            if group_by_column and group_by_column in df_processed.columns:
                outlier_mask = pd.Series([False] * len(df_processed), index=df_processed.index)

                for group_name, group_data in df_processed.groupby(group_by_column):
                    group_outlier_mask = pd.Series(
                        [False] * len(group_data), index=group_data.index
                    )

                    for col in columns:
                        if col in group_data.columns and len(group_data) > 4:
                            # 中央値とMAD（Median Absolute Deviation）を使用
                            median = group_data[col].median()
                            mad = np.median(np.abs(group_data[col] - median))

                            if mad > 0:
                                modified_z_scores = 0.6745 * (group_data[col] - median) / mad
                                col_outliers = np.abs(modified_z_scores) > 3.5
                                group_outlier_mask |= col_outliers

                                outlier_count = col_outliers.sum()
                                if outlier_count > 0:
                                    self.logger.info(
                                        f"{group_name} - {col}: {outlier_count} 件の外れ値を検出"
                                    )

                    outlier_mask[group_data.index] = group_outlier_mask

                df_processed = df_processed[~outlier_mask]
            else:
                # 全体での堅牢Z-Score法
                for col in columns:
                    if col in df_processed.columns:
                        median = df_processed[col].median()
                        mad = np.median(np.abs(df_processed[col] - median))

                        if mad > 0:
                            modified_z_scores = 0.6745 * (df_processed[col] - median) / mad
                            outliers = np.abs(modified_z_scores) > 3.5

                            before_count = len(df_processed)
                            df_processed = df_processed[~outliers]
                            after_count = len(df_processed)
                            removed = before_count - after_count

                            if removed > 0:
                                self.logger.info(f"{col}: {removed} 件の外れ値を除去")

        elif method == "isolation_forest":
            # Isolation Forestを使用した多変量外れ値検出
            try:
                from sklearn.ensemble import IsolationForest

                # 数値列のみを選択
                numeric_data = df_processed[columns].select_dtypes(include=[np.number])

                if len(numeric_data.columns) > 0 and len(numeric_data) > 10:
                    # 欠損値を中央値で埋める
                    numeric_data_filled = numeric_data.fillna(numeric_data.median())

                    # Isolation Forest実行
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(numeric_data_filled)

                    # 外れ値（-1）を除去
                    inlier_mask = outlier_labels == 1
                    before_count = len(df_processed)
                    df_processed = df_processed[inlier_mask]
                    after_count = len(df_processed)
                    removed = before_count - after_count

                    self.logger.info(f"Isolation Forest: {removed} 件の多変量外れ値を除去")
                else:
                    self.logger.warning("Isolation Forest: データ不足のためスキップ")

            except ImportError:
                self.logger.warning("scikit-learn が利用できないため、Isolation Forest をスキップ")

        total_removed = initial_count - len(df_processed)
        removal_rate = (total_removed / initial_count) * 100 if initial_count > 0 else 0

        self.logger.info(
            f"強化外れ値除去完了: 合計 {total_removed} 件を除去 ({removal_rate:.2f}%)"
        )

        # 除去率が異常に高い場合は警告
        if removal_rate > 20:
            self.logger.warning(
                f"外れ値除去率が高すぎます ({removal_rate:.2f}%)。パラメータの見直しを推奨します。"
            )

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
                        except Exception:
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

        # 3. Phase 3: 強化された外れ値除去（商品別IQR法）
        df_clean = self.remove_outliers_advanced(
            df_clean,
            method="iqr_enhanced",
            iqr_multiplier=1.2,  # より厳しい閾値
            group_by_column="商品名称",
        )

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

    def stratified_product_sampling(
        self, df: pd.DataFrame, max_products: int = 150, min_products_per_category: int = 10
    ) -> List[str]:
        """
        層化サンプリングで代表商品を選択

        Args:
            df: 入力DataFrame
            max_products: 最大商品数
            min_products_per_category: カテゴリ別最低商品数

        Returns:
            選択された商品名リスト
        """
        try:
            self.logger.info("層化サンプリングによる商品選択を開始")

            # 商品別の基本統計を計算
            product_stats = (
                df.groupby("商品名称")
                .agg(
                    {
                        "数量": ["sum", "mean", "std", "count"],
                        "金額": ["sum", "mean"],
                        "商品カテゴリ": "first",
                        "年月日": ["min", "max"],
                    }
                )
                .reset_index()
            )

            # 列名を平坦化
            product_stats.columns = [
                "商品名称",
                "総販売数量",
                "平均販売数量",
                "販売数量標準偏差",
                "レコード数",
                "総売上金額",
                "平均売上金額",
                "商品カテゴリ",
                "最初販売日",
                "最終販売日",
            ]

            # データ期間を計算
            product_stats["販売期間_日"] = (
                product_stats["最終販売日"] - product_stats["最初販売日"]
            ).dt.days + 1

            # 売上規模でカテゴライズ（大・中・小）
            total_revenue_quantiles = product_stats["総売上金額"].quantile([0.33, 0.67])

            def categorize_revenue_size(revenue):
                if revenue >= total_revenue_quantiles[0.67]:
                    return "大規模"
                elif revenue >= total_revenue_quantiles[0.33]:
                    return "中規模"
                else:
                    return "小規模"

            product_stats["売上規模"] = product_stats["総売上金額"].apply(categorize_revenue_size)

            # カテゴリ別の商品数を確認
            category_counts = product_stats["商品カテゴリ"].value_counts()
            self.logger.info(f"カテゴリ別商品数: {dict(category_counts)}")

            selected_products = []

            # カテゴリ別に層化サンプリング
            for category in category_counts.index:
                category_products = product_stats[product_stats["商品カテゴリ"] == category]

                # カテゴリの割当商品数を計算（全体に占める比率に基づく）
                category_ratio = len(category_products) / len(product_stats)
                target_count = max(min_products_per_category, int(max_products * category_ratio))
                target_count = min(target_count, len(category_products))

                # 売上規模別にバランスよく選択
                size_groups = category_products.groupby("売上規模")
                category_selected = []

                for size_group_name, size_group in size_groups:
                    # 売上規模グループ別の割当数
                    size_ratio = len(size_group) / len(category_products)
                    size_target = max(1, int(target_count * size_ratio))
                    size_target = min(size_target, len(size_group))

                    # データ品質が高い商品を優先選択
                    # 評価基準: レコード数, 販売期間, 売上安定性
                    size_group_scored = size_group.copy()

                    # スコア計算（正規化）
                    size_group_scored["レコード数_正規化"] = (
                        size_group_scored["レコード数"] / size_group_scored["レコード数"].max()
                    )
                    size_group_scored["販売期間_正規化"] = (
                        size_group_scored["販売期間_日"] / size_group_scored["販売期間_日"].max()
                    )
                    size_group_scored["安定性スコア"] = 1 / (
                        1
                        + size_group_scored["販売数量標準偏差"].fillna(0)
                        / size_group_scored["平均販売数量"].replace(0, 1)
                    )

                    # 総合スコア（重み付き平均）
                    size_group_scored["総合スコア"] = (
                        0.4 * size_group_scored["レコード数_正規化"]
                        + 0.3 * size_group_scored["販売期間_正規化"]
                        + 0.3 * size_group_scored["安定性スコア"]
                    )

                    # 上位商品を選択
                    top_products = size_group_scored.nlargest(size_target, "総合スコア")
                    category_selected.extend(top_products["商品名称"].tolist())

                selected_products.extend(category_selected)

                self.logger.info(
                    f"カテゴリ '{category}': {len(category_selected)}商品選択 "
                    f"(大規模:{len([p for p in category_selected if p in category_products[category_products['売上規模']=='大規模']['商品名称'].values])}, "
                    f"中規模:{len([p for p in category_selected if p in category_products[category_products['売上規模']=='中規模']['商品名称'].values])}, "
                    f"小規模:{len([p for p in category_selected if p in category_products[category_products['売上規模']=='小規模']['商品名称'].values])})"
                )

            # 重複除去
            selected_products = list(set(selected_products))

            # 最大数を超えた場合は総合スコア上位を選択
            if len(selected_products) > max_products:
                all_selected_stats = product_stats[
                    product_stats["商品名称"].isin(selected_products)
                ]
                # 総合スコア再計算
                all_selected_stats = all_selected_stats.copy()
                all_selected_stats["レコード数_正規化"] = (
                    all_selected_stats["レコード数"] / all_selected_stats["レコード数"].max()
                )
                all_selected_stats["販売期間_正規化"] = (
                    all_selected_stats["販売期間_日"] / all_selected_stats["販売期間_日"].max()
                )
                all_selected_stats["安定性スコア"] = 1 / (
                    1
                    + all_selected_stats["販売数量標準偏差"].fillna(0)
                    / all_selected_stats["平均販売数量"].replace(0, 1)
                )
                all_selected_stats["総合スコア"] = (
                    0.4 * all_selected_stats["レコード数_正規化"]
                    + 0.3 * all_selected_stats["販売期間_正規化"]
                    + 0.3 * all_selected_stats["安定性スコア"]
                )

                top_products = all_selected_stats.nlargest(max_products, "総合スコア")
                selected_products = top_products["商品名称"].tolist()

            # 最終結果をログ出力
            final_category_dist = product_stats[product_stats["商品名称"].isin(selected_products)][
                "商品カテゴリ"
            ].value_counts()

            final_size_dist = product_stats[product_stats["商品名称"].isin(selected_products)][
                "売上規模"
            ].value_counts()

            self.logger.info(
                f"層化サンプリング完了: {len(selected_products)}商品選択\n"
                f"カテゴリ分布: {dict(final_category_dist)}\n"
                f"売上規模分布: {dict(final_size_dist)}"
            )

            return selected_products

        except Exception as e:
            self.logger.error(f"層化サンプリングエラー: {e}")
            # フォールバック: 売上上位商品を返す
            fallback_products = (
                df.groupby("商品名称")["金額"].sum().nlargest(max_products).index.tolist()
            )
            self.logger.warning(f"フォールバック処理: 売上上位{len(fallback_products)}商品を選択")
            return fallback_products
