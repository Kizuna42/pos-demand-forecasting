from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.exceptions import ModelBuildingError


class ModelBuilder:
    """機械学習モデル構築クラス"""

    def __init__(self, config: Config):
        """
        初期化

        Args:
            config: 設定管理オブジェクト
        """
        self.config = config
        self.logger = Logger(config.get_logging_config()).get_logger("model_builder")
        self.model_config = config.get_model_config()
        self.scaler = StandardScaler()
        self.model = None

    def build_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        """
        モデルを構築

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries

        Returns:
            訓練済みRandomForestRegressor
        """
        self.logger.info("モデル構築を開始")

        try:
            # モデルパラメータ
            n_estimators = self.model_config.get("n_estimators", 100)
            max_depth = self.model_config.get("max_depth", 10)
            random_state = self.model_config.get("random_state", 42)

            # RandomForestRegressorを初期化
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )

            # 特徴量の前処理
            X_processed = self._preprocess_features(X)

            # モデル訓練
            model.fit(X_processed, y)

            self.model = model
            self.logger.info(f"モデル構築完了: n_estimators={n_estimators}, max_depth={max_depth}")

            return model

        except Exception as e:
            raise ModelBuildingError(f"モデル構築エラー: {e}")

    def train_with_cv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        交差検証付きでモデルを訓練

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries

        Returns:
            訓練結果辞書
        """
        self.logger.info("交差検証付きモデル訓練を開始")

        try:
            # データの前処理
            X_processed = self._preprocess_features(X)

            # データを訓練・テストに分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42
            )

            # モデル構築
            model = self.build_model(pd.DataFrame(X_train, columns=X_processed.columns), y_train)

            # 交差検証の実行
            cv_results = self._perform_cross_validation(X_processed, y)

            # テストデータでの評価
            test_results = self.evaluate_model(model, X_test, y_test)

            # 訓練データでの評価（過学習検出用）
            train_results = self.evaluate_model(model, X_train, y_train)

            # 過学習スコア計算
            overfitting_score = self.detect_overfitting(
                train_results["r2_score"], test_results["r2_score"]
            )

            # 特徴量重要度計算
            feature_importance = self.get_feature_importance(model, X_processed.columns.tolist())

            # 結果を統合
            results = {
                "model": model,
                "cv_scores": cv_results,
                "train_metrics": train_results,
                "test_metrics": test_results,
                "overfitting_score": overfitting_score,
                "feature_importance": feature_importance,
                "feature_names": X_processed.columns.tolist(),
            }

            self.logger.info("交差検証付きモデル訓練完了")
            return results

        except Exception as e:
            raise ModelBuildingError(f"交差検証付き訓練エラー: {e}")

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量の前処理

        Args:
            X: 入力特徴量DataFrame

        Returns:
            前処理済み特徴量DataFrame
        """
        X_processed = X.copy()

        # 無限値・NaN値の処理
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        X_processed = X_processed.fillna(X_processed.median())

        # 数値列のみを選択
        numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
        X_processed = X_processed[numeric_columns]

        # 標準化は行わない（RandomForestは標準化不要）

        return X_processed

    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        交差検証を実行

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries

        Returns:
            交差検証結果
        """
        cv_folds = self.model_config.get("cv_folds", 5)

        # KFold設定
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

        # モデル初期化
        model = RandomForestRegressor(
            n_estimators=self.model_config.get("n_estimators", 100),
            max_depth=self.model_config.get("max_depth", 10),
            random_state=self.model_config.get("random_state", 42),
            n_jobs=-1,
        )

        # 交差検証スコア計算
        cv_scores = cross_val_score(model, X, y, cv=kfold, scoring="r2")

        cv_results = {
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "scores": cv_scores.tolist(),
            "folds": cv_folds,
        }

        self.logger.info(
            f"交差検証結果: R² = {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}"
        )

        return cv_results

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        モデル性能を評価

        Args:
            model: 訓練済みモデル
            X_test: テスト特徴量
            y_test: テストターゲット

        Returns:
            評価メトリクス辞書
        """
        try:
            # 予測実行
            y_pred = model.predict(X_test)

            # メトリクス計算
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # 予測値の統計
            pred_mean = np.mean(y_pred)
            pred_std = np.std(y_pred)

            metrics = {
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae,
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "n_samples": len(y_test),
            }

            self.logger.info(f"モデル評価: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

            return metrics

        except Exception as e:
            raise ModelBuildingError(f"モデル評価エラー: {e}")

    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        特徴量重要度を取得

        Args:
            model: 訓練済みモデル
            feature_names: 特徴量名リスト

        Returns:
            特徴量重要度辞書
        """
        try:
            # RandomForestの特徴量重要度を取得
            importances = model.feature_importances_

            # 特徴量名と重要度をペアにしてソート
            feature_importance = dict(zip(feature_names, importances))
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )

            self.logger.info(f"特徴量重要度上位5位: {list(sorted_importance.keys())[:5]}")

            return sorted_importance

        except Exception as e:
            raise ModelBuildingError(f"特徴量重要度取得エラー: {e}")

    def calculate_shap_values(
        self, model: Any, X: pd.DataFrame, max_samples: int = 100
    ) -> np.ndarray:
        """
        SHAP値を計算

        Args:
            model: 訓練済みモデル
            X: 特徴量DataFrame
            max_samples: 計算に使用する最大サンプル数

        Returns:
            SHAP値配列
        """
        try:
            self.logger.info("SHAP値計算を開始")

            # サンプル数を制限（計算時間短縮のため）
            if len(X) > max_samples:
                X_sample = X.sample(n=max_samples, random_state=42)
            else:
                X_sample = X

            # TreeExplainerを使用（RandomForest用）
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            self.logger.info("SHAP値計算完了")

            return shap_values

        except Exception as e:
            self.logger.warning(f"SHAP値計算エラー: {e}")
            return np.array([])

    def detect_overfitting(self, train_score: float, test_score: float) -> float:
        """
        過学習を検出

        Args:
            train_score: 訓練データスコア
            test_score: テストデータスコア

        Returns:
            過学習スコア（0-1、1に近いほど過学習）
        """
        # スコア差を過学習の指標とする
        score_diff = max(0, train_score - test_score)

        # 過学習閾値
        overfitting_threshold = self.config.get_quality_config().get("overfitting_threshold", 0.1)

        # 0-1にスケール
        overfitting_score = min(1.0, score_diff / overfitting_threshold)

        if overfitting_score > 0.5:
            self.logger.warning(f"過学習の兆候: 訓練={train_score:.4f}, テスト={test_score:.4f}")

        return overfitting_score

    def save_model(self, model: Any, filepath: str) -> str:
        """
        モデルを保存

        Args:
            model: 保存するモデル
            filepath: 保存先パス

        Returns:
            保存先ファイルパス
        """
        try:
            # 保存ディレクトリを作成
            save_path = Path(filepath)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # pickleで保存
            with open(save_path, "wb") as f:
                pickle.dump(model, f)

            self.logger.info(f"モデルを保存: {save_path}")

            return str(save_path)

        except Exception as e:
            raise ModelBuildingError(f"モデル保存エラー: {e}")

    def load_model(self, filepath: str) -> Any:
        """
        モデルを読み込み

        Args:
            filepath: モデルファイルパス

        Returns:
            読み込まれたモデル
        """
        try:
            with open(filepath, "rb") as f:
                model = pickle.load(f)

            self.logger.info(f"モデルを読み込み: {filepath}")

            return model

        except Exception as e:
            raise ModelBuildingError(f"モデル読み込みエラー: {e}")

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        予測を実行

        Args:
            model: 訓練済みモデル
            X: 特徴量DataFrame

        Returns:
            予測結果
        """
        try:
            # 特徴量前処理
            X_processed = self._preprocess_features(X)

            # 予測実行
            predictions = model.predict(X_processed)

            return predictions

        except Exception as e:
            raise ModelBuildingError(f"予測エラー: {e}")

    def get_model_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        モデルサマリーを生成

        Args:
            results: 訓練結果辞書

        Returns:
            モデルサマリー辞書
        """
        summary = {
            "model_type": "RandomForestRegressor",
            "cv_mean_r2": results["cv_scores"]["mean_score"],
            "cv_std_r2": results["cv_scores"]["std_score"],
            "test_r2": results["test_metrics"]["r2_score"],
            "test_rmse": results["test_metrics"]["rmse"],
            "test_mae": results["test_metrics"]["mae"],
            "overfitting_score": results["overfitting_score"],
            "n_features": len(results["feature_names"]),
            "top_features": list(results["feature_importance"].keys())[:5],
        }

        return summary
