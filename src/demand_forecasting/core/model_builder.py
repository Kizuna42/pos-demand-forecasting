from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real
    from skopt.utils import use_named_args

    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False

from ..utils.config import Config
from ..utils.exceptions import ModelBuildingError
from ..utils.logger import Logger


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
        self.ensemble_models = {}

        if not XGBOOST_AVAILABLE:
            self.logger.warning(
                "XGBoost がインストールされていません。RandomForest のみ使用します。"
            )

        if not BAYESIAN_OPT_AVAILABLE:
            self.logger.warning(
                "scikit-optimize がインストールされていません。ベイズ最適化は利用できません。"
            )

    def build_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = "ensemble") -> Any:
        """
        モデルを構築

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries
            model_type: モデルタイプ ("random_forest", "xgboost", "ensemble")

        Returns:
            訓練済みモデル（アンサンブルの場合は辞書）
        """
        self.logger.info(f"モデル構築を開始: {model_type}")

        try:
            # 特徴量の前処理
            X_processed = self._preprocess_features(X)

            if model_type == "random_forest":
                model = self._build_random_forest(X_processed, y)
                self.model = model
                return model
            elif model_type == "xgboost":
                if not XGBOOST_AVAILABLE:
                    self.logger.warning("XGBoost が利用できません。RandomForest を使用します。")
                    return self.build_model(X, y, "random_forest")
                model = self._build_xgboost(X_processed, y)
                self.model = model
                return model
            elif model_type == "ensemble":
                models = self._build_ensemble(X_processed, y)
                self.ensemble_models = models
                return models
            else:
                raise ValueError(f"サポートされていないモデルタイプ: {model_type}")

        except Exception as e:
            raise ModelBuildingError(f"モデル構築エラー: {e}")

    def train_with_cv(
        self, X: pd.DataFrame, y: pd.Series, model_type: str = "ensemble"
    ) -> Dict[str, Any]:
        """
        交差検証付きでモデルを訓練

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries
            model_type: モデルタイプ ("random_forest", "xgboost", "ensemble")

        Returns:
            訓練結果辞書
        """
        self.logger.info(f"交差検証付きモデル訓練を開始: {model_type}")

        try:
            # データの前処理
            X_processed = self._preprocess_features(X)

            # 時系列Hold-out分割（最新3ヶ月をテストセットに固定）
            # 時系列順序を維持した分割
            total_len = len(X_processed)
            test_size = int(total_len * 0.2)  # 最新20%をテストデータ
            val_size = int(total_len * 0.15)  # その前15%を検証データ

            # 最新データから逆算して分割点を決定
            test_start = total_len - test_size
            val_start = test_start - val_size

            # 時系列順序を保持した分割
            X_train = X_processed.iloc[:val_start].copy()
            y_train = y.iloc[:val_start].copy()
            X_val = X_processed.iloc[val_start:test_start].copy()
            y_val = y.iloc[val_start:test_start].copy()
            X_test = X_processed.iloc[test_start:].copy()
            y_test = y.iloc[test_start:].copy()

            self.logger.info(
                f"時系列分割: 訓練={len(X_train)}, 検証={len(X_val)}, テスト={len(X_test)}"
            )

            # モデル構築
            model = self.build_model(
                pd.DataFrame(X_train, columns=X_processed.columns), y_train, model_type
            )

            # 交差検証の実行
            cv_results = self._perform_cross_validation(X_processed, y)

            # 評価実行
            if model_type == "ensemble" and isinstance(model, dict):
                # アンサンブルモデルの場合

                # 重み最適化
                optimal_weights = self.optimize_ensemble_weights(
                    model, pd.DataFrame(X_val, columns=X_processed.columns), y_val
                )

                # テストデータでの評価
                test_pred = self.ensemble_predict(
                    model, pd.DataFrame(X_test, columns=X_processed.columns), optimal_weights
                )
                test_results = self._evaluate_predictions(y_test, test_pred)

                # 訓練データでの評価（過学習検出用）
                train_pred = self.ensemble_predict(
                    model, pd.DataFrame(X_train, columns=X_processed.columns), optimal_weights
                )
                train_results = self._evaluate_predictions(y_train, train_pred)

                # 特徴量重要度（RandomForestから取得）
                rf_model = model.get("random_forest")
                if rf_model:
                    feature_importance = self.get_feature_importance(
                        rf_model, X_processed.columns.tolist()
                    )
                else:
                    feature_importance = {}

                # 結果に追加情報
                results = {
                    "model": model,
                    "model_type": "ensemble",
                    "ensemble_weights": optimal_weights,
                    "cv_scores": cv_results,
                    "train_metrics": train_results,
                    "test_metrics": test_results,
                    "overfitting_score": self.detect_overfitting(
                        train_results["r2_score"], test_results["r2_score"]
                    ),
                    "feature_importance": feature_importance,
                    "feature_names": X_processed.columns.tolist(),
                }

            else:
                # 単一モデルの場合
                test_results = self.evaluate_model(model, X_test, y_test)
                train_results = self.evaluate_model(model, X_train, y_train)

                # 特徴量重要度計算
                feature_importance = self.get_feature_importance(
                    model, X_processed.columns.tolist()
                )

                results = {
                    "model": model,
                    "model_type": model_type,
                    "cv_scores": cv_results,
                    "train_metrics": train_results,
                    "test_metrics": test_results,
                    "overfitting_score": self.detect_overfitting(
                        train_results["r2_score"], test_results["r2_score"]
                    ),
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
        交差検証を実行（TimeSeriesSplit使用）

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries

        Returns:
            交差検証結果
        """
        cv_folds = self.model_config.get("cv_folds", 5)

        # Phase 2: TimeSeriesSplitを使用（時系列データに適した分割）
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # モデル初期化
        model = RandomForestRegressor(
            n_estimators=self.model_config.get("n_estimators", 100),
            max_depth=self.model_config.get("max_depth", 10),
            random_state=self.model_config.get("random_state", 42),
            n_jobs=-1,
        )

        # TimeSeriesSplitで交差検証スコア計算
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")

        # 追加メトリクス計算
        mae_scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
        mse_scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error")

        cv_results = {
            "mean_score": cv_scores.mean(),
            "std_score": cv_scores.std(),
            "scores": cv_scores.tolist(),
            "folds": cv_folds,
            "cv_method": "TimeSeriesSplit",
            "mae_scores": (-mae_scores).tolist(),
            "rmse_scores": np.sqrt(-mse_scores).tolist(),
            "out_of_sample_months": 3,  # 約3ヶ月のOut-of-sample期間
        }

        self.logger.info(
            f"時系列交差検証結果: R² = {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}"
        )
        self.logger.info(
            f"Out-of-sample評価: MAE = {np.mean(cv_results['mae_scores']):.4f}, "
            f"RMSE = {np.mean(cv_results['rmse_scores']):.4f}"
        )

        return cv_results

    def evaluate_model(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """
        モデル性能を評価（複数指標）

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

            # 基本メトリクス計算
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            # Phase 2: 追加評価指標
            # MAPE (Mean Absolute Percentage Error)
            mape = self._calculate_mape(y_test, y_pred)

            # WAPE (Weighted Absolute Percentage Error)
            wape = self._calculate_wape(y_test, y_pred)

            # 予測精度の分布統計
            pred_mean = np.mean(y_pred)
            pred_std = np.std(y_pred)

            # 誤差分析
            errors = y_test - y_pred
            error_mean = np.mean(errors)
            error_std = np.std(errors)

            # 予測区間の精度（±1σ内の的中率）
            within_1sigma = np.mean(np.abs(errors) <= error_std)
            within_2sigma = np.mean(np.abs(errors) <= 2 * error_std)

            metrics = {
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "wape": wape,
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "error_mean": error_mean,
                "error_std": error_std,
                "within_1sigma": within_1sigma,
                "within_2sigma": within_2sigma,
                "n_samples": len(y_test),
            }

            self.logger.info(
                f"モデル評価: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}, "
                f"MAPE={mape:.2f}%, WAPE={wape:.2f}%"
            )

            return metrics

        except Exception as e:
            raise ModelBuildingError(f"モデル評価エラー: {e}")

    def _calculate_mape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        MAPE（平均絶対パーセント誤差）を計算

        Args:
            y_true: 実際の値
            y_pred: 予測値

        Returns:
            MAPE値（%）
        """
        try:
            # ゼロ除算を回避
            mask = y_true != 0
            if mask.sum() == 0:
                return float("inf")

            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            return float(mape)
        except:
            return float("inf")

    def _calculate_wape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        WAPE（重み付き絶対パーセント誤差）を計算

        Args:
            y_true: 実際の値
            y_pred: 予測値

        Returns:
            WAPE値（%）
        """
        try:
            total_true = y_true.sum()
            if total_true == 0:
                return float("inf")

            wape = (np.sum(np.abs(y_true - y_pred)) / total_true) * 100
            return float(wape)
        except:
            return float("inf")

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
        overfitting_threshold = self.config.get_quality_config().get("overfitting_threshold", 0.01)

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

    def predict(self, model: Any, X: pd.DataFrame, weights: Dict[str, float] = None) -> np.ndarray:
        """
        予測を実行

        Args:
            model: 訓練済みモデル（単一モデルまたはアンサンブル辞書）
            X: 特徴量DataFrame
            weights: アンサンブルの場合の重み辞書

        Returns:
            予測結果
        """
        try:
            if isinstance(model, dict):
                # アンサンブルモデルの場合
                return self.ensemble_predict(model, X, weights)
            else:
                # 単一モデルの場合
                X_processed = self._preprocess_features(X)
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
        model_type = results.get("model_type", "RandomForestRegressor")

        summary = {
            "model_type": model_type,
            "cv_mean_r2": results["cv_scores"]["mean_score"],
            "cv_std_r2": results["cv_scores"]["std_score"],
            "test_r2": results["test_metrics"]["r2_score"],
            "test_rmse": results["test_metrics"]["rmse"],
            "test_mae": results["test_metrics"]["mae"],
            "overfitting_score": results["overfitting_score"],
            "n_features": len(results["feature_names"]),
            "top_features": (
                list(results["feature_importance"].keys())[:5]
                if results["feature_importance"]
                else []
            ),
        }

        # アンサンブルモデルの場合は追加情報
        if model_type == "ensemble" and "ensemble_weights" in results:
            summary["ensemble_weights"] = results["ensemble_weights"]
            summary["ensemble_models"] = (
                list(results["model"].keys()) if isinstance(results["model"], dict) else []
            )

        # ベイズ最適化情報（あれば追加）
        if "optimization_results" in results:
            summary["optimization_performed"] = True
            summary["optimization_best_score"] = results["optimization_results"].get(
                "best_score", 0.0
            )
        else:
            summary["optimization_performed"] = False

        return summary

    def _build_random_forest(self, X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
        """
        RandomForestモデルを構築
        """
        n_estimators = self.model_config.get("n_estimators", 100)
        max_depth = self.model_config.get("max_depth", 10)
        random_state = self.model_config.get("random_state", 42)

        # Phase 3: L1/L2正則化相当のパラメータ調整
        min_samples_split = self.model_config.get("min_samples_split", 5)
        min_samples_leaf = self.model_config.get("min_samples_leaf", 2)
        max_features = self.model_config.get("max_features", "sqrt")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )

        model.fit(X, y)
        self.logger.info(
            f"RandomForest構築完了: n_estimators={n_estimators}, max_depth={max_depth}"
        )
        return model

    def _build_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        XGBoostモデルを構築
        """
        # XGBoostパラメータ（L1/L2正則化付き）
        params = {
            "n_estimators": self.model_config.get("xgb_n_estimators", 100),
            "max_depth": self.model_config.get("xgb_max_depth", 6),
            "learning_rate": self.model_config.get("xgb_learning_rate", 0.1),
            "reg_alpha": self.model_config.get("xgb_reg_alpha", 0.1),  # L1正則化
            "reg_lambda": self.model_config.get("xgb_reg_lambda", 1.0),  # L2正則化
            "subsample": self.model_config.get("xgb_subsample", 0.8),
            "colsample_bytree": self.model_config.get("xgb_colsample_bytree", 0.8),
            "random_state": self.model_config.get("random_state", 42),
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X, y)

        self.logger.info(
            f"XGBoost構築完了: n_estimators={params['n_estimators']}, L1={params['reg_alpha']}, L2={params['reg_lambda']}"
        )
        return model

    def _build_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        アンサンブルモデルを構築
        """
        self.logger.info("アンサンブルモデル構築開始")
        models = {}

        # RandomForest モデル
        rf_model = self._build_random_forest(X, y)
        models["random_forest"] = rf_model

        # XGBoost モデル（利用可能な場合）
        if XGBOOST_AVAILABLE:
            xgb_model = self._build_xgboost(X, y)
            models["xgboost"] = xgb_model
        else:
            self.logger.warning(
                "XGBoost が利用できないため、RandomForest のみでアンサンブルを構成"
            )

        self.logger.info(f"アンサンブル構築完了: {list(models.keys())}")
        return models

    def ensemble_predict(
        self, models: Dict[str, Any], X: pd.DataFrame, weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        アンサンブル予測を実行

        Args:
            models: 訓練済みモデル辞書
            X: 特徴量DataFrame
            weights: モデル重み辞書（デフォルトは均等重み）

        Returns:
            アンサンブル予測結果
        """
        X_processed = self._preprocess_features(X)
        predictions = {}

        # 各モデルで予測
        for name, model in models.items():
            predictions[name] = model.predict(X_processed)

        # 重み設定（デフォルトは均等重み）
        if weights is None:
            weights = {name: 1.0 / len(models) for name in models.keys()}

        # 重み付き平均でアンサンブル予測
        ensemble_pred = np.zeros(len(X_processed))
        for name, pred in predictions.items():
            ensemble_pred += weights.get(name, 0) * pred

        self.logger.info(f"アンサンブル予測完了: 重み={weights}")
        return ensemble_pred

    def optimize_ensemble_weights(
        self, models: Dict[str, Any], X_val: pd.DataFrame, y_val: pd.Series
    ) -> Dict[str, float]:
        """
        検証データを使用してアンサンブル重みを最適化

        Args:
            models: 訓練済みモデル辞書
            X_val: 検証特徴量
            y_val: 検証ターゲット

        Returns:
            最適化された重み辞書
        """
        from scipy.optimize import minimize

        X_val_processed = self._preprocess_features(X_val)

        # 各モデルの予測を取得
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X_val_processed)

        model_names = list(predictions.keys())
        pred_matrix = np.column_stack([predictions[name] for name in model_names])

        # 目的関数（RMSE最小化）
        def objective(weights):
            weights = weights / np.sum(weights)  # 正規化
            ensemble_pred = np.dot(pred_matrix, weights)
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            return rmse

        # 制約条件
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in model_names]

        # 初期値（均等重み）
        x0 = np.ones(len(model_names)) / len(model_names)

        # 最適化実行
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        if result.success:
            optimal_weights = dict(zip(model_names, result.x))
            self.logger.info(f"アンサンブル重み最適化完了: {optimal_weights}")
            return optimal_weights
        else:
            self.logger.warning("重み最適化に失敗、均等重みを使用")
            return {name: 1.0 / len(model_names) for name in model_names}

    def _evaluate_predictions(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        予測結果を評価（evaluate_modelの簡易版）

        Args:
            y_true: 実際の値
            y_pred: 予測値

        Returns:
            評価メトリクス辞書
        """
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = self._calculate_mape(y_true, y_pred)
        wape = self._calculate_wape(y_true, y_pred)

        return {
            "r2_score": r2,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "wape": wape,
            "n_samples": len(y_true),
        }

    def bayesian_hyperparameter_optimization(
        self, X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest", n_calls: int = 20
    ) -> Dict[str, Any]:
        """
        ベイズ最適化によるハイパーパラメータ調整

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries
            model_type: モデルタイプ ("random_forest", "xgboost", "ensemble")
            n_calls: 最適化の試行回数

        Returns:
            最適化結果辞書
        """
        if not BAYESIAN_OPT_AVAILABLE:
            self.logger.warning("ベイズ最適化が利用できません。デフォルトパラメータを使用します。")
            return {"best_params": {}, "best_score": 0.0, "optimization_history": []}

        self.logger.info(f"ベイズ最適化開始: {model_type}, 試行回数={n_calls}")

        X_processed = self._preprocess_features(X)

        if model_type == "random_forest":
            return self._optimize_random_forest(X_processed, y, n_calls)
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            return self._optimize_xgboost(X_processed, y, n_calls)
        elif model_type == "ensemble":
            return self._optimize_ensemble(X_processed, y, n_calls)
        else:
            self.logger.warning(f"サポートされていないモデルタイプ: {model_type}")
            return {"best_params": {}, "best_score": 0.0, "optimization_history": []}

    def _optimize_random_forest(
        self, X: pd.DataFrame, y: pd.Series, n_calls: int
    ) -> Dict[str, Any]:
        """
        RandomForestのベイズ最適化
        """
        # パラメータ空間の定義
        space = [
            Integer(50, 300, name="n_estimators"),
            Integer(5, 20, name="max_depth"),
            Integer(2, 10, name="min_samples_split"),
            Integer(1, 5, name="min_samples_leaf"),
            Real(0.1, 1.0, name="max_features_ratio"),
        ]

        @use_named_args(space)
        def objective(**params):
            max_features = int(params["max_features_ratio"] * X.shape[1])
            max_features = max(1, min(max_features, X.shape[1]))

            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=max_features,
                random_state=42,
                n_jobs=-1,
            )

            # TimeSeriesSplitで交差検証
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")

            # 負の値を返す（最小化問題として解く）
            return -scores.mean()

        # ベイズ最適化実行
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        # 最適パラメータ
        best_params = {
            "n_estimators": result.x[0],
            "max_depth": result.x[1],
            "min_samples_split": result.x[2],
            "min_samples_leaf": result.x[3],
            "max_features": int(result.x[4] * X.shape[1]),
        }

        self.logger.info(f"RandomForest最適化完了: R² = {-result.fun:.4f}")
        self.logger.info(f"最適パラメータ: {best_params}")

        return {
            "best_params": best_params,
            "best_score": -result.fun,
            "optimization_history": [-score for score in result.func_vals],
        }

    def _optimize_xgboost(self, X: pd.DataFrame, y: pd.Series, n_calls: int) -> Dict[str, Any]:
        """
        XGBoostのベイズ最適化
        """
        # パラメータ空間の定義
        space = [
            Integer(50, 300, name="n_estimators"),
            Integer(3, 10, name="max_depth"),
            Real(0.01, 0.3, name="learning_rate"),
            Real(0.0, 1.0, name="reg_alpha"),  # L1正則化
            Real(0.0, 2.0, name="reg_lambda"),  # L2正則化
            Real(0.6, 1.0, name="subsample"),
            Real(0.6, 1.0, name="colsample_bytree"),
        ]

        @use_named_args(space)
        def objective(**params):
            model = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=42,
                n_jobs=-1,
            )

            # TimeSeriesSplitで交差検証
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")

            return -scores.mean()

        # ベイズ最適化実行
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        # 最適パラメータ
        best_params = {
            "n_estimators": result.x[0],
            "max_depth": result.x[1],
            "learning_rate": result.x[2],
            "reg_alpha": result.x[3],
            "reg_lambda": result.x[4],
            "subsample": result.x[5],
            "colsample_bytree": result.x[6],
        }

        self.logger.info(f"XGBoost最適化完了: R² = {-result.fun:.4f}")
        self.logger.info(f"最適パラメータ: {best_params}")

        return {
            "best_params": best_params,
            "best_score": -result.fun,
            "optimization_history": [-score for score in result.func_vals],
        }

    def _optimize_ensemble(self, X: pd.DataFrame, y: pd.Series, n_calls: int) -> Dict[str, Any]:
        """
        アンサンブルモデルのベイズ最適化
        """
        # より少ない試行回数で両モデルを最適化
        rf_calls = n_calls // 2
        xgb_calls = n_calls - rf_calls

        # RandomForest最適化
        rf_result = self._optimize_random_forest(X, y, rf_calls)

        # XGBoost最適化（利用可能な場合）
        xgb_result = {"best_params": {}, "best_score": 0.0, "optimization_history": []}
        if XGBOOST_AVAILABLE:
            xgb_result = self._optimize_xgboost(X, y, xgb_calls)

        self.logger.info("アンサンブル最適化完了")

        return {
            "random_forest": rf_result,
            "xgboost": xgb_result,
            "best_combined_score": max(rf_result["best_score"], xgb_result["best_score"]),
        }

    def build_optimized_model(
        self, X: pd.DataFrame, y: pd.Series, optimization_results: Dict[str, Any], model_type: str
    ) -> Any:
        """
        最適化されたパラメータでモデルを構築

        Args:
            X: 特徴量DataFrame
            y: ターゲットSeries
            optimization_results: ベイズ最適化結果
            model_type: モデルタイプ

        Returns:
            最適化済みモデル
        """
        X_processed = self._preprocess_features(X)

        if model_type == "random_forest":
            best_params = optimization_results["best_params"]
            model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_processed, y)
            self.logger.info("最適化RandomForestモデル構築完了")
            return model

        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            best_params = optimization_results["best_params"]
            model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_processed, y)
            self.logger.info("最適化XGBoostモデル構築完了")
            return model

        elif model_type == "ensemble":
            models = {}

            # 最適化されたRandomForest
            rf_params = optimization_results["random_forest"]["best_params"]
            rf_model = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
            rf_model.fit(X_processed, y)
            models["random_forest"] = rf_model

            # 最適化されたXGBoost（利用可能な場合）
            if XGBOOST_AVAILABLE and optimization_results["xgboost"]["best_params"]:
                xgb_params = optimization_results["xgboost"]["best_params"]
                xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42, n_jobs=-1)
                xgb_model.fit(X_processed, y)
                models["xgboost"] = xgb_model

            self.logger.info("最適化アンサンブルモデル構築完了")
            return models

        else:
            raise ValueError(f"サポートされていないモデルタイプ: {model_type}")
