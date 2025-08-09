import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """設定管理クラス"""

    def __init__(self, config_path: str = None):
        """
        設定ファイルを読み込んで初期化

        Args:
            config_path: 設定ファイルのパス（空や未設定の場合はデフォルト設定を使用）
        """
        # プロジェクトルートとデフォルト設定パス
        project_root = Path(__file__).resolve().parents[3]
        self.default_config_path = project_root / "config" / "config.yaml"

        # 優先設定パス
        self.config_path = Path(config_path) if config_path else self.default_config_path

        # 設定読み込み（デフォルト設定にユーザー設定を上書き）
        self.config = self._load_and_merge_configs()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """YAMLを読み込んで辞書を返す（空ファイルは空dict）"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data or {}
        except FileNotFoundError:
            # デフォルト設定が無ければエラー。ユーザー指定パスが無い場合もエラーにする
            raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
        except yaml.YAMLError as e:
            raise ValueError(f"設定ファイルの形式が正しくありません: {e}")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """辞書の再帰マージ（overrideがbaseを上書き）"""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _load_and_merge_configs(self) -> Dict[str, Any]:
        """デフォルト設定にユーザー設定をマージして返す"""
        default_cfg = self._load_yaml(self.default_config_path)
        user_cfg = self._load_yaml(self.config_path)
        return self._deep_merge(default_cfg, user_cfg)

    def get(self, key: str, default=None):
        """設定値を取得"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_data_config(self) -> Dict[str, Any]:
        """データ関連の設定を取得"""
        return self.get("data", {})

    def get_model_config(self) -> Dict[str, Any]:
        """モデル関連の設定を取得"""
        return self.get("model", {})

    def get_feature_config(self) -> Dict[str, Any]:
        """特徴量エンジニアリング関連の設定を取得"""
        return self.get("feature_engineering", {})

    def get_quality_config(self) -> Dict[str, Any]:
        """品質評価関連の設定を取得"""
        return self.get("quality", {})

    def get_visualization_config(self) -> Dict[str, Any]:
        """可視化関連の設定を取得"""
        return self.get("visualization", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """ログ関連の設定を取得"""
        return self.get("logging", {})
