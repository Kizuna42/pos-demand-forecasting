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
            config_path: 設定ファイルのパス
        """
        if config_path is None:
            # プロジェクトルートから設定ファイルを検索
            project_root = Path(__file__).resolve().parents[3]
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"設定ファイルの形式が正しくありません: {e}")

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
