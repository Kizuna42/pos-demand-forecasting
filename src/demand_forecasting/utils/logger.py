import logging
import os
from pathlib import Path
from typing import Optional


class Logger:
    """ログ管理クラス"""

    _instance = None
    _logger = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: dict = None):
        """
        ログ設定を初期化

        Args:
            config: ログ設定辞書
        """
        if self._logger is not None:
            return

        if config is None:
            config = {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/demand_forecasting.log",
            }

        self._setup_logger(config)

    def _setup_logger(self, config: dict):
        """ログの設定"""
        # ログレベルの設定
        level = getattr(logging, config.get("level", "INFO").upper())

        # フォーマッターの設定
        formatter = logging.Formatter(config.get("format"))

        # ロガーの作成
        self._logger = logging.getLogger("demand_forecasting")
        self._logger.setLevel(level)

        # 既存のハンドラーをクリア
        self._logger.handlers = []

        # コンソールハンドラーの追加
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # ファイルハンドラーの追加
        if "file" in config:
            log_file = Path(config["file"])
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """ロガーを取得"""
        if name:
            return logging.getLogger(f"demand_forecasting.{name}")
        return self._logger

    def debug(self, message: str):
        """デバッグログ"""
        self._logger.debug(message)

    def info(self, message: str):
        """情報ログ"""
        self._logger.info(message)

    def warning(self, message: str):
        """警告ログ"""
        self._logger.warning(message)

    def error(self, message: str):
        """エラーログ"""
        self._logger.error(message)

    def critical(self, message: str):
        """重要エラーログ"""
        self._logger.critical(message)
