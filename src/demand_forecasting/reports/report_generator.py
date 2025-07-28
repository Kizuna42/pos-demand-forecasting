from typing import Dict, Any
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.exceptions import ReportGenerationError


class ReportGenerator:
    """レポート生成クラス（スタブ実装）"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(config.get_logging_config()).get_logger('report_generator')
    
    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Markdownレポートを生成（スタブ）"""
        self.logger.info("Markdownレポート生成（未実装）")
        return ""