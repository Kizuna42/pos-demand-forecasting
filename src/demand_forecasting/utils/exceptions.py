class DemandForecastingError(Exception):
    """需要予測システムのベース例外クラス"""

    pass


class DataProcessingError(DemandForecastingError):
    """データ処理関連のエラー"""

    pass


class FeatureEngineeringError(DemandForecastingError):
    """特徴量エンジニアリング関連のエラー"""

    pass


class ModelBuildingError(DemandForecastingError):
    """モデル構築関連のエラー"""

    pass


class DemandAnalysisError(DemandForecastingError):
    """需要曲線分析関連のエラー"""

    pass


class QualityEvaluationError(DemandForecastingError):
    """品質評価関連のエラー"""

    pass


class VisualizationError(DemandForecastingError):
    """可視化関連のエラー"""

    pass


class ReportGenerationError(DemandForecastingError):
    """レポート生成関連のエラー"""

    pass


class ConfigurationError(DemandForecastingError):
    """設定関連のエラー"""

    pass


class ExternalAPIError(DemandForecastingError):
    """外部API関連のエラー"""

    pass
