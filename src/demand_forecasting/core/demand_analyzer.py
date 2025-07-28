from typing import Dict, Any, Callable, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, curve_fit
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score

from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.exceptions import DemandAnalysisError


class DemandCurveAnalyzer:
    """需要曲線分析クラス"""
    
    def __init__(self, config: Config):
        """
        初期化
        
        Args:
            config: 設定管理オブジェクト
        """
        self.config = config
        self.logger = Logger(config.get_logging_config()).get_logger('demand_analyzer')
        
    def analyze_demand_curve(self, df: pd.DataFrame, product: str) -> Dict[str, Any]:
        """
        需要曲線を分析
        
        Args:
            df: 商品データを含むDataFrame
            product: 分析対象商品名
            
        Returns:
            需要曲線分析結果辞書
        """
        self.logger.info(f"需要曲線分析を開始: {product}")
        
        try:
            # 商品データを抽出
            product_data = self._extract_product_data(df, product)
            
            if len(product_data) < 5:
                raise DemandAnalysisError(f"データ不足: {product} のデータが{len(product_data)}件しかありません")
            
            # 価格-需要データを準備
            price_demand_data = self._prepare_price_demand_data(product_data)
            
            # 外れ値処理
            price_demand_data = self._remove_outliers_iqr(price_demand_data)
            
            # データ平滑化
            price_demand_data = self._apply_smoothing(price_demand_data)
            
            # 需要曲線をフィッティング
            demand_curve_function, fit_params, r2 = self._fit_demand_curve(price_demand_data)
            
            # 最適価格を計算
            optimal_price = self._calculate_optimal_price(demand_curve_function, price_demand_data)
            
            # 価格弾力性を計算
            price_elasticity = self._calculate_price_elasticity(
                demand_curve_function, optimal_price
            )
            
            # 現在価格を取得
            current_price = product_data['平均価格'].median()
            
            # 価格範囲を取得
            price_range = (
                price_demand_data['price'].min(),
                price_demand_data['price'].max()
            )
            
            # 数量範囲を取得
            quantity_range = (
                price_demand_data['quantity'].min(),
                price_demand_data['quantity'].max()
            )
            
            results = {
                'product_name': product,
                'optimal_price': optimal_price,
                'current_price': current_price,
                'price_elasticity': price_elasticity,
                'demand_curve_function': demand_curve_function,
                'fit_params': fit_params,
                'r2_score': r2,
                'price_range': price_range,
                'quantity_range': quantity_range,
                'data_points': len(price_demand_data),
                'price_demand_data': price_demand_data
            }
            
            self.logger.info(f"需要曲線分析完了: {product}")
            self.logger.info(f"最適価格: {optimal_price:.2f}, R²: {r2:.4f}")
            
            return results
            
        except Exception as e:
            raise DemandAnalysisError(f"需要曲線分析エラー ({product}): {e}")
    
    def _extract_product_data(self, df: pd.DataFrame, product: str) -> pd.DataFrame:
        """商品データを抽出"""
        # 商品名で完全一致検索
        product_data = df[df['商品名称'] == product].copy()
        
        if len(product_data) == 0:
            # 部分一致で再検索
            product_data = df[df['商品名称'].str.contains(product, na=False)].copy()
        
        if len(product_data) == 0:
            raise DemandAnalysisError(f"商品が見つかりません: {product}")
        
        return product_data
    
    def _prepare_price_demand_data(self, product_data: pd.DataFrame) -> pd.DataFrame:
        """価格-需要データを準備"""
        # 必要な列が存在するかチェック
        required_columns = ['平均価格', '数量']
        missing_columns = [col for col in required_columns if col not in product_data.columns]
        
        if missing_columns:
            raise DemandAnalysisError(f"必要な列が不足: {missing_columns}")
        
        # 価格と数量のデータを準備
        price_demand = product_data[['平均価格', '数量']].copy()
        price_demand.columns = ['price', 'quantity']
        
        # 無効な値を除去
        price_demand = price_demand.dropna()
        price_demand = price_demand[
            (price_demand['price'] > 0) & (price_demand['quantity'] > 0)
        ]
        
        # 価格でソート
        price_demand = price_demand.sort_values('price')
        
        return price_demand
    
    def _remove_outliers_iqr(self, data: pd.DataFrame) -> pd.DataFrame:
        """IQR法による外れ値除去"""
        data_clean = data.copy()
        
        for column in ['price', 'quantity']:
            Q1 = data_clean[column].quantile(0.25)
            Q3 = data_clean[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            before_count = len(data_clean)
            data_clean = data_clean[
                (data_clean[column] >= lower_bound) & 
                (data_clean[column] <= upper_bound)
            ]
            after_count = len(data_clean)
            
            if before_count != after_count:
                self.logger.info(f"{column}の外れ値を{before_count - after_count}件除去")
        
        return data_clean
    
    def _apply_smoothing(self, data: pd.DataFrame, method: str = 'savgol') -> pd.DataFrame:
        """データ平滑化を適用"""
        if len(data) < 5:
            self.logger.warning("データ数が少ないため平滑化をスキップ")
            return data
        
        data_smooth = data.copy()
        
        if method == 'savgol':
            # Savitzky-Golay フィルタを適用
            window_length = min(5, len(data) if len(data) % 2 == 1 else len(data) - 1)
            if window_length >= 3:
                data_smooth['quantity'] = savgol_filter(
                    data['quantity'], 
                    window_length=window_length, 
                    polyorder=min(2, window_length - 1)
                )
                self.logger.info("Savitzky-Golay平滑化を適用")
        
        elif method == 'rolling':
            # 移動平均を適用
            window_size = min(3, len(data))
            data_smooth['quantity'] = data['quantity'].rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
            self.logger.info("移動平均平滑化を適用")
        
        return data_smooth
    
    def _fit_demand_curve(self, data: pd.DataFrame) -> Tuple[Callable, Dict[str, float], float]:
        """需要曲線をフィッティング"""
        prices = data['price'].values
        quantities = data['quantity'].values
        
        # 複数の需要曲線モデルを試行
        models = {
            'linear': self._linear_demand,
            'exponential': self._exponential_demand,
            'power': self._power_demand
        }
        
        best_model = None
        best_params = None
        best_r2 = -np.inf
        best_function = None
        
        for model_name, model_func in models.items():
            try:
                params, r2, fitted_func = self._fit_model(prices, quantities, model_func)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = params
                    best_model = model_name
                    best_function = fitted_func
                    
                self.logger.debug(f"{model_name}モデル: R² = {r2:.4f}")
                
            except Exception as e:
                self.logger.warning(f"{model_name}モデルのフィッティング失敗: {e}")
                continue
        
        if best_function is None:
            # フォールバック: 線形近似
            self.logger.warning("すべてのモデルが失敗、線形近似を使用")
            slope, intercept = np.polyfit(prices, quantities, 1)
            best_function = lambda p: slope * p + intercept
            best_params = {'slope': slope, 'intercept': intercept}
            predictions = best_function(prices)
            best_r2 = r2_score(quantities, predictions)
            best_model = 'linear_fallback'
        
        self.logger.info(f"最適モデル: {best_model}, R² = {best_r2:.4f}")
        
        return best_function, {'model': best_model, **best_params}, best_r2
    
    def _linear_demand(self, price: np.ndarray, a: float, b: float) -> np.ndarray:
        """線形需要関数: Q = a - b*P"""
        return a - b * price
    
    def _exponential_demand(self, price: np.ndarray, a: float, b: float) -> np.ndarray:
        """指数需要関数: Q = a * exp(-b*P)"""
        return a * np.exp(-b * price)
    
    def _power_demand(self, price: np.ndarray, a: float, b: float) -> np.ndarray:
        """べき乗需要関数: Q = a * P^(-b)"""
        return a * np.power(price, -b)
    
    def _fit_model(self, prices: np.ndarray, quantities: np.ndarray, 
                   model_func: Callable) -> Tuple[Dict[str, float], float, Callable]:
        """モデルをフィッティング"""
        # パラメータの初期推定
        if model_func == self._linear_demand:
            initial_guess = [quantities.max(), (quantities.max() - quantities.min()) / (prices.max() - prices.min())]
        elif model_func == self._exponential_demand:
            initial_guess = [quantities.max(), 0.01]
        else:  # power_demand
            initial_guess = [quantities.mean() * prices.mean(), 1.0]
        
        # curve_fitでパラメータを最適化
        popt, _ = curve_fit(
            model_func, 
            prices, 
            quantities, 
            p0=initial_guess,
            maxfev=1000
        )
        
        # R²スコアを計算
        predictions = model_func(prices, *popt)
        r2 = r2_score(quantities, predictions)
        
        # フィッティングされた関数を作成
        fitted_function = lambda p: model_func(p, *popt)
        
        # パラメータ辞書を作成
        if model_func == self._linear_demand:
            params = {'a': popt[0], 'b': popt[1]}
        elif model_func == self._exponential_demand:
            params = {'a': popt[0], 'b': popt[1]}
        else:  # power_demand
            params = {'a': popt[0], 'b': popt[1]}
        
        return params, r2, fitted_function
    
    def _calculate_optimal_price(self, demand_func: Callable, data: pd.DataFrame) -> float:
        """最適価格を計算（収益最大化）"""
        try:
            # 価格範囲を設定
            price_min = data['price'].min()
            price_max = data['price'].max()
            
            # 収益関数（負の値で最小化問題に変換）
            def negative_revenue(price):
                quantity = demand_func(price)
                if quantity <= 0:
                    return np.inf
                return -(price * quantity)  # 負の収益
            
            # 最適化実行
            result = minimize_scalar(
                negative_revenue,
                bounds=(price_min, price_max),
                method='bounded'
            )
            
            if result.success:
                optimal_price = result.x
                self.logger.info(f"最適価格計算成功: {optimal_price:.2f}")
                return optimal_price
            else:
                # 最適化失敗時は中央値を返す
                self.logger.warning("最適価格計算失敗、中央値を使用")
                return data['price'].median()
                
        except Exception as e:
            self.logger.warning(f"最適価格計算エラー: {e}")
            return data['price'].median()
    
    def _calculate_price_elasticity(self, demand_func: Callable, price: float) -> float:
        """価格弾力性を計算"""
        try:
            # 価格弾力性 = (dQ/dP) * (P/Q)
            delta_p = 0.01  # 価格の微小変化
            
            q_current = demand_func(price)
            q_delta = demand_func(price + delta_p)
            
            if q_current <= 0 or q_delta <= 0:
                return -1.0  # デフォルト値
            
            dq_dp = (q_delta - q_current) / delta_p
            elasticity = dq_dp * (price / q_current)
            
            self.logger.debug(f"価格弾力性: {elasticity:.4f}")
            
            return elasticity
            
        except Exception as e:
            self.logger.warning(f"価格弾力性計算エラー: {e}")
            return -1.0  # デフォルト値
    
    def analyze_multiple_products(self, df: pd.DataFrame, 
                                  products: list = None) -> Dict[str, Dict[str, Any]]:
        """複数商品の需要曲線分析"""
        if products is None:
            products = df['商品名称'].unique()[:10]  # 上位10商品
        
        results = {}
        
        for product in products:
            try:
                result = self.analyze_demand_curve(df, product)
                results[product] = result
            except Exception as e:
                self.logger.error(f"商品 {product} の分析に失敗: {e}")
                continue
        
        self.logger.info(f"複数商品分析完了: {len(results)}/{len(products)} 件成功")
        
        return results
    
    def get_demand_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """需要分析サマリーを生成"""
        summary = {
            'product_name': results['product_name'],
            'current_price': results['current_price'],
            'optimal_price': results['optimal_price'],
            'price_change_pct': (
                (results['optimal_price'] - results['current_price']) / 
                results['current_price'] * 100
            ),
            'price_elasticity': results['price_elasticity'],
            'elasticity_category': self._categorize_elasticity(results['price_elasticity']),
            'model_fit_quality': results['r2_score'],
            'data_points': results['data_points']
        }
        
        return summary
    
    def _categorize_elasticity(self, elasticity: float) -> str:
        """価格弾力性をカテゴリ化"""
        abs_elasticity = abs(elasticity)
        
        if abs_elasticity > 1.5:
            return "高弾力性"
        elif abs_elasticity > 0.5:
            return "中弾力性"
        else:
            return "低弾力性"