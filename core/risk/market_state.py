import pandas as pd
import numpy as np
from logger import logger
from config import MARKET_STATE_CONFIG, SYMBOL, DEFAULT_WEIGHTS, TREND_INDICATOR_WEIGHTS, TREND_THRESHOLDS, MARKET_STATE_WEIGHTS, CONFIDENCE_THRESHOLDS

class MarketStateAnalyzer:
    """
    市场状态分析器 (支持参数优化)
    """
    
    def __init__(self, data_provider, market_state_params=None, trend_weights=None, trend_thresholds=None, confidence_thresholds=None):
        self.data_provider = data_provider
        self.symbol = SYMBOL
        self.timeframe = 16385 # TIMEFRAME_H1
        self.hourly_data_count = 100
        
        # 使用传入的参数或默认配置
        market_state_config = market_state_params or MARKET_STATE_CONFIG
        self.trend_period = market_state_config.get("trend_period", 50)
        self.retracement_tolerance = market_state_config.get("retracement_tolerance", 0.30)
        self.volume_period = market_state_config.get("volume_period", 20)
        self.volume_ma_period = market_state_config.get("volume_ma_period", 10)
        
        self.indicator_weights = trend_weights or TREND_INDICATOR_WEIGHTS
        self.thresholds = trend_thresholds or TREND_THRESHOLDS
        self.confidence_thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS
        
        self.current_trend = "none"
        self.trend_peak = 0.0
        self.trend_trough = float('inf')

    def _calculate_volume_indicators(self, df):
        df['volume'] = df['tick_volume']
        df['volume_ma'] = df['volume'].rolling(self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        volume_corr = df['close'].rolling(self.volume_period).corr(df['volume'])
        return df, volume_corr.iloc[-1] if len(volume_corr) > 0 and not pd.isna(volume_corr.iloc[-1]) else 0
    
    def _calculate_momentum_indicators(self, df):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        return df
    
    def _calculate_price_breakout_score(self, df):
        current_price = df['close'].iloc[-1]
        high_period = df['high'].rolling(self.trend_period).max().iloc[-2]
        low_period = df['low'].rolling(self.trend_period).min().iloc[-2]
        price_range = high_period - low_period
        if price_range == 0: return 0.5
        price_position = (current_price - low_period) / price_range
        if current_price > high_period: return 0.8
        elif current_price < low_period: return 0.2
        else: return np.clip(price_position, 0, 1)
    
    def _calculate_volume_score(self, df, volume_corr):
        current_volume = df['volume'].iloc[-1]
        volume_ma = df['volume_ma'].iloc[-1]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
        score = 0.5
        if volume_ratio > self.thresholds['volume_spike']: score += 0.3
        if volume_corr > 0.5: score += 0.2
        elif volume_corr < -0.5: score -= 0.2
        return np.clip(score, 0, 1)
    
    def _calculate_momentum_score(self, df):
        current_rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
        current_macd_hist = df['macd_histogram'].iloc[-1] if not pd.isna(df['macd_histogram'].iloc[-1]) else 0
        score = 0.5
        if current_rsi > self.thresholds['overbought']: score += 0.2
        elif current_rsi < self.thresholds['oversold']: score -= 0.2
        if current_macd_hist > 0: score += 0.2
        else: score -= 0.2
        return np.clip(score, 0, 1)
    
    def _calculate_ma_score(self, df):
        current_price = df['close'].iloc[-1]
        ma20 = df['ma20'].iloc[-1] if not pd.isna(df['ma20'].iloc[-1]) else current_price
        ma50 = df['ma50'].iloc[-1] if not pd.isna(df['ma50'].iloc[-1]) else current_price
        score = 0.5
        if current_price > ma20 > ma50: score += 0.3
        elif current_price < ma20 < ma50: score -= 0.3
        if len(df) >= 3:
            ma20_slope = df['ma20'].iloc[-1] - df['ma20'].iloc[-3]
            ma50_slope = df['ma50'].iloc[-1] - df['ma50'].iloc[-3]
            if ma20_slope > 0 and ma50_slope > 0: score += 0.2
            elif ma20_slope < 0 and ma50_slope < 0: score -= 0.2
        return np.clip(score, 0, 1)

    def _update_trend_state(self, df, new_state):
        current_price = df['close'].iloc[-1]
        if new_state == "uptrend":
            self.current_trend = "uptrend"
            self.trend_peak = max(self.trend_peak, current_price)
        elif new_state == "downtrend":
            self.current_trend = "downtrend"
            self.trend_trough = min(self.trend_trough, current_price)
        else:
            if self.current_trend == "uptrend" and current_price < self.trend_peak * (1 - self.retracement_tolerance):
                self.current_trend = "none"
            elif self.current_trend == "downtrend" and current_price > self.trend_trough * (1 + self.retracement_tolerance):
                self.current_trend = "none"

    def get_market_state(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.hourly_data_count)
        if rates is None or len(rates) < max(self.trend_period, 50):
            return "none", 0.0
            
        df = pd.DataFrame(rates)
        df, volume_corr = self._calculate_volume_indicators(df)
        df = self._calculate_momentum_indicators(df)
        
        price_score = self._calculate_price_breakout_score(df)
        volume_score = self._calculate_volume_score(df, volume_corr)
        momentum_score = self._calculate_momentum_score(df)
        ma_score = self._calculate_ma_score(df)
        
        trend_strength = (price_score * self.indicator_weights['price_breakout'] + 
                          volume_score * self.indicator_weights['volume_confirmation'] + 
                          momentum_score * self.indicator_weights['momentum oscillator'] + 
                          ma_score * self.indicator_weights['moving_average'])
        
        if trend_strength >= self.thresholds['strong_trend']: state, confidence = "uptrend", min(0.95, trend_strength)
        elif trend_strength <= (1 - self.thresholds['strong_trend']): state, confidence = "downtrend", min(0.95, 1 - trend_strength)
        elif trend_strength >= self.thresholds['weak_trend']: state, confidence = "ranging", 0.6
        else: state, confidence = "none", 0.4
        
        self._update_trend_state(df, state)
        return state, confidence

    def get_strategy_weights(self, market_state, confidence):
        base_weights = MARKET_STATE_WEIGHTS.get(market_state, DEFAULT_WEIGHTS)
        
        high_confidence_threshold = self.confidence_thresholds.get("high_confidence", 0.7)
        medium_confidence_threshold = self.confidence_thresholds.get("medium_confidence", 0.4)
        
        if confidence > high_confidence_threshold:
            return {k: v * confidence for k, v in base_weights.items()}
        elif confidence > medium_confidence_threshold:
            return {k: (v * confidence + DEFAULT_WEIGHTS.get(k, 1.0) * (1 - confidence)) for k, v in base_weights.items()}
        else:
            return DEFAULT_WEIGHTS
