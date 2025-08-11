import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from logger import logger
from utils import get_rates
from config import MARKET_STATE_CONFIG, SYMBOL, DEFAULT_WEIGHTS, TREND_INDICATOR_WEIGHTS, TREND_THRESHOLDS

class MarketStateAnalyzer:
    """
    市场状态分析器，基于resilient_trend逻辑
    判断当前市场状态：uptrend, downtrend, none
    """
    
    def __init__(self):
        self.symbol = SYMBOL
        self.timeframe = mt5.TIMEFRAME_H1  # 使用小时线数据
        self.hourly_data_count = 100  # 获取最近100小时的数据
        self.trend_period = MARKET_STATE_CONFIG.get("trend_period", 50)
        self.retracement_tolerance = MARKET_STATE_CONFIG.get("retracement_tolerance", 0.30)
        self.volume_period = MARKET_STATE_CONFIG.get("volume_period", 20)
        self.volume_ma_period = MARKET_STATE_CONFIG.get("volume_ma_period", 10)
        
        # 市场状态变量
        self.current_trend = "none"  # none, uptrend, downtrend
        self.trend_peak = 0.0
        self.trend_trough = float('inf')
        
        # 趋势指标权重
        self.indicator_weights = TREND_INDICATOR_WEIGHTS
        self.thresholds = TREND_THRESHOLDS
        
    def calculate_volume_indicators(self, df):
        """计算成交量指标"""
        # MT5使用tick_volume和real_volume，这里使用tick_volume作为成交量
        df['volume'] = df['tick_volume']  # 将tick_volume映射为volume以便后续计算
        
        # 成交量移动平均
        df['volume_ma'] = df['volume'].rolling(self.volume_ma_period).mean()
        
        # 成交量相对强度
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 价量相关性 (最近n期)
        volume_corr = df['close'].rolling(self.volume_period).corr(df['volume'])
        
        return df, volume_corr.iloc[-1] if len(volume_corr) > 0 else 0
    
    def calculate_momentum_indicators(self, df):
        """计算动量指标"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 移动平均线
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        
        return df
    
    def calculate_price_breakout_score(self, df):
        """计算价格突破得分 (0-1)"""
        current_price = df['close'].iloc[-1]
        high_period = df['high'].rolling(self.trend_period).max().iloc[-2]
        low_period = df['low'].rolling(self.trend_period).min().iloc[-2]
        
        # 计算价格在区间中的位置
        price_position = (current_price - low_period) / (high_period - low_period)
        
        # 突破确认
        if current_price > high_period:
            return 0.8  # 向上突破
        elif current_price < low_period:
            return 0.2  # 向下突破
        else:
            return price_position  # 在区间内的位置
    
    def calculate_volume_score(self, df, volume_corr):
        """计算成交量确认得分 (0-1)"""
        current_volume = df['volume'].iloc[-1]
        volume_ma = df['volume_ma'].iloc[-1]
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1
        
        score = 0.5  # 基础分数
        
        # 成交量突增确认
        if volume_ratio > self.thresholds['volume_spike']:
            score += 0.3
        
        # 价量相关性
        if volume_corr > 0.5:  # 正相关，价量齐升
            score += 0.2
        elif volume_corr < -0.5:  # 负相关，价跌量增
            score -= 0.2
        
        return np.clip(score, 0, 1)
    
    def calculate_momentum_score(self, df):
        """计算动量指标得分 (0-1)"""
        current_rsi = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
        current_macd_hist = df['macd_histogram'].iloc[-1] if not pd.isna(df['macd_histogram'].iloc[-1]) else 0
        
        score = 0.5  # 基础分数
        
        # RSI判断
        if current_rsi > self.thresholds['overbought']:
            score += 0.2  # 超买，可能继续上涨
        elif current_rsi < self.thresholds['oversold']:
            score -= 0.2  # 超卖，可能继续下跌
        
        # MACD直方图判断
        if current_macd_hist > 0:
            score += 0.2  # MACD正向动能
        else:
            score -= 0.2  # MACD负向动能
        
        return np.clip(score, 0, 1)
    
    def calculate_ma_score(self, df):
        """计算移动平均线得分 (0-1)"""
        current_price = df['close'].iloc[-1]
        ma20 = df['ma20'].iloc[-1] if not pd.isna(df['ma20'].iloc[-1]) else current_price
        ma50 = df['ma50'].iloc[-1] if not pd.isna(df['ma50'].iloc[-1]) else current_price
        
        score = 0.5  # 基础分数
        
        # 价格与均线关系
        if current_price > ma20 > ma50:
            score += 0.3  # 多头排列
        elif current_price < ma20 < ma50:
            score -= 0.3  # 空头排列
        
        # 均线方向
        if len(df) >= 3:
            ma20_slope = df['ma20'].iloc[-1] - df['ma20'].iloc[-3]
            ma50_slope = df['ma50'].iloc[-1] - df['ma50'].iloc[-3]
            
            if ma20_slope > 0 and ma50_slope > 0:
                score += 0.2  # 均线向上
            elif ma20_slope < 0 and ma50_slope < 0:
                score -= 0.2  # 均线向下
        
        return np.clip(score, 0, 1)
    
    def get_market_state(self):
        """
        获取当前市场状态 - 使用加权多指标分析
        返回: (state, confidence) 
        state: "uptrend", "downtrend", "none", "ranging"
        confidence: 0.0-1.0 的置信度
        """
        logger.info(f"开始获取市场状态数据: {self.symbol}, 时间周期: {self.timeframe}, 数据量: {self.hourly_data_count}")
        
        rates = get_rates(self.symbol, self.timeframe, self.hourly_data_count)
        
        # 检查数据获取状态
        if rates is None:
            logger.error(f"获取市场数据失败: {self.symbol}")
            return "none", 0.0
            
        logger.info(f"成功获取数据，数据量: {len(rates)}, 需要最小数据量: {max(self.trend_period, 50)}")
        
        if len(rates) < max(self.trend_period, 50):
            logger.warning(f"数据量不足: 当前{len(rates)}, 需要{max(self.trend_period, 50)}")
            return "none", 0.0
            
        df = pd.DataFrame(rates)
        
        # 计算各项指标
        df, volume_corr = self.calculate_volume_indicators(df)
        df = self.calculate_momentum_indicators(df)
        
        # 计算各项得分
        price_score = self.calculate_price_breakout_score(df)
        volume_score = self.calculate_volume_score(df, volume_corr)
        momentum_score = self.calculate_momentum_score(df)
        ma_score = self.calculate_ma_score(df)
        
        logger.info(f"各项指标得分: 价格突破={price_score:.3f}, 成交量确认={volume_score:.3f}, "
                   f"动量指标={momentum_score:.3f}, 移动平均={ma_score:.3f}")
        
        # 加权计算趋势强度
        trend_strength = (
            price_score * self.indicator_weights['price_breakout'] +
            volume_score * self.indicator_weights['volume_confirmation'] +
            momentum_score * self.indicator_weights['momentum oscillator'] +
            ma_score * self.indicator_weights['moving_average']
        )
        
        logger.info(f"趋势强度计算结果: {trend_strength:.3f}")
        logger.info(f"阈值参考: 强趋势={self.thresholds['strong_trend']}, 弱趋势={self.thresholds['weak_trend']}")
        
        # 判断市场状态
        if trend_strength >= self.thresholds['strong_trend']:
            state = "uptrend"
            confidence = min(0.9, trend_strength)
            logger.info(f"判断为上涨趋势: 趋势强度{trend_strength:.3f} >= 强趋势阈值{self.thresholds['strong_trend']}")
        elif trend_strength <= (1 - self.thresholds['strong_trend']):
            state = "downtrend"
            confidence = min(0.9, 1 - trend_strength)
            logger.info(f"判断为下跌趋势: 趋势强度{trend_strength:.3f} <= {1 - self.thresholds['strong_trend']:.3f}")
        elif trend_strength >= self.thresholds['weak_trend']:
            state = "ranging"
            confidence = 0.6
            logger.info(f"判断为震荡市场: 趋势强度{trend_strength:.3f} 在弱趋势阈值{self.thresholds['weak_trend']}和强趋势阈值{self.thresholds['strong_trend']}之间")
        else:
            state = "none"
            confidence = 0.4
            logger.info(f"判断为无明确趋势: 趋势强度{trend_strength:.3f} 低于弱趋势阈值{self.thresholds['weak_trend']}")
        
        # 更新趋势状态（用于状态机逻辑）
        self.update_trend_state(df, trend_strength, state)
        
        logger.debug(f"市场状态分析: {state}, 置信度: {confidence:.2f}, "
                    f"趋势强度: {trend_strength:.2f}, "
                    f"价格得分: {price_score:.2f}, 成交量得分: {volume_score:.2f}, "
                    f"动量得分: {momentum_score:.2f}, 均线得分: {ma_score:.2f}")
        
        return state, confidence
    
    def update_trend_state(self, df, trend_strength, new_state):
        """更新趋势状态（保持原有状态机逻辑的兼容性）"""
        current_price = df['close'].iloc[-1]
        
        if new_state == "uptrend":
            self.current_trend = "uptrend"
            self.trend_peak = max(self.trend_peak, current_price)
        elif new_state == "downtrend":
            self.current_trend = "downtrend"
            self.trend_trough = min(self.trend_trough, current_price)
        else:
            # 检查是否需要反转趋势
            if self.current_trend == "uptrend":
                if current_price < self.trend_peak * (1 - self.retracement_tolerance):
                    self.current_trend = "none"
            elif self.current_trend == "downtrend":
                if current_price > self.trend_trough * (1 + self.retracement_tolerance):
                    self.current_trend = "none"
    
    def get_strategy_weights(self, market_state, confidence):
        """
        根据市场状态返回推荐的策略权重配置
        """
        weight_configs = {
            "uptrend": {
                "ma_cross": 2.5,
                "momentum_breakout": 2.2,
                "turtle": 2.0,
                "macd": 1.8,
                "rsi": 1.2,
                "bollinger": 1.0,
                "kdj": 0.8,
                "mean_reversion": 0.5,
                "daily_breakout": 1.5,
                "wave_theory": 1.8  # 趋势市中波浪理论权重适中
            },
            "downtrend": {
                "ma_cross": 2.5,
                "turtle": 2.2,
                "momentum_breakout": 2.0,
                "macd": 1.8,
                "rsi": 1.5,
                "bollinger": 1.2,
                "kdj": 1.0,
                "mean_reversion": 0.8,
                "daily_breakout": 1.0,
                "wave_theory": 1.8  # 趋势市中波浪理论权重适中
            },
            "ranging": {
                "rsi": 2.5,
                "bollinger": 2.2,
                "mean_reversion": 2.0,
                "kdj": 1.8,
                "wave_theory": 3.0,  # 震荡市中波浪理论权重最高
                "ma_cross": 1.2,
                "macd": 1.0,
                "turtle": 0.8,
                "momentum_breakout": 0.5,
                "daily_breakout": 1.5
            },
            "none": DEFAULT_WEIGHTS
        }
        
        base_weights = weight_configs.get(market_state, weight_configs["none"])
        
        # 根据置信度调整权重
        if confidence > 0.7:
            # 高置信度，使用推荐权重
            return base_weights
        elif confidence > 0.4:
            # 中等置信度，混合默认权重
            default_weights = weight_configs["none"]
            return {k: (base_weights[k] * 0.7 + default_weights[k] * 0.3) 
                   for k in base_weights}
        else:
            # 低置信度，使用默认权重
            return weight_configs["none"]
    
    def reset_state(self):
        """重置市场状态"""
        self.current_trend = "none"
        self.trend_peak = 0.0
        self.trend_trough = float('inf')