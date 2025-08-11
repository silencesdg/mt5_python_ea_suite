import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from utils import get_rates
from logger import logger
from config import WAVE_THEORY_CONFIG


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1  # 使用1分钟数据，与其他策略保持一致
        
        # 从配置文件加载参数
        config = WAVE_THEORY_CONFIG
        self.daily_data_count = config.get("daily_data_count", 30)
        self.ema_short = config.get("ema_short", 5)
        self.ema_medium = config.get("ema_medium", 13)
        self.ema_long = config.get("ema_long", 34)
        self.wave_period = config.get("wave_period", 21)
        self.range_period = config.get("range_period", 20)
        self.adx_period = config.get("adx_period", 14)
        
        # 波浪理论参数
        self.retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.momentum_period = 14
        
        # 震荡市检测参数（调整为适合1分钟数据）
        self.range_threshold = 0.005  # 0.5%的价格波动范围，适合1分钟数据
        self.adx_threshold = 25  # ADX小于25表示震荡市，适合1分钟数据

    def _calculate_indicators(self, df):
        """
        计算波浪理论相关指标
        """
        # 计算EMA
        df['ema_short'] = df['close'].ewm(span=self.ema_short).mean()
        df['ema_medium'] = df['close'].ewm(span=self.ema_medium).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long).mean()
        
        # 计算动量指标
        df['momentum'] = df['close'].diff(self.momentum_period) / df['close'].shift(self.momentum_period) * 100
        
        # 计算波动范围
        df['high_max'] = df['high'].rolling(self.range_period).max()
        df['low_min'] = df['low'].rolling(self.range_period).min()
        df['range_pct'] = (df['high_max'] - df['low_min']) / df['close'] * 100
        
        # 计算ADX（平均趋向指数）用于判断趋势强度
        df['adx'] = self._calculate_adx(df)
        
        # 识别潜在的波浪点
        df['potential_wave_points'] = self._identify_wave_points(df)
        
        # 计算斐波那契回撤位
        df = self._calculate_fibonacci_levels(df)
        
        return df
    
    def _calculate_adx(self, df):
        """
        计算ADX指标
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # 计算真实波幅
        df['tr'] = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        # 计算方向移动
        df['up_move'] = high - high.shift(1)
        df['down_move'] = low.shift(1) - low
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # 计算平滑值
        df['plus_di'] = 100 * (df['plus_dm'].ewm(span=self.adx_period).mean() / df['tr'].ewm(span=self.adx_period).mean())
        df['minus_di'] = 100 * (df['minus_dm'].ewm(span=self.adx_period).mean() / df['tr'].ewm(span=self.adx_period).mean())
        
        # 计算DX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        
        # 计算ADX
        adx = df['dx'].ewm(span=self.adx_period).mean()
        
        return adx
    
    def _identify_wave_points(self, df):
        """
        识别潜在的波浪转折点
        """
        wave_points = pd.Series(0, index=df.index)
        
        for i in range(self.wave_period, len(df) - self.wave_period):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # 检查是否为波峰
            is_peak = (current_high == df['high'].iloc[i-self.wave_period:i+self.wave_period].max())
            
            # 检查是否为波谷
            is_trough = (current_low == df['low'].iloc[i-self.wave_period:i+self.wave_period].min())
            
            if is_peak:
                wave_points.iloc[i] = 1  # 波峰
            elif is_trough:
                wave_points.iloc[i] = -1  # 波谷
        
        return wave_points
    
    def _calculate_fibonacci_levels(self, df):
        """
        计算斐波那契回撤位
        """
        # 为每个斐波那契级别创建单独的列
        for level in self.retracement_levels:
            df[f'fib_{level}'] = None
        
        for i in range(self.wave_period * 2, len(df)):
            # 寻找最近的波峰和波谷
            wave_points = df['potential_wave_points'].iloc[:i+1]
            peaks = wave_points[wave_points == 1]
            troughs = wave_points[wave_points == -1]
            
            if len(peaks) > 0 and len(troughs) > 0:
                last_peak_idx = peaks.index[-1]
                last_trough_idx = troughs.index[-1]
                
                if last_peak_idx > last_trough_idx:
                    # 下降趋势，计算回撤位
                    high_price = df['high'].iloc[last_peak_idx]
                    low_price = df['low'].iloc[last_trough_idx]
                    price_range = high_price - low_price
                    
                    for level in self.retracement_levels:
                        df.at[i, f'fib_{level}'] = high_price - price_range * level
                else:
                    # 上升趋势，计算回撤位
                    low_price = df['low'].iloc[last_trough_idx]
                    high_price = df['high'].iloc[last_peak_idx]
                    price_range = high_price - low_price
                    
                    for level in self.retracement_levels:
                        df.at[i, f'fib_{level}'] = low_price + price_range * level
        
        return df
    
    def _is_sideways_market(self, df):
        """
        判断是否为震荡市
        """
        if len(df) < self.range_period:
            return False
        
        # 使用ADX判断趋势强度
        adx_value = df['adx'].iloc[-1]
        is_low_adx = adx_value < self.adx_threshold
        
        # 使用价格范围判断
        range_pct = df['range_pct'].iloc[-1]
        is_tight_range = range_pct < self.range_threshold * 100
        
        # 结合两个条件
        return is_low_adx and is_tight_range
    
    def generate_signal(self):
        """
        波浪理论策略实盘信号生成
        """
        rates = get_rates(self.symbol, self.timeframe, self.daily_data_count)
        if rates is None or len(rates) < self.wave_period * 3:
            return 0
        
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)
        
        # 判断市场状态
        is_sideways = self._is_sideways_market(df)
        
        # 获取最近的波浪点
        recent_wave_points = df['potential_wave_points'].iloc[-self.wave_period:]
        current_momentum = df['momentum'].iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # 震荡市中的波浪理论信号
        if is_sideways:
            # 在震荡市中，寻找区间边界的反转机会
            upper_bound = df['high_max'].iloc[-1]
            lower_bound = df['low_min'].iloc[-1]
            
            # 价格接近上边界且有转弱迹象
            if current_price > upper_bound * 0.98 and current_momentum < 0:
                logger.info(f"震荡市中价格接近上边界，产生卖出信号: {self.symbol}")
                return -1
            
            # 价格接近下边界且有转强迹象
            elif current_price < lower_bound * 1.02 and current_momentum > 0:
                logger.info(f"震荡市中价格接近下边界，产生买入信号: {self.symbol}")
                return 1
        
        # 趋势市场中的波浪理论信号
        else:
            # 寻找波浪模式的确认信号
            ema_alignment = (df['ema_short'].iloc[-1] > df['ema_medium'].iloc[-1] > df['ema_long'].iloc[-1])
            
            # 上升趋势中的回调买入
            if ema_alignment and current_momentum > 0:
                # 检查是否在斐波那契回撤位附近
                if f'fib_0.618' in df.columns and not pd.isna(df[f'fib_0.618'].iloc[-1]):
                    fib_618 = df[f'fib_0.618'].iloc[-1]
                    if abs(current_price - fib_618) / fib_618 < 0.01:  # 1%误差范围内
                        logger.info(f"上升趋势中回调至斐波那契61.8%位，产生买入信号: {self.symbol}")
                        return 1
            
            # 下降趋势中的反弹卖出
            elif not ema_alignment and current_momentum < 0:
                # 检查是否在斐波那契回撤位附近
                if f'fib_0.618' in df.columns and not pd.isna(df[f'fib_0.618'].iloc[-1]):
                    fib_618 = df[f'fib_0.618'].iloc[-1]
                    if abs(current_price - fib_618) / fib_618 < 0.01:  # 1%误差范围内
                        logger.info(f"下降趋势中反弹至斐波那契61.8%位，产生卖出信号: {self.symbol}")
                        return -1
        
        return 0
    
    def run_backtest(self, df):
        """
        波浪理论策略回测
        """
        df = df.copy()
        df = self._calculate_indicators(df)
        
        signals = pd.Series(0, index=df.index)
        
        for i in range(self.wave_period * 3, len(df)):
            current_df = df.iloc[:i+1]
            
            # 判断市场状态
            is_sideways = self._is_sideways_market(current_df)
            
            # 获取当前时刻的数据
            current_price = current_df['close'].iloc[-1]
            current_momentum = current_df['momentum'].iloc[-1]
            
            if is_sideways:
                # 震荡市信号
                upper_bound = current_df['high_max'].iloc[-1]
                lower_bound = current_df['low_min'].iloc[-1]
                
                if current_price > upper_bound * 0.98 and current_momentum < 0:
                    signals.iat[i] = -1
                elif current_price < lower_bound * 1.02 and current_momentum > 0:
                    signals.iat[i] = 1
            
            else:
                # 趋势市信号
                ema_alignment = (current_df['ema_short'].iloc[-1] > current_df['ema_medium'].iloc[-1] > current_df['ema_long'].iloc[-1])
                
                if ema_alignment and current_momentum > 0:
                    # 检查斐波那契回撤位
                    if f'fib_0.618' in current_df.columns and not pd.isna(current_df[f'fib_0.618'].iloc[-1]):
                        fib_618 = current_df[f'fib_0.618'].iloc[-1]
                        if abs(current_price - fib_618) / fib_618 < 0.01:
                            signals.iat[i] = 1
                
                elif not ema_alignment and current_momentum < 0:
                    # 检查斐波那契回撤位
                    if f'fib_0.618' in current_df.columns and not pd.isna(current_df[f'fib_0.618'].iloc[-1]):
                        fib_618 = current_df[f'fib_0.618'].iloc[-1]
                        if abs(current_price - fib_618) / fib_618 < 0.01:
                            signals.iat[i] = -1
        
        return signals
    
    def get_market_state(self, df):
        """
        获取当前市场状态信息
        """
        if len(df) < self.wave_period * 3:
            return {"state": "insufficient_data", "confidence": 0}
        
        is_sideways = self._is_sideways_market(df)
        adx_value = df['adx'].iloc[-1]
        range_pct = df['range_pct'].iloc[-1]
        
        return {
            "state": "sideways" if is_sideways else "trending",
            "confidence": max(0, min(1, (30 - adx_value) / 30)),  # ADX越低，震荡市置信度越高
            "adx": adx_value,
            "range_pct": range_pct
        }