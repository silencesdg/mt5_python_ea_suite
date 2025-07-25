
import pandas as pd
from logger import logger
from utils import get_rates

class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        # --- 策略核心参数 ---
        self.trend_period = 50
        self.retracement_tolerance = 0.30

        # --- 策略状态变量 ---
        self.current_trend = "none"  # none, uptrend, downtrend
        self.trend_peak = 0.0        # 上升趋势中的最高价
        self.trend_trough = float('inf') # 下降趋势中的最低价

    def generate_signal(self):
        """
        带状态维护的实盘信号生成方法。
        """
        # 获取足够的数据来计算滚动高低点
        rates = get_rates(self.symbol, 1, self.trend_period + 5)
        if rates is None or len(rates) < self.trend_period:
            return 0 # 数据不足，不产生信号
        
        df = pd.DataFrame(rates)
        
        # 获取当前价格和用于判断突破的历史高低点
        current_price = df['close'].iloc[-1]
        high_period = df['high'].rolling(self.trend_period).max().iloc[-2]
        low_period = df['low'].rolling(self.trend_period).min().iloc[-2]

        signal = 0

        # 状态 1: 当前无趋势，等待趋势开始
        if self.current_trend == "none":
            if current_price > high_period:
                self.current_trend = "uptrend"
                self.trend_peak = current_price
                signal = 1
                logger.info(f"实盘: 突破进入上升趋势，买入价: {current_price:.2f}")
            elif current_price < low_period:
                self.current_trend = "downtrend"
                self.trend_trough = current_price
                signal = -1
                logger.info(f"实盘: 跌破进入下降趋势，卖出价: {current_price:.2f}")
        
        # 状态 2: 当前处于上升趋势
        elif self.current_trend == "uptrend":
            if current_price < self.trend_peak * (1 - self.retracement_tolerance):
                logger.info(f"实盘: 上升趋势结束。最高点: {self.trend_peak:.2f}, 当前价: {current_price:.2f}。平仓卖出。")
                signal = -1
                self.current_trend = "none" # 重置状态
            else:
                self.trend_peak = max(self.trend_peak, current_price)

        # 状态 3: 当前处于下降趋势
        elif self.current_trend == "downtrend":
            if current_price > self.trend_trough * (1 + self.retracement_tolerance):
                logger.info(f"实盘: 下降趋势结束。最低点: {self.trend_trough:.2f}, 当前价: {current_price:.2f}。平仓买入。")
                signal = 1
                self.current_trend = "none" # 重置状态
            else:
                self.trend_trough = min(self.trend_trough, current_price)
        
        return signal

    def run_backtest(self, df):
        """
        带容错的趋势跟踪策略回测：
        - 突破N周期高点，进入上升趋势，回撤30%则趋势结束。
        - 跌破N周期低点，进入下降趋势，反弹30%则趋势结束。
        """
        df = df.copy()
        signals = pd.Series(0, index=df.index)

        df['high_period'] = df['high'].rolling(self.trend_period).max().shift(1)
        df['low_period'] = df['low'].rolling(self.trend_period).min().shift(1)

        # 回测时使用局部变量来管理状态，避免干扰实盘状态
        backtest_trend = "none"
        backtest_peak = 0.0
        backtest_trough = float('inf')

        for i in range(self.trend_period, len(df)):
            current_price = df['close'].iloc[i]
            
            if backtest_trend == "none":
                if current_price > df['high_period'].iloc[i]:
                    backtest_trend = "uptrend"
                    backtest_peak = current_price
                    signals.iat[i] = 1
                elif current_price < df['low_period'].iloc[i]:
                    backtest_trend = "downtrend"
                    backtest_trough = current_price
                    signals.iat[i] = -1
            
            elif backtest_trend == "uptrend":
                if current_price < backtest_peak * (1 - self.retracement_tolerance):
                    signals.iat[i] = -1
                    backtest_trend = "none"
                else:
                    backtest_peak = max(backtest_peak, current_price)

            elif backtest_trend == "downtrend":
                if current_price > backtest_trough * (1 + self.retracement_tolerance):
                    signals.iat[i] = 1
                    backtest_trend = "none"
                else:
                    backtest_trough = min(backtest_trough, current_price)
                    
        return signals
