import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, close_all, send_order
from logger import logger


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.fast_ma_period = 5
        self.slow_ma_period = 20

    def _calculate_indicators(self, df):
        """
        计算技术指标
        """
        df['fast_ma'] = df['close'].rolling(self.fast_ma_period).mean()
        df['slow_ma'] = df['close'].rolling(self.slow_ma_period).mean()
        return df

    def generate_signal(self):
        """
        均线交叉策略实盘：
        短期均线上穿长期均线买入，下穿卖出。
        """
        rates = get_rates(self.symbol, self.timeframe, self.slow_ma_period + 30)
        if rates is None or len(rates) < self.slow_ma_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['fast_ma'].iloc[-2] > df['slow_ma'].iloc[-2] and df['fast_ma'].iloc[-3] <= df['slow_ma'].iloc[-3]:
            logger.info(f"短期均线上穿长期均线，产生买入信号: {self.symbol}")
            return 1
        elif df['fast_ma'].iloc[-2] < df['slow_ma'].iloc[-2] and df['fast_ma'].iloc[-3] >= df['slow_ma'].iloc[-3]:
            logger.info(f"短期均线下穿长期均线，产生卖出信号: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        均线交叉回测：
        短期均线和长期均线交叉产生信号
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        # 从 slow_ma_period 开始循环，避免早期数据 NaN 问题
        for i in range(self.slow_ma_period, len(df)):
            if df['fast_ma'].iloc[i-1] > df['slow_ma'].iloc[i-1] and df['fast_ma'].iloc[i-2] <= df['slow_ma'].iloc[i-2]:
                signals.iat[i] = 1
            elif df['fast_ma'].iloc[i-1] < df['slow_ma'].iloc[i-1] and df['fast_ma'].iloc[i-2] >= df['slow_ma'].iloc[i-2]:
                signals.iat[i] = -1
        return signals
