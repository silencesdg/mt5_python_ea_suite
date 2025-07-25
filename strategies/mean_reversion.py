import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, close_all, send_order
from logger import logger


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.mean_reversion_period = 20
        self.mean_reversion_std_dev = 2

    def _calculate_indicators(self, df):
        """
        计算均值回归指标
        """
        mean = df['close'].rolling(self.mean_reversion_period).mean()
        std = df['close'].rolling(self.mean_reversion_period).std()
        df['upper_band'] = mean + self.mean_reversion_std_dev * std
        df['lower_band'] = mean - self.mean_reversion_std_dev * std
        return df

    def generate_signal(self):
        """
        均值回归策略实盘：
        当价格超过20日均线正负2个标准差买卖。
        """
        rates = get_rates(self.symbol, self.timeframe, self.mean_reversion_period + 30)
        if rates is None or len(rates) < self.mean_reversion_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['close'].iloc[-2] > df['upper_band'].iloc[-2]:
            logger.info(f"价格超过上轨，产生卖出信号: {self.symbol}")
            return -1
        elif df['close'].iloc[-2] < df['lower_band'].iloc[-2]:
            logger.info(f"价格低于下轨，产生买入信号: {self.symbol}")
            return 1
        return 0

    def run_backtest(self, df):
        """
        均值回归回测：
        价格突破上下轨卖出/买入
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        for i in range(self.mean_reversion_period, len(df)):
            if df['close'].iloc[i-1] > df['upper_band'].iloc[i-1]:
                signals.iat[i] = -1
            elif df['close'].iloc[i-1] < df['lower_band'].iloc[i-1]:
                signals.iat[i] = 1
        return signals
