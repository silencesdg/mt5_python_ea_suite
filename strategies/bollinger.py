import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, close_all, send_order
from logger import logger

class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.bollinger_period = 20
        self.bollinger_std_dev = 2

    def _calculate_indicators(self, df):
        """
        计算布林带指标
        """
        mean = df['close'].rolling(self.bollinger_period).mean()
        std = df['close'].rolling(self.bollinger_period).std()
        df['upper_band'] = mean + self.bollinger_std_dev * std
        df['lower_band'] = mean - self.bollinger_std_dev * std
        return df

    def generate_signal(self):
        """
        布林带策略实盘：
        当价格跌破下轨买入，涨破上轨卖出。
        """
        rates = get_rates(self.symbol, self.timeframe, self.bollinger_period + 30)
        if rates is None or len(rates) < self.bollinger_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['close'].iloc[-2] < df['lower_band'].iloc[-2]:
            logger.info(f"价格跌破下轨，产生买入信号: {self.symbol}")
            return 1
        elif df['close'].iloc[-2] > df['upper_band'].iloc[-2]:
            logger.info(f"价格涨破上轨，产生卖出信号: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        布林带策略回测：
        价格突破下轨买入，突破上轨卖出
        返回信号序列：1买入，-1卖出，0无操作
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        for i in range(self.bollinger_period, len(df)):
            if df['close'].iloc[i-1] < df['lower_band'].iloc[i-1]:
                signals.iat[i] = 1
            elif df['close'].iloc[i-1] > df['upper_band'].iloc[i-1]:
                signals.iat[i] = -1
        return signals
