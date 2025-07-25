import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, close_all, send_order
from logger import logger


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.rsi_period = 14
        self.rsi_buy_threshold = 30
        self.rsi_sell_threshold = 70

    def _calculate_indicators(self, df):
        """
        计算RSI指标
        """
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def generate_signal(self):
        """
        RSI策略实盘：
        RSI < 30买入，RSI > 70卖出。
        """
        rates = get_rates(self.symbol, self.timeframe, self.rsi_period + 30)
        if rates is None or len(rates) < self.rsi_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['rsi'].iloc[-2] < self.rsi_buy_threshold:
            logger.info(f"RSI小于{self.rsi_buy_threshold}，产生买入信号: {self.symbol}")
            return 1
        elif df['rsi'].iloc[-2] > self.rsi_sell_threshold:
            logger.info(f"RSI大于{self.rsi_sell_threshold}，产生卖出信号: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        RSI回测：
        RSI < 30买入，RSI > 70卖出。
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        for i in range(self.rsi_period, len(df)):
            if df['rsi'].iloc[i-1] < self.rsi_buy_threshold:
                signals.iat[i] = 1
            elif df['rsi'].iloc[i-1] > self.rsi_sell_threshold:
                signals.iat[i] = -1
        return signals
