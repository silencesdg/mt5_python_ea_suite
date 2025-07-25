import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, has_open_position, close_all, send_order
from logger import logger


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.turtle_period = 20

    def _calculate_indicators(self, df):
        """
        计算海龟交易指标
        """
        df['high_20'] = df['high'].rolling(self.turtle_period).max()
        df['low_20'] = df['low'].rolling(self.turtle_period).min()
        return df

    def generate_signal(self):
        """
        海龟交易策略实盘
        20日最高突破买入，20日最低突破卖出
        """
        rates = get_rates(self.symbol, self.timeframe, self.turtle_period + 30)
        if rates is None or len(rates) < self.turtle_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['close'].iloc[-2] > df['high_20'].iloc[-3]:
            logger.info(f"价格突破{self.turtle_period}日最高点，产生买入信号: {self.symbol}")
            return 1
        elif df['close'].iloc[-2] < df['low_20'].iloc[-3]:
            logger.info(f"价格突破{self.turtle_period}日最低点，产生卖出信号: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        海龟交易回测方法
        根据20日高低突破生成买卖信号
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        for i in range(self.turtle_period, len(df)):
            if df['close'].iloc[i-1] > df['high_20'].iloc[i-2]:
                signals.iat[i] = 1
            elif df['close'].iloc[i-1] < df['low_20'].iloc[i-2]:
                signals.iat[i] = -1
        return signals
