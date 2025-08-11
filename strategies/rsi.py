import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, close_all, send_order
from logger import logger


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.rsi_period = 14

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
        RSI策略实盘
        RSI上穿超卖线买入，下穿超买线卖出
        """
        rates = get_rates(self.symbol, self.timeframe, self.rsi_period + 30)
        if rates is None or len(rates) < self.rsi_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        oversold_level = 30
        overbought_level = 70
        # RSI crosses above oversold level
        if df['rsi'].iloc[-2] < oversold_level and df['rsi'].iloc[-1] > oversold_level:
            logger.info(f"RSI crosses above {oversold_level}, creating buy signal: {self.symbol}")
            return 1
        # RSI crosses below overbought level
        elif df['rsi'].iloc[-2] > overbought_level and df['rsi'].iloc[-1] < overbought_level:
            logger.info(f"RSI crosses below {overbought_level}, creating sell signal: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        RSI回测方法
        根据RSI穿越超买超卖线生成信号
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        oversold_level = 30
        overbought_level = 70
        for i in range(1, len(df)):
            # RSI crosses above oversold level
            if df['rsi'].iloc[i-1] < oversold_level and df['rsi'].iloc[i] > oversold_level:
                signals.iat[i] = 1
            # RSI crosses below overbought level
            elif df['rsi'].iloc[i-1] > overbought_level and df['rsi'].iloc[i] < overbought_level:
                signals.iat[i] = -1
        return signals
