import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, has_open_position, close_all, send_order
from logger import logger


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.fast_ema_period = 12
        self.slow_ema_period = 26
        self.signal_period = 9

    def _calculate_indicators(self, df):
        """
        计算MACD指标
        """
        df['exp12'] = df['close'].ewm(span=self.fast_ema_period, adjust=False).mean()
        df['exp26'] = df['close'].ewm(span=self.slow_ema_period, adjust=False).mean()
        df['dif'] = df['exp12'] - df['exp26']
        df['dea'] = df['dif'].ewm(span=self.signal_period, adjust=False).mean()
        return df

    def generate_signal(self):
        """
        MACD策略实盘
        DIF线上穿DEA买入，反之卖出
        """
        rates = get_rates(self.symbol, self.timeframe, self.slow_ema_period + self.signal_period + 30)
        if rates is None or len(rates) < self.slow_ema_period + self.signal_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['dif'].iloc[-2] > df['dea'].iloc[-2] and df['dif'].iloc[-3] <= df['dea'].iloc[-3]:
            logger.info(f"DIF线上穿DEA，产生买入信号: {self.symbol}")
            return 1
        elif df['dif'].iloc[-2] < df['dea'].iloc[-2] and df['dif'].iloc[-3] >= df['dea'].iloc[-3]:
            logger.info(f"DIF线下穿DEA，产生卖出信号: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        MACD回测方法
        根据DIF和DEA金叉死叉生成信号
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        for i in range(2, len(df)):
            if df['dif'].iloc[i-1] > df['dea'].iloc[i-1] and df['dif'].iloc[i-2] <= df['dea'].iloc[i-2]:
                signals.iat[i] = 1
            elif df['dif'].iloc[i-1] < df['dea'].iloc[i-1] and df['dif'].iloc[i-2] >= df['dea'].iloc[i-2]:
                signals.iat[i] = -1
        return signals
