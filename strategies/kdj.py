import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, has_open_position, close_all, send_order
from logger import logger


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.kdj_period = 9
    
    def _calculate_indicators(self, df):
        """
        计算KDJ指标
        """
        low_min = df['low'].rolling(self.kdj_period).min()
        high_max = df['high'].rolling(self.kdj_period).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(com=2).mean()
        df['d'] = df['k'].ewm(com=2).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        return df

    def generate_signal(self):
        """
        KDJ策略实盘
        K线向上穿越D线（金叉）买入，K线向下穿越D线（死叉）卖出
        """
        rates = get_rates(self.symbol, self.timeframe, self.kdj_period + 30)
        if rates is None or len(rates) < self.kdj_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        # Golden cross
        if df['k'].iloc[-2] < df['d'].iloc[-2] and df['k'].iloc[-1] > df['d'].iloc[-1]:
            logger.info(f"KDJ Golden Cross, creating buy signal: {self.symbol}")
            return 1
        # Dead cross
        elif df['k'].iloc[-2] > df['d'].iloc[-2] and df['k'].iloc[-1] < df['d'].iloc[-1]:
            logger.info(f"KDJ Dead Cross, creating sell signal: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        KDJ回测方法
        根据金叉和死叉生成信号
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            # Golden cross
            if df['k'].iloc[i-1] < df['d'].iloc[i-1] and df['k'].iloc[i] > df['d'].iloc[i]:
                signals.iat[i] = 1
            # Dead cross
            elif df['k'].iloc[i-1] > df['d'].iloc[i-1] and df['k'].iloc[i] < df['d'].iloc[i]:
                signals.iat[i] = -1
        return signals
