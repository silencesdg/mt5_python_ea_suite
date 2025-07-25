import MetaTrader5 as mt5
import pandas as pd
from utils import get_rates, has_open_position, close_all, send_order
from logger import logger


class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1
        self.kdj_period = 9
        self.kdj_buy_threshold = 10
        self.kdj_sell_threshold = 90

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
        J值小于10买入，大于90卖出
        """
        rates = get_rates(self.symbol, self.timeframe, self.kdj_period + 30)
        if rates is None or len(rates) < self.kdj_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['j'].iloc[-2] < self.kdj_buy_threshold:
            logger.info(f"J值小于{self.kdj_buy_threshold}，产生买入信号: {self.symbol}")
            return 1
        elif df['j'].iloc[-2] > self.kdj_sell_threshold:
            logger.info(f"J值大于{self.kdj_sell_threshold}，产生卖出信号: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        KDJ回测方法
        根据J值极端生成信号
        """
        df = df.copy()
        df = self._calculate_indicators(df)

        signals = pd.Series(0, index=df.index)
        for i in range(self.kdj_period, len(df)):
            if df['j'].iloc[i-1] < self.kdj_buy_threshold:
                signals.iat[i] = 1
            elif df['j'].iloc[i-1] > self.kdj_sell_threshold:
                signals.iat[i] = -1
        return signals
