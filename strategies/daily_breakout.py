import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
from utils import get_rates, has_open_position, close_all, send_order
from logger import logger

class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        self.timeframe = mt5.TIMEFRAME_M1

    def _calculate_indicators(self, df):
        """
        计算日内突破指标
        """
        df['time'] = pd.to_datetime(df['time'], unit='s')
        today = datetime.now().date()
        day_data = df[df['time'].dt.date == today]
        if day_data.empty:
            return df, None, None
        day_high = day_data['high'].max()
        day_low = day_data['low'].min()
        return df, day_high, day_low

    def generate_signal(self):
        """
        日内突破策略实盘
        当价格突破当日最高买入，突破当日最低卖出
        """
        rates = get_rates(self.symbol, self.timeframe, 1440) # 24 hours * 60 minutes
        if rates is None or len(rates) < 2:
            return 0
        df = pd.DataFrame(rates)
        df, day_high, day_low = self._calculate_indicators(df)

        if day_high is None or day_low is None:
            return 0

        if df['close'].iloc[-2] > day_high:
            logger.info(f"价格突破当日最高，产生买入信号: {self.symbol}")
            return 1
        elif df['close'].iloc[-2] < day_low:
            logger.info(f"价格突破当日最低，产生卖出信号: {self.symbol}")
            return -1
        return 0

    def run_backtest(self, df):
        """
        日内突破回测方法
        计算每个交易日的高低点，突破买卖信号
        """
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], unit='s')

        signals = pd.Series(0, index=df.index)

        grouped = df.groupby(df['time'].dt.date)

        for date, group in grouped:
            day_high = group['high'].max()
            day_low = group['low'].min()
            for i, row in group.iterrows():
                if row['close'] > day_high:
                    signals.loc[i] = 1
                elif row['close'] < day_low:
                    signals.loc[i] = -1

        return signals
