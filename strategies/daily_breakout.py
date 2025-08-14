import pandas as pd
from datetime import datetime
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG

class DailyBreakoutStrategy(BaseStrategy):
    def __init__(self, data_provider, symbol, timeframe, bars_count=None):
        super().__init__(data_provider, symbol, timeframe)
        # 从配置中获取参数，如果传入参数则使用传入的参数
        config = STRATEGY_CONFIG.get('daily_breakout', {})
        self.bars_count = bars_count if bars_count is not None else config.get('bars_count', 1440)

    def _calculate_indicators(self, df):
        df['time'] = pd.to_datetime(df['time'], unit='s')
        today = datetime.now().date()
        day_data = df[df['time'].dt.date == today]
        if day_data.empty:
            return df, None, None
        day_high = day_data['high'].max()
        day_low = day_data['low'].min()
        return df, day_high, day_low

    def generate_signal(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.bars_count)
        if rates is None or len(rates) < 2:
            return 0
        df = pd.DataFrame(rates)
        df, day_high, day_low = self._calculate_indicators(df)

        if day_high is None or day_low is None:
            return 0

        if df['close'].iloc[-1] > day_high:
            return 1
        elif df['close'].iloc[-1] < day_low:
            return -1
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['date'] = df['time'].dt.date
        
        daily_highs = df.groupby('date')['high'].transform('max')
        daily_lows = df.groupby('date')['low'].transform('max')

        signals = pd.Series(0, index=df.index)
        signals[df['close'] > daily_highs.shift(1)] = 1
        signals[df['close'] < daily_lows.shift(1)] = -1
        return signals