import pandas as pd
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG

class KDJStrategy(BaseStrategy):
    def __init__(self, data_provider, symbol, timeframe, period=None):
        super().__init__(data_provider, symbol, timeframe)
        # 从配置中获取参数，如果传入参数则使用传入的参数
        config = STRATEGY_CONFIG.get('kdj', {})
        self.period = period if period is not None else config.get('period', 14)
    
    def _calculate_indicators(self, df):
        low_min = df['low'].rolling(self.period).min()
        high_max = df['high'].rolling(self.period).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(com=2).mean()
        df['d'] = df['k'].ewm(com=2).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']
        return df

    def generate_signal(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.period + 5)
        if rates is None or len(rates) < self.period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['k'].iloc[-1] > df['d'].iloc[-1] and df['k'].iloc[-2] < df['d'].iloc[-2]:
            return 1
        elif df['k'].iloc[-1] < df['d'].iloc[-1] and df['k'].iloc[-2] > df['d'].iloc[-2]:
            return -1
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        signals[(df['k'] > df['d']) & (df['k'].shift(1) < df['d'].shift(1))] = 1
        signals[(df['k'] < df['d']) & (df['k'].shift(1) > df['d'].shift(1))] = -1
        return signals