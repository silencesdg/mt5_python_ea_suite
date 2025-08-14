import pandas as pd
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG

class MomentumBreakoutStrategy(BaseStrategy):
    def __init__(self, data_provider, symbol, timeframe, period=None):
        super().__init__(data_provider, symbol, timeframe)
        # 从配置中获取参数，如果传入参数则使用传入的参数
        config = STRATEGY_CONFIG.get('momentum_breakout', {})
        self.period = period if period is not None else config.get('period', 20)

    def _calculate_indicators(self, df):
        df['high_period'] = df['high'].rolling(self.period).max()
        df['low_period'] = df['low'].rolling(self.period).min()
        return df

    def generate_signal(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.period + 2)
        if rates is None or len(rates) < self.period + 1:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['close'].iloc[-1] > df['high_period'].iloc[-2]:
            return 1
        elif df['close'].iloc[-1] < df['low_period'].iloc[-2]:
            return -1
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > df['high_period'].shift(1)] = 1
        signals[df['close'] < df['low_period'].shift(1)] = -1
        return signals