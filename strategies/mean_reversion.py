import pandas as pd
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG

class MeanReversionStrategy(BaseStrategy):
    def __init__(self, data_provider, symbol, timeframe, period=None, std_dev=None):
        super().__init__(data_provider, symbol, timeframe)
        # 从配置中获取参数，如果传入参数则使用传入的参数
        config = STRATEGY_CONFIG.get('mean_reversion', {})
        self.period = period if period is not None else config.get('period', 20)
        self.std_dev = std_dev if std_dev is not None else config.get('std_dev', 2.0)

    def _calculate_indicators(self, df):
        mean = df['close'].rolling(self.period).mean()
        std = df['close'].rolling(self.period).std()
        df['upper_band'] = mean + self.std_dev * std
        df['lower_band'] = mean - self.std_dev * std
        return df

    def generate_signal(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.period + 5)
        if rates is None or len(rates) < self.period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['close'].iloc[-1] > df['upper_band'].iloc[-1]:
            return -1
        elif df['close'].iloc[-1] < df['lower_band'].iloc[-1]:
            return 1
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        signals[df['close'] > df['upper_band']] = -1
        signals[df['close'] < df['lower_band']] = 1
        return signals