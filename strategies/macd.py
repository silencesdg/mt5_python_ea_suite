import pandas as pd
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG

class MACDStrategy(BaseStrategy):
    def __init__(self, data_provider, symbol, timeframe, fast_ema=None, slow_ema=None, signal_period=None):
        super().__init__(data_provider, symbol, timeframe)
        # 从配置中获取参数，如果传入参数则使用传入的参数
        config = STRATEGY_CONFIG.get('macd', {})
        self.fast_ema = fast_ema if fast_ema is not None else config.get('fast_ema', 12)
        self.slow_ema = slow_ema if slow_ema is not None else config.get('slow_ema', 26)
        self.signal_period = signal_period if signal_period is not None else config.get('signal_period', 9)

    def _calculate_indicators(self, df):
        df['exp12'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['exp26'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['dif'] = df['exp12'] - df['exp26']
        df['dea'] = df['dif'].ewm(span=self.signal_period, adjust=False).mean()
        return df

    def generate_signal(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.slow_ema + self.signal_period + 5)
        if rates is None or len(rates) < self.slow_ema + self.signal_period:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        if df['dif'].iloc[-1] > df['dea'].iloc[-1] and df['dif'].iloc[-2] <= df['dea'].iloc[-2]:
            return 1
        elif df['dif'].iloc[-1] < df['dea'].iloc[-1] and df['dif'].iloc[-2] >= df['dea'].iloc[-2]:
            return -1
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        signals[(df['dif'] > df['dea']) & (df['dif'].shift(1) <= df['dea'].shift(1))] = 1
        signals[(df['dif'] < df['dea']) & (df['dif'].shift(1) >= df['dea'].shift(1))] = -1
        return signals