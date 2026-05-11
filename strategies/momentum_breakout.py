import pandas as pd
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG

class MomentumBreakoutStrategy(BaseStrategy):
    """动量突破策略 — 在通道突破基础上增加动量方向确认（与 TurtleStrategy 的纯突破区分）"""

    def __init__(self, data_provider, symbol, timeframe, period=None, momentum_period=None):
        super().__init__(data_provider, symbol, timeframe)
        config = STRATEGY_CONFIG.get('momentum_breakout', {})
        self.period = period if period is not None else config.get('period', 20)
        self.momentum_period = momentum_period if momentum_period is not None else config.get('momentum_period', 10)

    def _calculate_indicators(self, df):
        df['high_period'] = df['high'].rolling(self.period).max()
        df['low_period'] = df['low'].rolling(self.period).min()
        # 计算动量：当前收盘价相对于 N 根前的涨跌幅
        df['momentum'] = df['close'] - df['close'].shift(self.momentum_period)
        return df

    def generate_signal(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, max(self.period, self.momentum_period) + 5)
        if rates is None or len(rates) < max(self.period, self.momentum_period) + 1:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        close = df['close'].iloc[-1]
        breakout_high = df['high_period'].iloc[-2]
        breakout_low = df['low_period'].iloc[-2]
        momentum = df['momentum'].iloc[-1]

        # 突破上轨 + 正动量确认
        if close > breakout_high and momentum > 0:
            return 1
        # 跌破下轨 + 负动量确认
        elif close < breakout_low and momentum < 0:
            return -1
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        signals[(df['close'] > df['high_period'].shift(1)) & (df['momentum'] > 0)] = 1
        signals[(df['close'] < df['low_period'].shift(1)) & (df['momentum'] < 0)] = -1
        return signals
