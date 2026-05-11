import pandas as pd
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略 — 价格突破布林带后等待回归确认再入场（与 BollingerStrategy 的即时入场区分）"""

    def __init__(self, data_provider, symbol, timeframe, period=None, std_dev=None):
        super().__init__(data_provider, symbol, timeframe)
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
        if rates is None or len(rates) < self.period + 1:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)

        # 回归确认：价格曾突破边界，现已回归内侧
        prev_close = df['close'].iloc[-2]
        prev_lower = df['lower_band'].iloc[-2]
        prev_upper = df['upper_band'].iloc[-2]
        curr_close = df['close'].iloc[-1]
        curr_lower = df['lower_band'].iloc[-1]
        curr_upper = df['upper_band'].iloc[-1]

        # 买入：上一根K线跌破下轨，当前回升至下轨上方（回归确认）
        if prev_close < prev_lower and curr_close >= curr_lower:
            return 1
        # 卖出：上一根K线突破上轨，当前回落至上轨下方（回归确认）
        elif prev_close > prev_upper and curr_close <= curr_upper:
            return -1
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        # 前一根在轨外 + 当前回归轨内 = 买入
        signals[(df['close'].shift(1) < df['lower_band'].shift(1)) & (df['close'] >= df['lower_band'])] = 1
        # 前一根在轨外 + 当前回归轨内 = 卖出
        signals[(df['close'].shift(1) > df['upper_band'].shift(1)) & (df['close'] <= df['upper_band'])] = -1
        return signals
