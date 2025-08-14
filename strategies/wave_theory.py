import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG, DATA_CONFIG

class WaveTheoryStrategy(BaseStrategy):
    def __init__(self, data_provider, symbol, timeframe, m1_bars_count=None, ema_short=None, ema_medium=None, ema_long=None, wave_period=None, range_period=None, adx_period=None, momentum_period=None, range_threshold=None, adx_threshold=None):
        super().__init__(data_provider, symbol, timeframe)
        # 从配置中获取参数，如果传入参数则使用传入的参数
        config = STRATEGY_CONFIG.get('wave_theory', {})
        self.m1_bars_count = m1_bars_count if m1_bars_count is not None else DATA_CONFIG.get('m1_bars_count', 500)
        self.ema_short = ema_short if ema_short is not None else config.get('ema_short', 5)
        self.ema_medium = ema_medium if ema_medium is not None else config.get('ema_medium', 13)
        self.ema_long = ema_long if ema_long is not None else config.get('ema_long', 34)
        self.wave_period = wave_period if wave_period is not None else config.get('wave_period', 21)
        self.range_period = range_period if range_period is not None else config.get('range_period', 20)
        self.adx_period = adx_period if adx_period is not None else config.get('adx_period', 14)
        self.retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.momentum_period = momentum_period if momentum_period is not None else config.get('momentum_period', 14)
        self.range_threshold = range_threshold if range_threshold is not None else config.get('range_threshold', 0.005)
        self.adx_threshold = adx_threshold if adx_threshold is not None else config.get('adx_threshold', 25)

    def _calculate_adx(self, df):
        high = df['high']
        low = df['low']
        close = df['close']
        df['tr'] = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        df['up_move'] = high - high.shift(1)
        df['down_move'] = low.shift(1) - low
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        df['plus_di'] = 100 * (df['plus_dm'].ewm(span=self.adx_period).mean() / df['tr'].ewm(span=self.adx_period).mean())
        df['minus_di'] = 100 * (df['minus_dm'].ewm(span=self.adx_period).mean() / df['tr'].ewm(span=self.adx_period).mean())
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        return df['dx'].ewm(span=self.adx_period).mean()

    def _identify_wave_points(self, df):
        window_size = self.wave_period * 2 + 1
        df['local_high'] = df['high'].rolling(window=window_size, center=False).max().shift(-self.wave_period)
        df['local_low'] = df['low'].rolling(window=window_size, center=False).min().shift(-self.wave_period)
        wave_points = pd.Series(0, index=df.index)
        wave_points[df['high'] == df['local_high']] = 1
        wave_points[df['low'] == df['local_low']] = -1
        return wave_points

    def _calculate_fibonacci_levels(self, df):
        # 确保fibonacci列存在
        for level in self.retracement_levels:
            col_name = f'fib_{level}'
            if col_name not in df.columns:
                df[col_name] = np.nan
        
        last_peak_idx, last_trough_idx = None, None
        for i in range(1, len(df)):
            if df['potential_wave_points'].iat[i] == 1:
                last_peak_idx = df.index[i]
            elif df['potential_wave_points'].iat[i] == -1:
                last_trough_idx = df.index[i]
            
            if last_peak_idx is not None and last_trough_idx is not None:
                try:
                    if last_peak_idx > last_trough_idx:
                        high_price, low_price = df['high'].loc[last_peak_idx], df['low'].loc[last_trough_idx]
                        price_range = high_price - low_price
                        for level in self.retracement_levels:
                            col_name = f'fib_{level}'
                            df.loc[df.index[i], col_name] = high_price - price_range * level
                    else:
                        low_price, high_price = df['low'].loc[last_trough_idx], df['high'].loc[last_peak_idx]
                        price_range = high_price - low_price
                        for level in self.retracement_levels:
                            col_name = f'fib_{level}'
                            df.loc[df.index[i], col_name] = low_price + price_range * level
                except Exception as e:
                    # 如果计算出错，跳过此位置
                    continue
        return df

    def _is_sideways_market(self, df):
        if len(df) < self.range_period: return False
        adx_value = df['adx'].iloc[-1]
        range_pct = df['range_pct'].iloc[-1]
        return (adx_value < self.adx_threshold) and (range_pct < self.range_threshold * 100)

    def _calculate_indicators(self, df):
        df['ema_short'] = df['close'].ewm(span=self.ema_short).mean()
        df['ema_medium'] = df['close'].ewm(span=self.ema_medium).mean()
        df['ema_long'] = df['close'].ewm(span=self.ema_long).mean()
        df['momentum'] = df['close'].diff(self.momentum_period) / df['close'].shift(self.momentum_period) * 100
        df['high_max'] = df['high'].rolling(self.range_period).max()
        df['low_min'] = df['low'].rolling(self.range_period).min()
        df['range_pct'] = (df['high_max'] - df['low_min']) / df['close'] * 100
        df['adx'] = self._calculate_adx(df)
        df['potential_wave_points'] = self._identify_wave_points(df)
        df = self._calculate_fibonacci_levels(df)
        return df

    def generate_signal(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.m1_bars_count)
        if rates is None or len(rates) < self.wave_period * 3:
            return 0
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)
        is_sideways = self._is_sideways_market(df)
        current_momentum = df['momentum'].iloc[-1]
        current_price = df['close'].iloc[-1]
        if is_sideways:
            upper_bound, lower_bound = df['high_max'].iloc[-1], df['low_min'].iloc[-1]
            if current_price > upper_bound * 0.98 and current_momentum < 0: return -1
            elif current_price < lower_bound * 1.02 and current_momentum > 0: return 1
        else:
            ema_alignment = (df['ema_short'].iloc[-1] > df['ema_medium'].iloc[-1] > df['ema_long'].iloc[-1])
            if f'fib_0.618' in df.columns and not pd.isna(df[f'fib_0.618'].iloc[-1]):
                fib_618 = df[f'fib_0.618'].iloc[-1]
                if abs(current_price - fib_618) / fib_618 < 0.01:
                    if ema_alignment and current_momentum > 0: return 1
                    elif not ema_alignment and current_momentum < 0: return -1
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df = self._calculate_indicators(df)
        signals = pd.Series(0, index=df.index)
        for i in range(self.wave_period * 3, len(df)):
            adx_value = df['adx'].iloc[i]
            range_pct = df['range_pct'].iloc[i]
            is_sideways = (adx_value < self.adx_threshold) and (range_pct < self.range_threshold * 100)
            current_price = df['close'].iloc[i]
            current_momentum = df['momentum'].iloc[i]
            if is_sideways:
                upper_bound, lower_bound = df['high_max'].iloc[i], df['low_min'].iloc[i]
                if current_price > upper_bound * 0.98 and current_momentum < 0: signals.iat[i] = -1
                elif current_price < lower_bound * 1.02 and current_momentum > 0: signals.iat[i] = 1
            else:
                ema_alignment = (df['ema_short'].iloc[i] > df['ema_medium'].iloc[i] > df['ema_long'].iloc[i])
                if f'fib_0.618' in df.columns and not pd.isna(df[f'fib_0.618'].iloc[i]):
                    fib_618 = df[f'fib_0.618'].iloc[i]
                    if abs(current_price - fib_618) / fib_618 < 0.01:
                        if ema_alignment and current_momentum > 0: signals.iat[i] = 1
                        elif not ema_alignment and current_momentum < 0: signals.iat[i] = -1
        return signals