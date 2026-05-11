import numpy as np
import pandas as pd
from utils.constants import TIMEFRAME_TO_MINUTES, RESAMPLE_RULES, PERIOD_M1


class MultiTimeframeDataStore:
    """多周期数据存储 — 从M1数据按需生成更高周期的OHLC数据

    回测/优化器模式：加载M1数据后，通过 pandas resample 生成 H1/H4/D1 等。
    实盘模式：不需要此组件（DataProvider 直接返回 MT5 原生数据）。

    用法:
        store = MultiTimeframeDataStore()
        store.load_m1_data(m1_rates_array)
        h1_data = store.get_historical_data("XAUUSD", PERIOD_H1, 100, current_index=5000)
    """

    def __init__(self):
        self._main_df = None
        self._main_timeframe = PERIOD_M1
        self._cache = {}  # timeframe: DataFrame

    def load_m1_data(self, rates: np.ndarray) -> None:
        """加载M1原始数据并初始化缓存"""
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        self._main_df = df
        self._cache = {PERIOD_M1: df}

    @property
    def main_df(self) -> pd.DataFrame:
        return self._main_df

    @property
    def length(self) -> int:
        if self._main_df is None:
            return 0
        return len(self._main_df)

    def _resample(self, timeframe: int) -> pd.DataFrame:
        """从M1数据 resample 到目标周期"""
        if timeframe == PERIOD_M1:
            return self._main_df

        rule = RESAMPLE_RULES.get(timeframe)
        if rule is None:
            raise ValueError(f"不支持的周期: {timeframe}")

        # 确保 M1 DataFrame 包含 OHLCV 列
        m1 = self._main_df[['open', 'high', 'low', 'close']].copy()
        if 'tick_volume' in self._main_df.columns:
            m1['tick_volume'] = self._main_df['tick_volume']

        resampled = m1.resample(rule, label='right', closed='right').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        })

        if 'tick_volume' in m1.columns:
            resampled['tick_volume'] = m1['tick_volume'].resample(
                rule, label='right', closed='right').sum()

        # 生成 time 列（Unix秒）
        resampled['time'] = [int(ts.timestamp()) for ts in resampled.index]

        # 丢掉由未来数据填充出来的最后一个未完成 bar
        last_m1_time = self._main_df.index[-1]
        last_complete = resampled[resampled.index <= last_m1_time]
        if len(last_complete) > 0:
            resampled = last_complete
        elif len(resampled) > 0:
            logger_warning = None
            try:
                from logger import logger
                logger_warning = logger
            except Exception:
                pass
            if logger_warning:
                logger_warning.warning(
                    f"resample {rule}: 所有bar都在最后M1时间之后，返回空DataFrame"
                )
            resampled = resampled.iloc[0:0]

        return resampled

    def ensure_timeframe(self, timeframe: int) -> pd.DataFrame:
        """确保指定周期的数据已缓存，若未缓存则从M1生成"""
        if timeframe in self._cache:
            return self._cache[timeframe]

        if self._main_df is None:
            raise RuntimeError("必须先调用 load_m1_data()")

        tf_df = self._resample(timeframe)
        self._cache[timeframe] = tf_df
        return tf_df

    def _find_tf_index(self, timeframe: int, m1_bar_index: int) -> int:
        """将M1 bar索引映射到目标周期的已完成bar索引

        返回目标周期中已经完成（不会看到未来）的最后一个 bar 的索引。
        即：目标周期 bar 的结束时间 <= 当前M1 bar 的时间。
        """
        if self._main_df is None:
            return -1

        tf_df = self.ensure_timeframe(timeframe)
        current_time = self._main_df.index[m1_bar_index]

        # 找到结束时间 <= current_time 的最后一个目标周期 bar
        positions = tf_df.index.get_indexer([current_time], method='bfill')
        tf_idx = positions[0]

        # bfill 返回的 bar 可能结束时间 > current_time（未完成），需要退回一个
        if tf_idx >= 0 and tf_idx < len(tf_df):
            if tf_df.index[tf_idx] > current_time:
                tf_idx -= 1

        return tf_idx

    def get_historical_data(self, symbol: str, timeframe: int,
                            count: int, current_index: int) -> np.ndarray | None:
        """从 current_index 位置向前获取 count 条指定周期的历史数据

        Args:
            symbol: 品种名（当前未使用，为接口一致性保留）
            timeframe: MT5周期常量
            count: 需要的bar数量
            current_index: 当前M1 bar的索引（0-based）

        Returns:
            numpy recarray 或 None（数据不足时）
        """
        if self._main_df is None or current_index >= self.length:
            return None

        tf_idx = self._find_tf_index(timeframe, current_index)

        if tf_idx < count - 1:
            return None

        tf_df = self.ensure_timeframe(timeframe)

        start = tf_idx - count + 1
        end = tf_idx + 1

        # 转换为MT5兼容的记录数组格式
        # 避免 index name 与 'time' 列冲突
        records = tf_df.iloc[start:end].reset_index(drop=True).to_records(index=False)
        return np.array(records)
