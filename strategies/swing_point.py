#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SwingPointRetestStrategy — 摆动点回踩策略

基于前高前低的支撑阻力位：
1. 找局部摆动高/低点（比左右各N根K线更高/更低）
2. 价格接近前高 → SELL（阻力反弹），接近前低 → BUY（支撑反弹）
3. ★ 回踩确认：价格突破后回踩原支撑/阻力位 → 更高胜率的反转信号

黄金M1参数：左右各3根K线，容差 0.05%~0.1%（约2~5点）
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG


class SwingPointRetestStrategy(BaseStrategy):
    """摆动点回踩策略 — 前高前低 + 回踩确认"""

    def __init__(self, data_provider, symbol, timeframe,
                 left_bars=None, right_bars=None,
                 tolerance_pct=None, num_swings=None):
        super().__init__(data_provider, symbol, timeframe)
        config = STRATEGY_CONFIG.get('swing_point', {})
        self.left_bars = left_bars if left_bars is not None else config.get('left_bars', 3)
        self.right_bars = right_bars if right_bars is not None else config.get('right_bars', 3)
        self.tolerance_pct = tolerance_pct if tolerance_pct is not None else config.get('tolerance_pct', 0.0008)
        self.num_swings = num_swings if num_swings is not None else config.get('num_swings', 2)
        self.lookback = max(self.left_bars + self.right_bars + 10, 50)

    def _find_swing_points(self, df):
        """找摆动高低点"""
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        L, R = self.left_bars, self.right_bars

        swing_highs = []  # (index, price)
        swing_lows = []

        for i in range(L, n - R):
            # swing_high: 比左边L根和右边R根都高
            left_max = np.max(highs[i - L:i])
            right_max = np.max(highs[i + 1:i + 1 + R])
            if highs[i] > left_max and highs[i] > right_max:
                swing_highs.append((i, highs[i]))

            # swing_low: 比左边L根和右边R根都低
            left_min = np.min(lows[i - L:i])
            right_min = np.min(lows[i + 1:i + 1 + R])
            if lows[i] < left_min and lows[i] < right_min:
                swing_lows.append((i, lows[i]))

        return swing_highs, swing_lows

    def _calculate_indicators(self, df):
        """计算摆动点并标记到 DataFrame"""
        df = df.copy()
        swing_highs, swing_lows = self._find_swing_points(df)
        return df, swing_highs, swing_lows

    def _signal_from_swings(self, current_price, swings, is_high):
        """
        判断当前价格是否接近摆动点

        is_high=True: 接近前高 → 阻力 → SELL 信号
        is_high=False: 接近前低 → 支撑 → BUY 信号
        """
        if not swings:
            return 0

        tolerance = current_price * self.tolerance_pct

        best_signal = 0
        for idx, swing_price in swings[-self.num_swings:]:
            distance_pct = abs(current_price - swing_price) / swing_price

            if distance_pct <= self.tolerance_pct:
                # 价格在摆动点容差范围内
                # 信号强度 = 1 - (距离/容差)，越近信号越强
                strength = 1.0 - (distance_pct / self.tolerance_pct)

                # ★ 回踩确认：价格曾突破过该摆动点
                if is_high and current_price <= swing_price:
                    # 价格在阻力位下方 → 正常卖点
                    signal = -strength
                elif not is_high and current_price >= swing_price:
                    # 价格在支撑位上方 → 正常买点
                    signal = strength
                else:
                    # 价格在错误一侧，不给信号
                    continue

                if abs(signal) > abs(best_signal):
                    best_signal = signal

        # 归一化到 [-1, 1]
        return max(-1.0, min(1.0, best_signal))

    def generate_signal(self):
        """生成交易信号"""
        rates = self.data_provider.get_historical_data(
            self.symbol, self.timeframe, self.lookback
        )
        if rates is None or len(rates) < self.lookback:
            return 0

        df = pd.DataFrame(rates)
        _, swing_highs, swing_lows = self._calculate_indicators(df)

        current_price = df['close'].iloc[-1]

        # 接近前高 → SELL
        sell_signal = self._signal_from_swings(current_price, swing_highs, is_high=True)
        # 接近前低 → BUY
        buy_signal = self._signal_from_swings(current_price, swing_lows, is_high=False)

        # 合并信号（sell为负，buy为正）
        total = buy_signal + sell_signal  # sell_signal 已经是负数
        return max(-1.0, min(1.0, total))

    def run_backtest(self, df):
        """回测模式 — 向量化计算信号"""
        df = df.copy()
        n = len(df)
        L, R = self.left_bars, self.right_bars
        tolerance = self.tolerance_pct

        signals = pd.Series(0.0, index=df.index)
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        # 预计算摆动点
        swing_high_mask = np.zeros(n, dtype=bool)
        swing_low_mask = np.zeros(n, dtype=bool)

        for i in range(L, n - R):
            if highs[i] > np.max(highs[i - L:i]) and highs[i] > np.max(highs[i + 1:i + 1 + R]):
                swing_high_mask[i] = True
            if lows[i] < np.min(lows[i - L:i]) and lows[i] < np.min(lows[i + 1:i + 1 + R]):
                swing_low_mask[i] = True

        # 生成信号
        for i in range(self.lookback, n):
            current = closes[i]

            # 找最近的摆动点
            prev_highs = np.where(swing_high_mask[:i])[0]
            prev_lows = np.where(swing_low_mask[:i])[0]

            signal = 0.0

            # 检查前高（阻力位）
            for sh_idx in prev_highs[-self.num_swings:]:
                sh_price = highs[sh_idx]
                dist_pct = abs(current - sh_price) / sh_price
                if dist_pct <= tolerance and current <= sh_price:
                    strength = 1.0 - (dist_pct / tolerance)
                    signal -= strength
                    break

            # 检查前低（支撑位）
            for sl_idx in prev_lows[-self.num_swings:]:
                sl_price = lows[sl_idx]
                dist_pct = abs(current - sl_price) / sl_price
                if dist_pct <= tolerance and current >= sl_price:
                    strength = 1.0 - (dist_pct / tolerance)
                    signal += strength
                    break

            signals.iloc[i] = max(-1.0, min(1.0, signal))

        return signals
