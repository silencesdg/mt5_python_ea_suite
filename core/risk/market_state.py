import pandas as pd
import numpy as np
from logger import logger
import config
from config import (
    MARKET_STATE_CONFIG, SYMBOL, DEFAULT_WEIGHTS,
    TREND_INDICATOR_WEIGHTS, TREND_THRESHOLDS,
    MARKET_STATE_WEIGHTS, CONFIDENCE_THRESHOLDS
)
from utils.constants import PERIOD_H1
from core.data.multi_tf import MultiTimeframeDataStore


class MarketStateAnalyzer:
    """市场状态分析器（重写版）

    两种工作模式：
      - 实盘模式：get_market_state() 实时从 DataProvider 获取H1数据计算
      - 回测模式：precompute_states() 一次性预计算全序列，get_market_state(bar_index) 从缓存读取

    关键改进：
      1. timeframe 可配置，不再硬编码
      2. precompute_states() 使用 MultiTimeframeDataStore 获取正确周期的数据
      3. get_strategy_weights() 接受 individual_weights 参数使优化器权重基因真正生效
    """

    def __init__(self, data_provider=None,
                 timeframe: int = PERIOD_H1,
                 market_state_params: dict = None,
                 trend_weights: dict = None,
                 trend_thresholds: dict = None,
                 confidence_thresholds: dict = None):
        self.data_provider = data_provider
        self.symbol = SYMBOL
        self.timeframe = timeframe

        # 参数合并
        ms_config = market_state_params or MARKET_STATE_CONFIG
        self.trend_period = ms_config.get("trend_period", 50)
        self.retracement_tolerance = ms_config.get("retracement_tolerance", 0.30)
        self.volume_period = ms_config.get("volume_period", 20)
        self.volume_ma_period = ms_config.get("volume_ma_period", 10)
        self.hourly_data_count = ms_config.get("hourly_data_count", 100)

        self.indicator_weights = trend_weights or TREND_INDICATOR_WEIGHTS
        self.thresholds = trend_thresholds or TREND_THRESHOLDS
        self.confidence_thresholds = confidence_thresholds or CONFIDENCE_THRESHOLDS

        # 趋势跟踪状态
        self.current_trend = "none"
        self.trend_peak = 0.0
        self.trend_trough = float('inf')

        # 回测模式缓存
        self._precomputed_states: list | None = None

    # ── 指标计算（从现有逻辑提取）──

    def _calculate_volume_indicators(self, df):
        df['volume'] = df.get('tick_volume', df.get('volume', pd.Series(0, index=df.index)))
        df['volume_ma'] = df['volume'].rolling(self.volume_ma_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        volume_corr = df['close'].rolling(self.volume_period).corr(df['volume'])
        last_corr = volume_corr.iloc[-1] if len(volume_corr) > 0 and not pd.isna(volume_corr.iloc[-1]) else 0
        return df, last_corr

    def _calculate_momentum_indicators(self, df):
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=13, min_periods=14).mean()
        avg_loss = loss.ewm(com=13, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, float('nan')).fillna(0)
        df['rsi'] = 100 - (100 / (1 + rs))
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        return df

    def _calculate_price_breakout_score(self, df):
        current_price = df['close'].iloc[-1]
        lookback = min(self.trend_period, len(df) - 1)
        if lookback < 1:
            return 0.5
        high_period = df['high'].rolling(self.trend_period).max().iloc[-2] if len(df) >= 2 else df['high'].iloc[-1]
        low_period = df['low'].rolling(self.trend_period).min().iloc[-2] if len(df) >= 2 else df['low'].iloc[-1]
        price_range = high_period - low_period
        if price_range == 0:
            return 0.5
        price_position = (current_price - low_period) / price_range
        if current_price > high_period:
            return 0.8
        elif current_price < low_period:
            return 0.2
        return float(np.clip(price_position, 0, 1))

    def _calculate_volume_score(self, df, volume_corr):
        current_volume = df['volume'].iloc[-1]
        volume_ma = df['volume_ma'].iloc[-1]
        volume_ratio = current_volume / volume_ma if (not pd.isna(volume_ma) and volume_ma > 0) else 1
        score = 0.5
        if volume_ratio > self.thresholds.get('volume_spike', 1.5):
            score += 0.3
        if volume_corr > 0.5:
            score += 0.2
        elif volume_corr < -0.5:
            score -= 0.2
        return float(np.clip(score, 0, 1))

    def _calculate_momentum_score(self, df):
        rsi_val = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50
        macd_hist = df['macd_histogram'].iloc[-1] if not pd.isna(df['macd_histogram'].iloc[-1]) else 0
        score = 0.5
        if rsi_val > self.thresholds.get('overbought', 70):
            score += 0.2
        elif rsi_val < self.thresholds.get('oversold', 30):
            score -= 0.2
        if macd_hist > 0:
            score += 0.2
        else:
            score -= 0.2
        return float(np.clip(score, 0, 1))

    def _calculate_ma_score(self, df):
        current_price = df['close'].iloc[-1]
        ma20 = df['ma20'].iloc[-1] if not pd.isna(df['ma20'].iloc[-1]) else current_price
        ma50 = df['ma50'].iloc[-1] if not pd.isna(df['ma50'].iloc[-1]) else current_price
        score = 0.5
        if current_price > ma20 > ma50:
            score += 0.3
        elif current_price < ma20 < ma50:
            score -= 0.3
        if len(df) >= 3:
            ma20_slope = df['ma20'].iloc[-1] - df['ma20'].iloc[-3]
            ma50_slope = df['ma50'].iloc[-1] - df['ma50'].iloc[-3]
            if ma20_slope > 0 and ma50_slope > 0:
                score += 0.2
            elif ma20_slope < 0 and ma50_slope < 0:
                score -= 0.2
        return float(np.clip(score, 0, 1))

    def _calculate_state_from_df(self, df):
        """从 DataFrame 计算市场状态 — 提取自原 get_market_state() 的 DataFrame 处理部分"""
        df = df.copy()
        df, volume_corr = self._calculate_volume_indicators(df)
        df = self._calculate_momentum_indicators(df)

        price_score = self._calculate_price_breakout_score(df)
        volume_score = self._calculate_volume_score(df, volume_corr)
        momentum_score = self._calculate_momentum_score(df)
        ma_score = self._calculate_ma_score(df)

        trend_strength = (
            price_score * self.indicator_weights.get('price_breakout', 0.35) +
            volume_score * self.indicator_weights.get('volume_confirmation', 0.25) +
            momentum_score * self.indicator_weights.get('momentum oscillator', 0.20) +
            ma_score * self.indicator_weights.get('moving_average', 0.20)
        )

        strong = self.thresholds.get('strong_trend', 0.6)
        weak = self.thresholds.get('weak_trend', 0.3)

        if trend_strength >= strong:
            state, confidence = "uptrend", min(0.95, trend_strength)
        elif trend_strength <= (1 - strong):
            state, confidence = "downtrend", min(0.95, 1 - trend_strength)
        elif trend_strength >= weak:
            state, confidence = "ranging", 0.6
        else:
            state, confidence = "none", 0.4

        return state, confidence

    def _update_trend_state(self, df, new_state):
        current_price = df['close'].iloc[-1]
        if new_state == "uptrend":
            self.current_trend = "uptrend"
            self.trend_peak = max(self.trend_peak, current_price)
        elif new_state == "downtrend":
            self.current_trend = "downtrend"
            self.trend_trough = min(self.trend_trough, current_price)
        else:
            if self.current_trend == "uptrend" and current_price < self.trend_peak * (1 - self.retracement_tolerance):
                self.current_trend = "none"
            elif self.current_trend == "downtrend" and current_price > self.trend_trough * (1 + self.retracement_tolerance):
                self.current_trend = "none"

    # ── 预计算（回测/优化器模式）──

    def precompute_from_multitf(self, multi_tf: MultiTimeframeDataStore) -> None:
        """从 MultiTimeframeDataStore 预计算全序列市场状态

        这是回测和优化器模式的核心入口。调用后，get_market_state(bar_index)
        直接从缓存读取，不再需要实时计算。

        Args:
            multi_tf: 已加载M1数据的 MultiTimeframeDataStore 实例
        """
        h1_df = multi_tf.ensure_timeframe(self.timeframe)
        if h1_df is None or len(h1_df) == 0:
            logger.warning("H1数据为空，所有市场状态设为 'none'")
            self._precomputed_states = [("none", 0.0)] * multi_tf.length
            return

        m1_index = multi_tf.main_df.index
        h1_index = h1_df.index
        m1_length = len(m1_index)

        # 计算每个H1 bar的状态
        h1_states = []
        for i in range(len(h1_df)):
            lookback = self.trend_period + 10
            start = max(0, i - lookback)
            df_slice = h1_df.iloc[start:i + 1]
            state, conf = self._calculate_state_from_df(df_slice)
            h1_states.append((state, conf))

        # 映射到M1时间轴
        self._precomputed_states = []
        h1_pos = 0
        for m1_time in m1_index:
            while h1_pos < len(h1_index) - 1 and m1_time >= h1_index[h1_pos + 1]:
                h1_pos += 1
            if h1_pos < len(h1_states):
                self._precomputed_states.append(h1_states[h1_pos])
            else:
                self._precomputed_states.append(("none", 0.0))

        # 前几个H1 bar之前的M1 bar标记为 "none"
        first_h1_time = h1_index[0] if len(h1_index) > 0 else None
        if first_h1_time is not None:
            for idx, m1_time in enumerate(m1_index):
                if m1_time < first_h1_time:
                    self._precomputed_states[idx] = ("none", 0.0)

        logger.info(f"市场状态预计算完成: {len(h1_df)} 个H1 bar → {len(self._precomputed_states)} 个M1 bar")

    # ── 状态获取 ──

    def get_market_state(self, bar_index: int = None) -> tuple:
        """获取市场状态

        回测模式（有预计算缓存）：从缓存读取第 bar_index 个状态
        实盘模式（无缓存）：通过 DataProvider 实时计算
        """
        if self._precomputed_states is not None:
            idx = bar_index if bar_index is not None else 0
            if idx >= len(self._precomputed_states):
                return "none", 0.0
            return self._precomputed_states[idx]

        # 实盘模式
        if self.data_provider is None:
            return "none", 0.0

        rates = self.data_provider.get_historical_data(
            self.symbol, self.timeframe, self.hourly_data_count
        )
        if rates is None or len(rates) < max(self.trend_period, 50):
            return "none", 0.0

        df = pd.DataFrame(rates)
        state, confidence = self._calculate_state_from_df(df)
        self._update_trend_state(df, state)
        return state, confidence

    # ── 权重计算 ──

    def get_strategy_weights(self, market_state: str, confidence: float,
                             individual_weights: dict = None) -> dict:
        """获取策略权重

        关键改进：individual_weights 参数使优化器的权重基因真正生效。

        Args:
            market_state: "uptrend" / "downtrend" / "ranging" / "none"
            confidence: 置信度 0~1
            individual_weights: 优化器传入 {config_key: weight}，非None时优先使用

        Returns:
            {config_key: weight} 用于信号加权组合
        """
        # ★ 优化器模式：使用基因中的权重
        if individual_weights is not None:
            base_weights = dict(individual_weights)
        else:
            # 正常模式：从配置获取市场状态对应权重
            base_weights = dict(config.MARKET_STATE_WEIGHTS.get(market_state, config.DEFAULT_WEIGHTS))

        high_conf = self.confidence_thresholds.get("high_confidence", 0.7)
        medium_conf = self.confidence_thresholds.get("medium_confidence", 0.4)

        if confidence > high_conf:
            return {k: v * confidence for k, v in base_weights.items()}
        elif confidence > medium_conf:
            return {
                k: (v * confidence + config.DEFAULT_WEIGHTS.get(k, 1.0) * (1 - confidence))
                for k, v in base_weights.items()
            }
        else:
            # 低置信度：individual_weights 优先（优化器模式），否则回退到 DEFAULT_WEIGHTS
            return dict(individual_weights) if individual_weights is not None else dict(config.DEFAULT_WEIGHTS)
