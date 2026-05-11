"""遗传算法优化器（重写版）

关键改进：
  1. 权重基因通过 individual_weights → MarketStateAnalyzer.get_strategy_weights() 真正生效
  2. 使用 StrategyRegistry 单例消，除重复的策略列表
  3. 使用 MultiTimeframeDataStore 支持正确的多周期市场状态分析
  4. evaluate_fitness 逐 bar 调用 get_market_state(bar_index) 获取每根bar的动态权重
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing
import pandas as pd
import sys
import os
from datetime import datetime
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import MultiTimeframeDataStore, BacktestDataProvider
from core.risk.market_state import MarketStateAnalyzer
from core.risk.position import PositionManager
from core.signal.registry import StrategyRegistry
from core.signal.combiner import SignalCombiner
from config import (
    SYMBOL, TIMEFRAME, OPTIMIZER_COUNT, OPTIMIZER_START_DATE, OPTIMIZER_END_DATE,
    USE_DATE_RANGE, INITIAL_CAPITAL, SIGNAL_THRESHOLDS, DEFAULT_WEIGHTS,
    RISK_CONFIG, GENETIC_OPTIMIZER_CONFIG,
    MARKET_STATE_CONFIG, TREND_INDICATOR_WEIGHTS, TREND_THRESHOLDS, CONFIDENCE_THRESHOLDS
)
from utils.constants import PERIOD_H1
from core.utils import get_rates, initialize, shutdown
from logger import logger

# 导入策略模块确保 StrategyRegistry 已注册
import strategies  # noqa: F401

# ── 全局缓存 ──
_multi_tf: MultiTimeframeDataStore | None = None
_cached_signals: pd.DataFrame | None = None
_registry: StrategyRegistry | None = None


def init_worker():
    """多进程worker初始化：抑制日志噪音"""
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    for name in ["StrategyLogger", "PositionManager", "RiskController", "DataProvider"]:
        logging.getLogger(name).setLevel(logging.WARNING)


# ── 参数定义（与旧版一致，保持兼容）──
PARAMETER_DEFINITIONS = [
    # MACrossStrategy
    {'name': 'ma_cross_short_window', 'type': 'int', 'min': 3, 'max': 15, 'strategy': 'MACrossStrategy'},
    {'name': 'ma_cross_long_window',  'type': 'int', 'min': 10, 'max': 60, 'strategy': 'MACrossStrategy'},
    # RSIStrategy
    {'name': 'rsi_period',     'type': 'int', 'min': 7, 'max': 21, 'strategy': 'RSIStrategy'},
    {'name': 'rsi_overbought', 'type': 'int', 'min': 65, 'max': 80, 'strategy': 'RSIStrategy'},
    {'name': 'rsi_oversold',   'type': 'int', 'min': 20, 'max': 35, 'strategy': 'RSIStrategy'},
    # BollingerStrategy
    {'name': 'bollinger_period',  'type': 'int', 'min': 10, 'max': 50, 'strategy': 'BollingerStrategy'},
    {'name': 'bollinger_std_dev', 'type': 'float', 'min': 1.5, 'max': 3.0, 'strategy': 'BollingerStrategy'},
    # MACDStrategy
    {'name': 'macd_fast_ema',      'type': 'int', 'min': 8, 'max': 20, 'strategy': 'MACDStrategy'},
    {'name': 'macd_slow_ema',      'type': 'int', 'min': 20, 'max': 35, 'strategy': 'MACDStrategy'},
    {'name': 'macd_signal_period', 'type': 'int', 'min': 5, 'max': 15, 'strategy': 'MACDStrategy'},
    # MeanReversionStrategy
    {'name': 'mean_reversion_period',  'type': 'int', 'min': 10, 'max': 50, 'strategy': 'MeanReversionStrategy'},
    {'name': 'mean_reversion_std_dev', 'type': 'float', 'min': 1.5, 'max': 3.0, 'strategy': 'MeanReversionStrategy'},
    # MomentumBreakoutStrategy
    {'name': 'momentum_breakout_period', 'type': 'int', 'min': 10, 'max': 50, 'strategy': 'MomentumBreakoutStrategy'},
    {'name': 'momentum_breakout_momentum_period', 'type': 'int', 'min': 5, 'max': 30, 'strategy': 'MomentumBreakoutStrategy'},
    # KDJStrategy
    {'name': 'kdj_period', 'type': 'int', 'min': 5, 'max': 21, 'strategy': 'KDJStrategy'},
    # TurtleStrategy
    {'name': 'turtle_period', 'type': 'int', 'min': 10, 'max': 50, 'strategy': 'TurtleStrategy'},
    # WaveTheoryStrategy
    {'name': 'wave_ema_short',       'type': 'int', 'min': 3, 'max': 10, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_ema_medium',      'type': 'int', 'min': 8, 'max': 20, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_ema_long',        'type': 'int', 'min': 20, 'max': 50, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_period',          'type': 'int', 'min': 10, 'max': 40, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_range_period',    'type': 'int', 'min': 10, 'max': 40, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_adx_period',      'type': 'int', 'min': 10, 'max': 25, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_momentum_period', 'type': 'int', 'min': 7, 'max': 21, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_range_threshold', 'type': 'float', 'min': 0.002, 'max': 0.01, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_adx_threshold',   'type': 'int', 'min': 15, 'max': 35, 'strategy': 'WaveTheoryStrategy'},
    # DailyBreakoutStrategy
    {'name': 'daily_breakout_bars_count', 'type': 'int', 'min': 720, 'max': 2880, 'strategy': 'DailyBreakoutStrategy'},
    # Signal Thresholds
    {'name': 'buy_threshold',  'type': 'float', 'min': 0.5, 'max': 3.0, 'strategy': 'signal'},
    {'name': 'sell_threshold', 'type': 'float', 'min': -3.0, 'max': -0.5, 'strategy': 'signal'},
    # Risk Management
    {'name': 'stop_loss_pct',          'type': 'float', 'min': -0.05, 'max': -0.01, 'strategy': 'risk'},
    {'name': 'profit_retracement_pct', 'type': 'float', 'min': 0.05, 'max': 0.20, 'strategy': 'risk'},
    {'name': 'min_profit_for_trailing', 'type': 'float', 'min': 0.005, 'max': 0.02, 'strategy': 'risk'},
    {'name': 'take_profit_pct',        'type': 'float', 'min': 0.10, 'max': 0.50, 'strategy': 'risk'},
    {'name': 'max_holding_minutes',    'type': 'int', 'min': 30, 'max': 180, 'strategy': 'risk'},
    {'name': 'min_profit_for_time_exit', 'type': 'float', 'min': 0.002, 'max': 0.01, 'strategy': 'risk'},
    # ★ Strategy Weights — 这些基因现在真正影响适应度
    {'name': 'weight_MACrossStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_RSIStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_BollingerStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_MeanReversionStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_MomentumBreakoutStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_MACDStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_KDJStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_TurtleStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_DailyBreakoutStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_WaveTheoryStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    # Market State Analysis parameters
    {'name': 'market_trend_period', 'type': 'int', 'min': 20, 'max': 100, 'strategy': 'market_state'},
    {'name': 'market_retracement_tolerance', 'type': 'float', 'min': 0.1, 'max': 0.5, 'strategy': 'market_state'},
    {'name': 'market_volume_period', 'type': 'int', 'min': 10, 'max': 50, 'strategy': 'market_state'},
    {'name': 'market_volume_ma_period', 'type': 'int', 'min': 5, 'max': 30, 'strategy': 'market_state'},
    # Trend Indicator Weights
    {'name': 'trend_price_breakout_weight', 'type': 'float', 'min': 0.1, 'max': 0.5, 'strategy': 'trend_weights'},
    {'name': 'trend_volume_confirmation_weight', 'type': 'float', 'min': 0.1, 'max': 0.5, 'strategy': 'trend_weights'},
    {'name': 'trend_momentum_oscillator_weight', 'type': 'float', 'min': 0.1, 'max': 0.5, 'strategy': 'trend_weights'},
    {'name': 'trend_moving_average_weight', 'type': 'float', 'min': 0.1, 'max': 0.5, 'strategy': 'trend_weights'},
    # Trend Thresholds
    {'name': 'trend_strong_threshold', 'type': 'float', 'min': 0.4, 'max': 0.8, 'strategy': 'trend_thresholds'},
    {'name': 'trend_weak_threshold', 'type': 'float', 'min': 0.2, 'max': 0.5, 'strategy': 'trend_thresholds'},
    {'name': 'trend_volume_spike', 'type': 'float', 'min': 1.0, 'max': 3.0, 'strategy': 'trend_thresholds'},
    {'name': 'trend_oversold', 'type': 'int', 'min': 20, 'max': 40, 'strategy': 'trend_thresholds'},
    {'name': 'trend_overbought', 'type': 'int', 'min': 60, 'max': 80, 'strategy': 'trend_thresholds'},
    # Confidence Thresholds
    {'name': 'confidence_high', 'type': 'float', 'min': 0.5, 'max': 0.9, 'strategy': 'confidence'},
    {'name': 'confidence_medium', 'type': 'float', 'min': 0.3, 'max': 0.7, 'strategy': 'confidence'},
]

# class_name → config_key 映射（从 StrategyRegistry 获取）
_class_name_to_config_key = {
    "MACrossStrategy": "ma_cross",
    "RSIStrategy": "rsi",
    "BollingerStrategy": "bollinger",
    "MeanReversionStrategy": "mean_reversion",
    "MomentumBreakoutStrategy": "momentum_breakout",
    "MACDStrategy": "macd",
    "KDJStrategy": "kdj",
    "TurtleStrategy": "turtle",
    "DailyBreakoutStrategy": "daily_breakout",
    "WaveTheoryStrategy": "wave_theory",
}


def _parse_individual(individual):
    """解析个体基因为命名参数字典，并裁剪到定义范围"""
    parsed = {}
    for idx, param_def in enumerate(PARAMETER_DEFINITIONS):
        val = individual[idx]
        if param_def['type'] == 'int':
            val = int(round(val))
        # 裁剪到 [min, max] 防止变异越界
        val = max(param_def['min'], min(param_def['max'], val))
        parsed[param_def['name']] = val
    return parsed


def _extract_strategy_params(parsed: dict) -> dict:
    """从解析后的基因提取策略参数字典"""
    strategy_params = {}
    for param_def in PARAMETER_DEFINITIONS:
        strategy_name = param_def.get('strategy')
        if strategy_name in ('weight', 'signal', 'risk', 'market_state',
                             'trend_weights', 'trend_thresholds', 'confidence'):
            continue
        if strategy_name not in strategy_params:
            strategy_params[strategy_name] = {}

        # 参数名映射（保持与旧版兼容）
        param_name = param_def['name']
        if param_name == 'ma_cross_short_window':
            param_name = 'short_window'
        elif param_name == 'ma_cross_long_window':
            param_name = 'long_window'
        elif param_name == 'rsi_period':
            param_name = 'period'
        elif param_name == 'rsi_overbought':
            param_name = 'overbought'
        elif param_name == 'rsi_oversold':
            param_name = 'oversold'
        elif param_name == 'bollinger_period':
            param_name = 'period'
        elif param_name == 'bollinger_std_dev':
            param_name = 'std_dev'
        elif param_name == 'macd_fast_ema':
            param_name = 'fast_ema'
        elif param_name == 'macd_slow_ema':
            param_name = 'slow_ema'
        elif param_name == 'macd_signal_period':
            param_name = 'signal_period'
        elif param_name == 'mean_reversion_period':
            param_name = 'period'
        elif param_name == 'mean_reversion_std_dev':
            param_name = 'std_dev'
        elif param_name == 'momentum_breakout_period':
            param_name = 'period'
        elif param_name == 'momentum_breakout_momentum_period':
            param_name = 'momentum_period'
        elif param_name == 'kdj_period':
            param_name = 'period'
        elif param_name == 'turtle_period':
            param_name = 'period'
        elif param_name.startswith("wave_"):
            param_name = param_name[5:]
            if param_name == "period":
                param_name = "wave_period"
        elif param_name == 'daily_breakout_bars_count':
            param_name = 'bars_count'

        strategy_params[strategy_name][param_name] = parsed[param_def['name']]
    return strategy_params


def _extract_weight_genes(parsed: dict) -> dict:
    """★ 提取权重基因 → {config_key: weight}

    这些权重会通过 individual_weights 参数传递给
    MarketStateAnalyzer.get_strategy_weights()，使权重基因真正生效。
    """
    weight_genes = {}
    for param_def in PARAMETER_DEFINITIONS:
        if param_def.get('strategy') != 'weight':
            continue
        class_name = param_def['name'].replace('weight_', '')
        config_key = _class_name_to_config_key.get(class_name, class_name.lower())
        weight_genes[config_key] = parsed[param_def['name']]
    return weight_genes


def _precompute_h1_states(multi_tf: MultiTimeframeDataStore, analyzer: MarketStateAnalyzer) -> list:
    """预计算H1市场状态序列 → 映射到M1时间轴"""
    h1_df = multi_tf.ensure_timeframe(PERIOD_H1)
    if h1_df is None or len(h1_df) == 0:
        return [("none", 0.0)] * multi_tf.length

    m1_index = multi_tf.main_df.index
    h1_index = h1_df.index

    # 计算每个H1 bar的状态
    h1_states = []
    for i in range(len(h1_df)):
        lookback = analyzer.trend_period + 10
        start = max(0, i - lookback)
        df_slice = h1_df.iloc[start:i + 1]
        state, conf = analyzer._calculate_state_from_df(df_slice)
        h1_states.append((state, conf))

    # 映射到M1时间轴
    m1_states = []
    h1_pos = 0
    first_h1_time = h1_index[0]

    for m1_time in m1_index:
        if m1_time < first_h1_time:
            m1_states.append(("none", 0.0))
            continue
        while h1_pos < len(h1_index) - 1 and m1_time >= h1_index[h1_pos + 1]:
            h1_pos += 1
        if h1_pos < len(h1_states):
            m1_states.append(h1_states[h1_pos])
        else:
            m1_states.append(("none", 0.0))

    return m1_states


def evaluate_fitness(individual, multi_tf: MultiTimeframeDataStore,
                    _signals_df_unused=None) -> tuple:
    """★ 重写的适应度函数 — 所有权重基因、策略参数和风控参数真正生效

    三个关键基因全部生效：
    1. 策略参数 → 重新实例化策略并 run_backtest → 影响信号序列
    2. 权重基因 → per-bar get_strategy_weights(individual_weights=...) → 影响信号组合
    3. 风控参数 → PositionManager 在回测中真正使用
    """
    parsed = _parse_individual(individual)

    # 1. 提取策略参数
    strategy_params = _extract_strategy_params(parsed)

    # 2. ★ 提取权重基因（核心修复）
    weight_genes = _extract_weight_genes(parsed)

    # 3. 提取风控参数
    risk_params = {
        'stop_loss_pct': parsed.get('stop_loss_pct', RISK_CONFIG.get('stop_loss_pct', -0.01)),
        'profit_retracement_pct': parsed.get('profit_retracement_pct', RISK_CONFIG.get('profit_retracement_pct', 0.10)),
        'min_profit_for_trailing': parsed.get('min_profit_for_trailing', RISK_CONFIG.get('min_profit_for_trailing', 0.01)),
        'take_profit_pct': parsed.get('take_profit_pct', RISK_CONFIG.get('take_profit_pct', 0.20)),
        'max_holding_minutes': parsed.get('max_holding_minutes', RISK_CONFIG.get('max_holding_minutes', 60)),
        'min_profit_for_time_exit': parsed.get('min_profit_for_time_exit', RISK_CONFIG.get('min_profit_for_time_exit', 0.005)),
        'max_daily_loss': parsed.get('max_daily_loss', RISK_CONFIG.get('max_daily_loss', -0.30)),
    }

    # 4. 提取市场状态和趋势参数
    market_params = {
        'trend_period': parsed.get('market_trend_period', MARKET_STATE_CONFIG.get('trend_period', 50)),
        'retracement_tolerance': parsed.get('market_retracement_tolerance', MARKET_STATE_CONFIG.get('retracement_tolerance', 0.30)),
        'volume_period': parsed.get('market_volume_period', MARKET_STATE_CONFIG.get('volume_period', 20)),
        'volume_ma_period': parsed.get('market_volume_ma_period', MARKET_STATE_CONFIG.get('volume_ma_period', 10)),
        'hourly_data_count': 100,
    }
    trend_weights = {
        'price_breakout': parsed.get('trend_price_breakout_weight', 0.35),
        'volume_confirmation': parsed.get('trend_volume_confirmation_weight', 0.25),
        'momentum oscillator': parsed.get('trend_momentum_oscillator_weight', 0.20),
        'moving_average': parsed.get('trend_moving_average_weight', 0.20),
    }
    trend_thresholds = {
        'strong_trend': parsed.get('trend_strong_threshold', 0.6),
        'weak_trend': parsed.get('trend_weak_threshold', 0.3),
        'volume_spike': parsed.get('trend_volume_spike', 1.5),
        'oversold': parsed.get('trend_oversold', 30),
        'overbought': parsed.get('trend_overbought', 70),
    }
    confidence_thresholds = {
        'high_confidence': parsed.get('confidence_high', 0.7),
        'medium_confidence': parsed.get('confidence_medium', 0.4),
    }

    # 5. 构建 MarketStateAnalyzer 并预计算H1状态
    analyzer = MarketStateAnalyzer(
        timeframe=PERIOD_H1,
        market_state_params=market_params,
        trend_weights=trend_weights,
        trend_thresholds=trend_thresholds,
        confidence_thresholds=confidence_thresholds,
    )
    analyzer._precomputed_states = _precompute_h1_states(multi_tf, analyzer)

    # 5.5. ★ 使用个体的策略参数重新实例化策略并重新计算信号
    # 将 class_name → {params} 映射为 config_key → {params}
    strategy_params_by_key = {
        _class_name_to_config_key.get(cn, cn.lower()): params
        for cn, params in strategy_params.items()
    }
    registry = StrategyRegistry()
    sig_instances = registry.instantiate_all(SYMBOL, TIMEFRAME, strategy_params_by_key)
    signals_df = pd.DataFrame(index=multi_tf.main_df.index)
    for config_key, strat in sig_instances.items():
        try:
            sig = strat.run_backtest(multi_tf.main_df)
            signals_df[config_key] = sig if sig is not None else pd.Series(0, index=signals_df.index)
        except Exception:
            signals_df[config_key] = pd.Series(0, index=signals_df.index)

    # 6. 回测环境
    data_provider = BacktestDataProvider(multi_tf, INITIAL_CAPITAL)
    pm = PositionManager(data_provider, trade_direction="both", risk_config=risk_params,
                         persist_peaks=False)

    buy_th = parsed.get('buy_threshold', 1.5)
    sell_th = parsed.get('sell_threshold', -1.5)
    total_bars = multi_tf.length

    # 7. ★ 逐 bar 回测（权重基因真正影响信号组合）
    for bar_index in range(total_bars):
        current_price = data_provider.get_current_price(SYMBOL)
        if not current_price:
            data_provider.tick()
            continue

        # 获取当前 bar 的市场状态
        state, conf = analyzer.get_market_state(bar_index)

        # ★ 权重基因在这里生效
        weights = analyzer.get_strategy_weights(
            state, conf, individual_weights=weight_genes
        )

        # 组合当前 bar 的信号（使用个体参数重新计算的信号）
        bar_signals = {
            col: signals_df[col].iloc[bar_index]
            for col in signals_df.columns
        }
        final_signal = SignalCombiner.combine_at_bar(bar_signals, weights, buy_th, sell_th)

        # 执行交易
        if final_signal == 1:
            pm.open_position("buy", current_price, 1.0, dry_run=True)
        elif final_signal == -1:
            pm.open_position("sell", current_price, 1.0, dry_run=True)

        pm.monitor_positions(current_price, dry_run=True)
        data_provider.tick()

    # 8. 适应度 = 总盈亏
    total_pnl = pm.total_equity - INITIAL_CAPITAL
    return (total_pnl,)


def load_historical_data():
    """一次性加载M1数据并返回 MultiTimeframeDataStore"""
    initialize()
    rates = (
        get_rates(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT, OPTIMIZER_START_DATE, OPTIMIZER_END_DATE)
        if USE_DATE_RANGE else
        get_rates(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT)
    )
    shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("获取历史数据失败")

    if len(rates) > OPTIMIZER_COUNT:
        rates = rates[-OPTIMIZER_COUNT:]

    store = MultiTimeframeDataStore()
    store.load_m1_data(rates)

    # 预生成H1数据
    store.ensure_timeframe(PERIOD_H1)

    return store


def precompute_all_signals(multi_tf: MultiTimeframeDataStore,
                          strategy_params: dict = None) -> pd.DataFrame:
    """使用默认参数预计算所有策略的回测信号"""
    registry = StrategyRegistry()
    strategies = registry.instantiate_all(SYMBOL, TIMEFRAME, strategy_params)
    signals_df = pd.DataFrame(index=multi_tf.main_df.index)

    for config_key, strategy in strategies.items():
        try:
            sig = strategy.run_backtest(multi_tf.main_df)
            signals_df[config_key] = sig if sig is not None else pd.Series(0, index=signals_df.index)
        except Exception:
            signals_df[config_key] = pd.Series(0, index=signals_df.index)

    return signals_df


def run_optimizer():
    """运行遗传算法优化（所有基因真正生效）"""
    try:
        # 1. 加载数据（一次性）
        logger.info("加载历史数据...")
        multi_tf = load_historical_data()
        logger.info(f"数据加载完成: {multi_tf.length} 条M1数据")

        # 2. DEAP 设置（策略信号在 evaluate_fitness 内部逐个体重新计算）
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        for i, param_def in enumerate(PARAMETER_DEFINITIONS):
            if param_def['type'] == 'int':
                toolbox.register(f"attr_param_{i}", random.randint, param_def['min'], param_def['max'])
            else:
                toolbox.register(f"attr_param_{i}", random.uniform, param_def['min'], param_def['max'])

        toolbox.register("individual", tools.initIterate, creator.Individual,
                         lambda: [toolbox.__getattribute__(f"attr_param_{j}")()
                                  for j in range(len(PARAMETER_DEFINITIONS))])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluate_fitness,
                        multi_tf=multi_tf)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian,
                        mu=GENETIC_OPTIMIZER_CONFIG["mutation_mu"],
                        sigma=GENETIC_OPTIMIZER_CONFIG["mutation_sigma"],
                        indpb=GENETIC_OPTIMIZER_CONFIG["mutation_indpb"])
        toolbox.register("select", tools.selTournament,
                        tournsize=GENETIC_OPTIMIZER_CONFIG["tournament_size"])

        # 多进程
        if GENETIC_OPTIMIZER_CONFIG["enable_multiprocessing"]:
            pool = multiprocessing.Pool(
                processes=GENETIC_OPTIMIZER_CONFIG["processes"],
                initializer=init_worker
            )
            toolbox.register("map", pool.map)
        else:
            toolbox.register("map", map)

        # 4. 统计
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # 5. 运行
        population = toolbox.population(n=GENETIC_OPTIMIZER_CONFIG["population_size"])
        ngen = GENETIC_OPTIMIZER_CONFIG["generations"]
        cxpb = GENETIC_OPTIMIZER_CONFIG["crossover_probability"]
        mutpb = GENETIC_OPTIMIZER_CONFIG["mutation_probability"]

        logger.info(f"开始进化: 种群 {len(population)}, {ngen} 代")
        generation_info = []

        for gen in range(ngen):
            offspring = toolbox.select(population, len(population))
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            fits = toolbox.map(toolbox.evaluate, offspring)

            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            population[:] = offspring

            best = tools.selBest(population, k=1)[0]
            best_fit = best.fitness.values[0]
            avg_fit = np.mean([ind.fitness.values[0] for ind in population])

            if GENETIC_OPTIMIZER_CONFIG.get("save_generation_info", True):
                generation_info.append({
                    'generation': gen + 1,
                    'best_fitness': best_fit,
                    'avg_fitness': avg_fit,
                })

            if GENETIC_OPTIMIZER_CONFIG.get("verbose", True):
                logger.info(f"代数 {gen + 1}/{ngen}: 最佳={best_fit:.2f}, 平均={avg_fit:.2f}")

        # 6. 结果
        if GENETIC_OPTIMIZER_CONFIG["enable_multiprocessing"]:
            pool.close()
            pool.join()

        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = best_individual.fitness.values[0]

        logger.info(f"优化完成: 最佳适应度={best_fitness:.2f}")
        save_optimization_results(best_individual, best_fitness, generation_info)

        return _parse_individual(best_individual), best_fitness

    except Exception as e:
        import traceback
        logger.error(f"优化器错误: {e}\n{traceback.format_exc()}")
        raise


def save_optimization_results(best_individual, best_fitness, generation_info):
    """保存优化结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parsed = _parse_individual(best_individual)

    # JSON
    import json
    results = {
        'best_fitness': best_fitness,
        'best_params': parsed,
        'generation_info': generation_info,
    }
    json_path = f"optimization_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"结果已保存: {json_path}")

    # TXT报告
    txt_path = f"optimization_report_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"优化完成时间: {datetime.now()}\n")
        f.write(f"最佳适应度: {best_fitness:.2f}\n\n")
        f.write("最佳参数:\n")
        for key, value in parsed.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n代数统计:\n")
        for gi in generation_info:
            f.write(f"  代数 {gi['generation']}: 最佳={gi['best_fitness']:.2f}, 平均={gi['avg_fitness']:.2f}\n")
    logger.info(f"报告已保存: {txt_path}")


if __name__ == "__main__":
    run_optimizer()
