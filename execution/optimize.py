"""贝叶斯优化器（Optuna TPE）
高效替代 DEAP 遗传算法，50次试验 ≈ 2分钟收敛

关键改进：
  1. TPE sampler 建模适应度曲面，采样效率高 3-5 倍
  2. 权重基因通过 individual_weights → MarketStateAnalyzer.get_strategy_weights() 真正生效
  3. evaluate_fitness 逐 bar 调用 get_market_state(bar_index) 获取每根bar的动态权重
"""

import random
import numpy as np
import pandas as pd
import sys
import os
import logging
from datetime import datetime

import optuna

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
    RISK_CONFIG,
    MARKET_STATE_CONFIG, TREND_INDICATOR_WEIGHTS, TREND_THRESHOLDS, CONFIDENCE_THRESHOLDS,
    DATA_PROVIDER_MODE, REMOTE_SERVER_HOST, REMOTE_SERVER_PORT
)
from utils.constants import PERIOD_H1
from core.utils import get_rates, initialize, shutdown
from core.data.remote import RemoteDataProvider
from logger import logger

# 导入策略模块确保 StrategyRegistry 已注册
import strategies  # noqa: F401

# ── 全局缓存 ──
_multi_tf: MultiTimeframeDataStore | None = None

# MT5 结构化数组 dtype（用于远程 API JSON → numpy 转换）
_MT5_RATES_DTYPE = np.dtype([
    ('time', 'i8'),
    ('open', 'f8'),
    ('high', 'f8'),
    ('low', 'f8'),
    ('close', 'f8'),
    ('tick_volume', 'i8'),
    ('spread', 'i4'),
    ('real_volume', 'i8'),
])


# ── 参数定义 ──
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
    # SwingPointRetestStrategy — 前高前低回踩
    {'name': 'swing_point_left_bars', 'type': 'int', 'min': 2, 'max': 5, 'strategy': 'SwingPointRetestStrategy'},
    {'name': 'swing_point_right_bars', 'type': 'int', 'min': 2, 'max': 5, 'strategy': 'SwingPointRetestStrategy'},
    {'name': 'swing_point_tolerance_pct', 'type': 'float', 'min': 0.0003, 'max': 0.002, 'strategy': 'SwingPointRetestStrategy'},
    {'name': 'swing_point_num_swings', 'type': 'int', 'min': 1, 'max': 3, 'strategy': 'SwingPointRetestStrategy'},
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
    # ★ Strategy Weights
    {'name': 'weight_MACrossStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_RSIStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_BollingerStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_MeanReversionStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_MomentumBreakoutStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_MACDStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_KDJStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
    {'name': 'weight_SwingPointRetestStrategy', 'type': 'float', 'min': 0.0, 'max': 2.0, 'strategy': 'weight'},
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

_class_name_to_config_key = {
    "MACrossStrategy": "ma_cross",
    "RSIStrategy": "rsi",
    "BollingerStrategy": "bollinger",
    "MeanReversionStrategy": "mean_reversion",
    "MomentumBreakoutStrategy": "momentum_breakout",
    "MACDStrategy": "macd",
    "KDJStrategy": "kdj",
    "SwingPointRetestStrategy": "swing_point",
    "DailyBreakoutStrategy": "daily_breakout",
    "WaveTheoryStrategy": "wave_theory",
}


def _trial_to_params(trial: optuna.Trial) -> dict:
    """从 Optuna trial 提取参数 → 兼容旧 _parse_individual 格式"""
    parsed = {}
    for param_def in PARAMETER_DEFINITIONS:
        name = param_def['name']
        if param_def['type'] == 'int':
            val = trial.suggest_int(name, param_def['min'], param_def['max'])
        else:
            val = trial.suggest_float(name, param_def['min'], param_def['max'])
        parsed[name] = val
    return parsed


def _extract_strategy_params(parsed: dict) -> dict:
    """从解析后的参数提取策略参数字典"""
    strategy_params = {}
    for param_def in PARAMETER_DEFINITIONS:
        strategy_name = param_def.get('strategy')
        if strategy_name in ('weight', 'signal', 'risk', 'market_state',
                             'trend_weights', 'trend_thresholds', 'confidence'):
            continue
        if strategy_name not in strategy_params:
            strategy_params[strategy_name] = {}

        param_name = param_def['name']
        if param_name == 'ma_cross_short_window': param_name = 'short_window'
        elif param_name == 'ma_cross_long_window': param_name = 'long_window'
        elif param_name == 'rsi_period': param_name = 'period'
        elif param_name == 'rsi_overbought': param_name = 'overbought'
        elif param_name == 'rsi_oversold': param_name = 'oversold'
        elif param_name == 'bollinger_period': param_name = 'period'
        elif param_name == 'bollinger_std_dev': param_name = 'std_dev'
        elif param_name == 'macd_fast_ema': param_name = 'fast_ema'
        elif param_name == 'macd_slow_ema': param_name = 'slow_ema'
        elif param_name == 'macd_signal_period': param_name = 'signal_period'
        elif param_name == 'mean_reversion_period': param_name = 'period'
        elif param_name == 'mean_reversion_std_dev': param_name = 'std_dev'
        elif param_name == 'momentum_breakout_period': param_name = 'period'
        elif param_name == 'momentum_breakout_momentum_period': param_name = 'momentum_period'
        elif param_name == 'kdj_period': param_name = 'period'
        elif param_name == 'swing_point_left_bars': param_name = 'left_bars'
        elif param_name == 'swing_point_right_bars': param_name = 'right_bars'
        elif param_name == 'swing_point_tolerance_pct': param_name = 'tolerance_pct'
        elif param_name == 'swing_point_num_swings': param_name = 'num_swings'
        elif param_name.startswith("wave_"):
            param_name = param_name[5:]
            if param_name == "period": param_name = "wave_period"
        elif param_name == 'daily_breakout_bars_count': param_name = 'bars_count'

        strategy_params[strategy_name][param_name] = parsed[param_def['name']]
    return strategy_params


def _extract_weight_genes(parsed: dict) -> dict:
    """提取权重基因 → {config_key: weight}"""
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

    h1_states = []
    for i in range(len(h1_df)):
        lookback = analyzer.trend_period + 10
        start = max(0, i - lookback)
        df_slice = h1_df.iloc[start:i + 1]
        state, conf = analyzer._calculate_state_from_df(df_slice)
        h1_states.append((state, conf))

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


def evaluate_fitness(trial: optuna.Trial, multi_tf: MultiTimeframeDataStore) -> float:
    """★ Optuna 适应度函数 — 所有权重/策略/趋势参数真正生效"""
    parsed = _trial_to_params(trial)

    # 1. 策略参数
    strategy_params = _extract_strategy_params(parsed)
    # 2. 权重基因
    weight_genes = _extract_weight_genes(parsed)
    # 3. 风控参数（从 config 读，不进优化器）
    risk_params = {
        'stop_loss_pct': RISK_CONFIG.get('stop_loss_pct', -0.50),
        'profit_retracement_pct': RISK_CONFIG.get('profit_retracement_pct', 0.35),
        'min_profit_for_trailing': RISK_CONFIG.get('min_profit_for_trailing', 0.70),
        'take_profit_pct': RISK_CONFIG.get('take_profit_pct', 1.00),
        'max_holding_minutes': RISK_CONFIG.get('max_holding_minutes', 0),
        'min_profit_for_time_exit': RISK_CONFIG.get('min_profit_for_time_exit', 0.005),
        'max_daily_loss': RISK_CONFIG.get('max_daily_loss', -0.30),
    }

    # 4. 市场状态和趋势参数
    market_params = {
        'trend_period': parsed.get('market_trend_period', 50),
        'retracement_tolerance': parsed.get('market_retracement_tolerance', 0.30),
        'volume_period': parsed.get('market_volume_period', 20),
        'volume_ma_period': parsed.get('market_volume_ma_period', 10),
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

    # 5. MarketStateAnalyzer
    analyzer = MarketStateAnalyzer(
        timeframe=PERIOD_H1,
        market_state_params=market_params,
        trend_weights=trend_weights,
        trend_thresholds=trend_thresholds,
        confidence_thresholds=confidence_thresholds,
    )
    analyzer._precomputed_states = _precompute_h1_states(multi_tf, analyzer)

    # 5.5 用个体策略参数重新计算信号
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

    # 7. 逐 bar 回测
    for bar_index in range(total_bars):
        current_price = data_provider.get_current_price(SYMBOL)
        if not current_price:
            data_provider.tick()
            continue

        state, conf = analyzer.get_market_state(bar_index)
        weights = analyzer.get_strategy_weights(state, conf, individual_weights=weight_genes)

        bar_signals = {col: signals_df[col].iloc[bar_index] for col in signals_df.columns}
        final_signal = SignalCombiner.combine_at_bar(bar_signals, weights, buy_th, sell_th)

        if final_signal == 1:
            pm.open_position("buy", current_price, 1.0, dry_run=True)
        elif final_signal == -1:
            pm.open_position("sell", current_price, 1.0, dry_run=True)

        pm.monitor_positions(current_price, dry_run=True)
        data_provider.tick()

    # 8. 适应度 = 总盈亏
    return pm.total_equity - INITIAL_CAPITAL


def _json_to_mt5_rates(rates_data):
    """远程API JSON → MT5 numpy 结构化数组"""
    if not rates_data:
        return None
    records = [(r['time'], r['open'], r['high'], r['low'], r['close'],
                r.get('tick_volume', 0), r.get('spread', 0), r.get('real_volume', 0))
               for r in rates_data]
    return np.array(records, dtype=_MT5_RATES_DTYPE)


def load_historical_data():
    """一次性加载M1数据并返回 MultiTimeframeDataStore"""
    if DATA_PROVIDER_MODE == "remote":
        provider = RemoteDataProvider(host=REMOTE_SERVER_HOST, port=REMOTE_SERVER_PORT)
        if not provider.initialize():
            raise RuntimeError("远程MT5 API初始化失败，请检查 Windows MT5 是否运行")
        rates_json = provider.get_historical_data(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT)
        provider.shutdown()
        if not rates_json:
            raise RuntimeError("远程获取历史数据失败")
        rates = _json_to_mt5_rates(rates_json)
    else:
        initialize()
        rates = (get_rates(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT, OPTIMIZER_START_DATE, OPTIMIZER_END_DATE)
                 if USE_DATE_RANGE else get_rates(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT))
        shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError("获取历史数据失败")
    if len(rates) > OPTIMIZER_COUNT:
        rates = rates[-OPTIMIZER_COUNT:]

    store = MultiTimeframeDataStore()
    store.load_m1_data(rates)
    store.ensure_timeframe(PERIOD_H1)
    return store


def save_optimization_results(best_params: dict, best_fitness: float, study: optuna.Study):
    """保存优化结果（兼容旧格式）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import json

    results = {
        'best_fitness': best_fitness,
        'best_params': best_params,
        'n_trials': len(study.trials),
        'optimizer': 'optuna_tpe',
    }
    json_path = f"optimization_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"结果已保存: {json_path}")

    txt_path = f"optimization_report_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"优化完成时间: {datetime.now()}\n")
        f.write(f"最佳适应度: {best_fitness:.2f}\n")
        f.write(f"试验次数: {len(study.trials)}\n\n")
        f.write("最佳参数:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
    logger.info(f"报告已保存: {txt_path}")


def run_optimizer():
    """★ Optuna TPE 贝叶斯优化（替代 DEAP 遗传算法）"""
    try:
        # 1. 加载数据
        logger.info("加载历史数据...")
        multi_tf = load_historical_data()
        logger.info(f"数据加载完成: {multi_tf.length} 条M1数据")

        # 2. Optuna study
        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # 3. 优化（4进程并行，静默回测日志避免 I/O 争用）
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        import logging as _logging
        _old_level = logger.level
        _old_handler_levels = [h.level for h in logger.handlers]
        logger.setLevel(_logging.WARNING)
        for h in logger.handlers:
            h.setLevel(_logging.WARNING)
        logger.info(f"开始贝叶斯优化: 50 次试验, TPE sampler (回测日志已静默)")
        try:
            study.optimize(
                lambda trial: evaluate_fitness(trial, multi_tf),
                n_trials=50,
                n_jobs=4,
                show_progress_bar=False,
            )
        finally:
            logger.setLevel(_old_level)
            for h, lvl in zip(logger.handlers, _old_handler_levels):
                h.setLevel(lvl)

        # 4. 结果
        best_fitness = study.best_value
        best_params = study.best_params
        logger.info(f"优化完成: 最佳适应度={best_fitness:.2f}, 试验次数={len(study.trials)}")

        save_optimization_results(best_params, best_fitness, study)

        return best_params, best_fitness

    except Exception as e:
        import traceback
        logger.error(f"优化器错误: {e}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    run_optimizer()
