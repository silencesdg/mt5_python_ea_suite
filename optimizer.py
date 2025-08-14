import random
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing
import pandas as pd
import sys
import os
from datetime import datetime
from tqdm import tqdm

# 固定随机种子，确保可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core components for backtesting
from core.data_providers import BacktestDataProvider
from core.risk import RiskController
from execution.dynamic_weights import DynamicWeightManager
from config import (
    SYMBOL, TIMEFRAME, OPTIMIZER_COUNT, OPTIMIZER_START_DATE, OPTIMIZER_END_DATE,
    USE_DATE_RANGE, BACKTEST_CONFIG, INITIAL_CAPITAL, SIGNAL_THRESHOLDS, DEFAULT_WEIGHTS,
    RISK_CONFIG, GENETIC_OPTIMIZER_CONFIG, MARKET_STATE_CONFIG, TREND_INDICATOR_WEIGHTS, 
    TREND_THRESHOLDS, CONFIDENCE_THRESHOLDS
)
from core.utils import get_rates, initialize, shutdown

# Import all strategy classes
from strategies.ma_cross import MACrossStrategy
from strategies.rsi import RSIStrategy
from strategies.bollinger import BollingerStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.macd import MACDStrategy
from strategies.kdj import KDJStrategy
from strategies.turtle import TurtleStrategy
from strategies.daily_breakout import DailyBreakoutStrategy
from strategies.wave_theory import WaveTheoryStrategy


# --- Global Data (Loaded once) ---
df_historical_data = None

# 模块级别的初始化函数，用于多进程
def init_worker():
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("StrategyLogger").setLevel(logging.WARNING)
    logging.getLogger("PositionManager").setLevel(logging.WARNING)
    logging.getLogger("RiskController").setLevel(logging.WARNING)
    logging.getLogger("DataProvider").setLevel(logging.WARNING)

def load_historical_data():
    global df_historical_data
    if df_historical_data is not None:
        return df_historical_data

    initialize()
    rates = (
        get_rates(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT, OPTIMIZER_START_DATE, OPTIMIZER_END_DATE)
        if USE_DATE_RANGE else
        get_rates(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT)
    )
    shutdown()

    if rates is None:
        print("获取历史数据失败，退出。")
        sys.exit(1)

    df = pd.DataFrame(rates)
    df.set_index(pd.to_datetime(df['time'], unit='s'), inplace=True)
    
    if len(df) > OPTIMIZER_COUNT:
        df = df.iloc[-OPTIMIZER_COUNT:]

    df_historical_data = df
    return df_historical_data


# --- Parameter Definition for DEAP Individual ---
# Define the search space for strategy parameters and weights
# Each entry: (name, type, min_val, max_val, strategy_class_name_if_param, default_val_if_weight)
# Type 'int' for integer parameters, 'float' for float parameters/weights
# Order matters for parsing the individual
PARAMETER_DEFINITIONS = [
    # MACrossStrategy parameters
    {'name': 'ma_cross_short_window', 'type': 'int', 'min': 3, 'max': 15, 'strategy': 'MACrossStrategy'},
    {'name': 'ma_cross_long_window',  'type': 'int', 'min': 10, 'max': 60, 'strategy': 'MACrossStrategy'},
    
    # RSIStrategy parameters
    {'name': 'rsi_period',            'type': 'int', 'min': 7, 'max': 21, 'strategy': 'RSIStrategy'},
    {'name': 'rsi_overbought',        'type': 'int', 'min': 65, 'max': 80, 'strategy': 'RSIStrategy'},
    {'name': 'rsi_oversold',          'type': 'int', 'min': 20, 'max': 35, 'strategy': 'RSIStrategy'},

    # BollingerStrategy parameters
    {'name': 'bollinger_period',      'type': 'int', 'min': 10, 'max': 50, 'strategy': 'BollingerStrategy'},
    {'name': 'bollinger_std_dev',     'type': 'float', 'min': 1.5, 'max': 3.0, 'strategy': 'BollingerStrategy'},

    # MACDStrategy parameters
    {'name': 'macd_fast_ema',         'type': 'int', 'min': 8, 'max': 20, 'strategy': 'MACDStrategy'},
    {'name': 'macd_slow_ema',         'type': 'int', 'min': 20, 'max': 35, 'strategy': 'MACDStrategy'},
    {'name': 'macd_signal_period',    'type': 'int', 'min': 5, 'max': 15, 'strategy': 'MACDStrategy'},

    # MeanReversionStrategy parameters
    {'name': 'mean_reversion_period', 'type': 'int', 'min': 10, 'max': 50, 'strategy': 'MeanReversionStrategy'},
    {'name': 'mean_reversion_std_dev','type': 'float', 'min': 1.5, 'max': 3.0, 'strategy': 'MeanReversionStrategy'},

    # MomentumBreakoutStrategy parameters
    {'name': 'momentum_breakout_period', 'type': 'int', 'min': 10, 'max': 50, 'strategy': 'MomentumBreakoutStrategy'},

    # KDJStrategy parameters
    {'name': 'kdj_period',            'type': 'int', 'min': 5, 'max': 21, 'strategy': 'KDJStrategy'},

    # TurtleStrategy parameters
    {'name': 'turtle_period',         'type': 'int', 'min': 10, 'max': 50, 'strategy': 'TurtleStrategy'},

    # WaveTheoryStrategy parameters
    {'name': 'wave_ema_short',        'type': 'int', 'min': 3, 'max': 10, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_ema_medium',       'type': 'int', 'min': 8, 'max': 20, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_ema_long',         'type': 'int', 'min': 20, 'max': 50, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_period',           'type': 'int', 'min': 10, 'max': 40, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_range_period',     'type': 'int', 'min': 10, 'max': 40, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_adx_period',       'type': 'int', 'min': 10, 'max': 25, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_momentum_period',  'type': 'int', 'min': 7, 'max': 21, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_range_threshold',  'type': 'float', 'min': 0.002, 'max': 0.01, 'strategy': 'WaveTheoryStrategy'},
    {'name': 'wave_adx_threshold',    'type': 'int', 'min': 15, 'max': 35, 'strategy': 'WaveTheoryStrategy'},

    # DailyBreakoutStrategy parameters
    {'name': 'daily_breakout_bars_count', 'type': 'int', 'min': 720, 'max': 2880, 'strategy': 'DailyBreakoutStrategy'},

    # Signal Thresholds
    {'name': 'buy_threshold',         'type': 'float', 'min': 0.5, 'max': 3.0, 'strategy': 'signal'},
    {'name': 'sell_threshold',        'type': 'float', 'min': -3.0, 'max': -0.5, 'strategy': 'signal'},

    # Risk Management parameters
    {'name': 'stop_loss_pct',         'type': 'float', 'min': -0.02, 'max': -0.005, 'strategy': 'risk'},
    {'name': 'profit_retracement_pct','type': 'float', 'min': 0.05, 'max': 0.20, 'strategy': 'risk'},
    {'name': 'min_profit_for_trailing','type': 'float', 'min': 0.005, 'max': 0.02, 'strategy': 'risk'},
    {'name': 'take_profit_pct',       'type': 'float', 'min': 0.10, 'max': 0.50, 'strategy': 'risk'},
    {'name': 'max_holding_minutes',   'type': 'int', 'min': 30, 'max': 180, 'strategy': 'risk'},
    {'name': 'min_profit_for_time_exit','type': 'float', 'min': 0.002, 'max': 0.01, 'strategy': 'risk'},

    # Strategy Weights (for all strategies)
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

# Map strategy class names to config keys for weights (for logging/display)
strategy_class_name_to_config_key = {
    "MACrossStrategy": "ma_cross",
    "RSIStrategy": "rsi",
    "BollingerStrategy": "bollinger",
    "MeanReversionStrategy": "mean_reversion",
    "MomentumBreakoutStrategy": "momentum_breakout",
    "MACDStrategy": "macd",
    "KDJStrategy": "kdj",
    "TurtleStrategy": "turtle",
    "DailyBreakoutStrategy": "daily_breakout",
    "WaveTheoryStrategy": "wave_theory"
}

# Map config keys to strategy class names (for instantiation)
strategy_config_key_to_class = {
    "ma_cross": MACrossStrategy,
    "rsi": RSIStrategy,
    "bollinger": BollingerStrategy,
    "mean_reversion": MeanReversionStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "macd": MACDStrategy,
    "kdj": KDJStrategy,
    "turtle": TurtleStrategy,
    "daily_breakout": DailyBreakoutStrategy,
    "wave_theory": WaveTheoryStrategy
}


def evaluate_fitness(individual, df_data):
    parsed_params = {}
    for idx, param_def in enumerate(PARAMETER_DEFINITIONS):
        val = individual[idx]
        parsed_params[param_def['name']] = int(round(val)) if param_def['type'] == 'int' else val

    # 构建策略参数字典
    strategy_params_dict = {}
    for param_def in PARAMETER_DEFINITIONS:
        strategy_name = param_def.get('strategy')
        if strategy_name and strategy_name != 'weight' and strategy_name != 'signal' and strategy_name != 'risk':
            if strategy_name not in strategy_params_dict:
                strategy_params_dict[strategy_name] = {}
            # 移除策略前缀并映射到正确的参数名
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
            elif param_name == 'kdj_period':
                param_name = 'period'
            elif param_name == 'turtle_period':
                param_name = 'period'
            elif param_name.startswith("wave_"):
                param_name = param_name[5:]  # 移除 "wave_" 前缀
                # 波浪理论策略特殊处理
                if param_name == "period":
                    param_name = "wave_period"
            elif param_name == 'daily_breakout_bars_count':
                param_name = 'bars_count'
            
            strategy_params_dict[strategy_name][param_name] = parsed_params[param_def['name']]

    # 策略实例化
    strategies_for_backtest_instance = {}
    for config_key, strategy_class in strategy_config_key_to_class.items():
        strategy_class_name = strategy_class.__name__
        strat_params = strategy_params_dict.get(strategy_class_name, {})
        
        # 实例化策略
        strategies_for_backtest_instance[config_key] = strategy_class(None, SYMBOL, TIMEFRAME, **strat_params)

    # 获取策略权重
    suggested_weights = {
        p['name'].replace('weight_', ''): parsed_params[p['name']]
        for p in PARAMETER_DEFINITIONS if p.get('strategy') == 'weight'
    }

    # 获取信号阈值
    signal_thresholds = {
        'buy_threshold': parsed_params.get('buy_threshold', SIGNAL_THRESHOLDS.get('buy_threshold', 1.0)),
        'sell_threshold': parsed_params.get('sell_threshold', SIGNAL_THRESHOLDS.get('sell_threshold', -1.0))
    }

    # 获取风险管理参数
    risk_params = {
        'stop_loss_pct': parsed_params.get('stop_loss_pct', RISK_CONFIG.get('stop_loss_pct', -0.01)),
        'profit_retracement_pct': parsed_params.get('profit_retracement_pct', .get('profit_retracement_pct', 0.10)),
        'min_profit_for_trailing': parsed_params.get('min_profit_for_trailing', RISK_CONFIG.get('min_profit_for_trailing', 0.01)),
        'take_profit_pct': parsed_params.get('take_profit_pct', RISK_CONFIG.get('take_profit_pct', 0.20)),
        'max_holding_minutes': parsed_params.get('max_holding_minutes', RISK_CONFIG.get('max_holding_minutes', 60)),
        'min_profit_for_time_exit': parsed_params.get('min_profit_for_time_exit', RISK_CONFIG.get('min_profit_for_time_exit', 0.005))
    }

    # 获取市场状态分析参数
    market_state_params = {
        'trend_period': parsed_params.get('market_trend_period', MARKET_STATE_CONFIG.get('trend_period', 50)),
        'retracement_tolerance': parsed_params.get('market_retracement_tolerance', MARKET_STATE_CONFIG.get('retracement_tolerance', 0.30)),
        'volume_period': parsed_params.get('market_volume_period', MARKET_STATE_CONFIG.get('volume_period', 20)),
        'volume_ma_period': parsed_params.get('market_volume_ma_period', MARKET_STATE_CONFIG.get('volume_ma_period', 10))
    }

    # 获取趋势指标权重
    trend_weights = {
        'price_breakout': parsed_params.get('trend_price_breakout_weight', TREND_INDICATOR_WEIGHTS.get('price_breakout', 0.35)),
        'volume_confirmation': parsed_params.get('trend_volume_confirmation_weight', TREND_INDICATOR_WEIGHTS.get('volume_confirmation', 0.25)),
        'momentum oscillator': parsed_params.get('trend_momentum_oscillator_weight', TREND_INDICATOR_WEIGHTS.get('momentum oscillator', 0.20)),
        'moving_average': parsed_params.get('trend_moving_average_weight', TREND_INDICATOR_WEIGHTS.get('moving_average', 0.20))
    }

    # 获取趋势阈值
    trend_thresholds = {
        'strong_trend': parsed_params.get('trend_strong_threshold', TREND_THRESHOLDS.get('strong_trend', 0.6)),
        'weak_trend': parsed_params.get('trend_weak_threshold', TREND_THRESHOLDS.get('weak_trend', 0.3)),
        'volume_spike': parsed_params.get('trend_volume_spike', TREND_THRESHOLDS.get('volume_spike', 1.5)),
        'oversold': parsed_params.get('trend_oversold', TREND_THRESHOLDS.get('oversold', 30)),
        'overbought': parsed_params.get('trend_overbought', TREND_THRESHOLDS.get('overbought', 70))
    }

    # 获取置信度阈值
    confidence_thresholds = {
        'high_confidence': parsed_params.get('confidence_high', CONFIDENCE_THRESHOLDS.get('high_confidence', 0.7)),
        'medium_confidence': parsed_params.get('confidence_medium', CONFIDENCE_THRESHOLDS.get('medium_confidence', 0.4))
    }

    df = df_data.copy()
    data_provider = BacktestDataProvider(df, initial_equity=INITIAL_CAPITAL)
    
    # 创建风险控制器并更新参数
    risk_controller = RiskController(data_provider, trade_direction=BACKTEST_CONFIG['trade_direction'])
    
    # 更新风险控制器参数
    for key, value in risk_params.items():
        if hasattr(risk_controller, key):
            setattr(risk_controller, key, value)
    
    # 创建动态权重管理器并传入优化参数
    dynamic_weight_manager = DynamicWeightManager(
        data_provider,
        market_state_params=market_state_params,
        trend_weights=trend_weights,
        trend_thresholds=trend_thresholds,
        confidence_thresholds=confidence_thresholds
    )

    all_signals_df = pd.DataFrame(index=df.index)
    for config_key, strat_instance in strategies_for_backtest_instance.items():
        if hasattr(strat_instance, 'run_backtest') and callable(getattr(strat_instance, 'run_backtest')):
            all_signals_df[strat_instance.name] = strat_instance.run_backtest(df.copy())
    
    if all_signals_df.empty:
        return (0,)

    # 使用动态权重管理器获取策略权重
    strategies_with_weights = dynamic_weight_manager.get_current_strategies_and_weights()
    
    # numpy加权计算
    weights_arr = []
    for name, sig in all_signals_df.items():
        # 从动态权重管理器获取权重，如果没有则使用建议权重
        w = 1.0
        for strat_instance, weight in strategies_with_weights:
            if strat_instance.name == name:
                w = weight
                break
        # 如果动态权重中没有找到，使用建议权重
        if w == 1.0:
            w = suggested_weights.get(name, 1.0)
        weights_arr.append(sig.values * w)

    combined = np.sum(np.column_stack(weights_arr), axis=1)
    buy_th, sell_th = signal_thresholds['buy_threshold'], signal_thresholds['sell_threshold']
    final_signals = np.where(combined > buy_th, 1, np.where(combined < sell_th, -1, 0))

    # 回测循环逻辑
    data_provider.current_index = 0
    for i in range(len(df)):
        current_price = data_provider.get_current_price(SYMBOL)
        if not current_price:
            continue
        
        current_signal = final_signals[i]

        direction = None
        if current_signal == 1:
            direction = "buy"
        elif current_signal == -1:
            direction = "sell"
        
        if direction:
            risk_controller.process_trading_signal(direction, current_price, abs(current_signal))

        risk_controller.monitor_positions(current_price)
        data_provider.tick()

    total_profit_loss = risk_controller.position_manager.get_trade_summary().get('total_profit_loss', 0)
    return (total_profit_loss,)


# --- Main Optimization Run ---
def run_optimizer():
    try:
        # Load data once
        df_historical_data = load_historical_data()

        # DEAP setup
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # Register attribute generators for each parameter
        for i, param_def in enumerate(PARAMETER_DEFINITIONS):
            if param_def['type'] == 'int':
                toolbox.register(f"attr_param_{i}", random.randint, param_def['min'], param_def['max'])
            else: # float
                toolbox.register(f"attr_param_{i}", random.uniform, param_def['min'], param_def['max'])
        
        # Register individual creator
        toolbox.register("individual", tools.initIterate, creator.Individual, 
                         lambda: [toolbox.__getattribute__(f"attr_param_{j}")() for j in range(len(PARAMETER_DEFINITIONS))])
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evaluate_fitness, df_data=df_historical_data)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, 
                        mu=GENETIC_OPTIMIZER_CONFIG["mutation_mu"],
                        sigma=GENETIC_OPTIMIZER_CONFIG["mutation_sigma"], 
                        indpb=GENETIC_OPTIMIZER_CONFIG["mutation_indpb"])
        toolbox.register("select", tools.selTournament, 
                        tournsize=GENETIC_OPTIMIZER_CONFIG["tournament_size"])

        # 多进程处理
        if GENETIC_OPTIMIZER_CONFIG["enable_multiprocessing"]:
            pool = multiprocessing.Pool(processes=GENETIC_OPTIMIZER_CONFIG["processes"], 
                                       initializer=init_worker)
            toolbox.register("map", pool.map)
        else:
            # 不使用多进程时，使用普通的map函数
            toolbox.register("map", map)

        # Statistics setup
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Run algorithm - 从配置获取参数
        population = toolbox.population(n=GENETIC_OPTIMIZER_CONFIG["population_size"])
        ngen = GENETIC_OPTIMIZER_CONFIG["generations"]
        cxpb = GENETIC_OPTIMIZER_CONFIG["crossover_probability"]
        mutpb = GENETIC_OPTIMIZER_CONFIG["mutation_probability"]

        if GENETIC_OPTIMIZER_CONFIG["verbose"]:
            print(f"开始多进程进化... (种群大小: {len(population)}, 进化代数: {ngen})")
        
        # 自定义统计回调以打印代数信息
        generation_info = []
        def generation_callback(gen, population):
            best_gen_individual = tools.selBest(population, k=1)[0]
            best_gen_fitness = best_gen_individual.fitness.values[0]
            avg_gen_fitness = sum(ind.fitness.values[0] for ind in population) / len(population)
            
            # 解析最佳个体的参数
            parsed_params = {}
            for idx, param_def in enumerate(PARAMETER_DEFINITIONS):
                param_value = best_gen_individual[idx]
                if param_def['type'] == 'int':
                    parsed_params[param_def['name']] = int(round(param_value))
                else:
                    parsed_params[param_def['name']] = param_value
            
            # 根据配置决定是否保存代数信息
            if GENETIC_OPTIMIZER_CONFIG["save_generation_info"]:
                generation_info.append({
                    'generation': gen,
                    'best_fitness': best_gen_fitness,
                    'avg_fitness': avg_gen_fitness,
                    'best_params': parsed_params.copy()
                })
            
            # 根据配置决定是否输出详细信息
            if GENETIC_OPTIMIZER_CONFIG["verbose"]:
                print(f"\n=== 代数 {gen} ===")
                print(f"最佳适应度 = {best_gen_fitness:.2f}, 平均适应度 = {avg_gen_fitness:.2f}")
                print("当前最佳参数组合:")
                for key, value in parsed_params.items():
                    print(f"  {key}: {value}")
        
        # 使用自定义进化循环
        for gen in range(ngen):
            offspring = toolbox.select(population, len(population))
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            fits = toolbox.map(toolbox.evaluate, offspring)
            
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            
            population[:] = offspring
            
            # 调用代数信息回调
            generation_callback(gen + 1, population)

        # 关闭进程池（如果使用了多进程）
        if GENETIC_OPTIMIZER_CONFIG["enable_multiprocessing"]:
            pool.close()
            pool.join() # Wait for all processes to finish

        # Results
        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = best_individual.fitness.values[0]

        print("--- DEAP 优化结束 ---")
        print(f"最佳适应度 (总盈亏): {best_fitness:.2f}")
        print("最佳参数组合:")
        
        # 打印详细的代数信息
        print("\n=== 代数统计信息 ===")
        for gen_info in generation_info:
            print(f"代数 {gen_info['generation']}: 最佳={gen_info['best_fitness']:.2f}, 平均={gen_info['avg_fitness']:.2f}")
        
        print("\n=== 运行结果总结 ===")
        print(f"总进化代数: {ngen}")
        print(f"种群大小: {len(population)}")
        print(f"交叉概率: {cxpb}")
        print(f"变异概率: {mutpb}")
        print(f"最终最佳适应度: {best_fitness:.2f}")
        
        # 解析最佳参数组合
        parsed_best_params = {}
        current_idx = 0
        for param_def in PARAMETER_DEFINITIONS:
            param_value = best_individual[current_idx]
            if param_def['type'] == 'int':
                parsed_best_params[param_def['name']] = int(round(param_value))
            else:
                parsed_best_params[param_def['name']] = param_value
            current_idx += 1

        print("\n=== 最终最佳参数组合 ===")
        # 分组显示参数：策略参数和权重
        print("\n策略参数:")
        for param_def in PARAMETER_DEFINITIONS:
            if param_def.get('strategy') != 'weight':
                print(f"  {param_def['name']}: {parsed_best_params[param_def['name']]}")
        
        print("\n策略权重:")
        for param_def in PARAMETER_DEFINITIONS:
            if param_def.get('strategy') == 'weight':
                strategy_name = param_def['name'].replace('weight_', '')
                print(f"  {strategy_name}: {parsed_best_params[param_def['name']]:.3f}")
        
        # 显示权重总和
        weight_sum = sum(parsed_best_params[p['name']] for p in PARAMETER_DEFINITIONS if p.get('strategy') == 'weight')
        print(f"\n权重总和: {weight_sum:.3f}")
        
        # 显示历史最佳参数变化
        print("\n=== 历史最佳参数变化 ===")
        for i, gen_info in enumerate(generation_info):
            if i % 5 == 0 or i == len(generation_info) - 1:  # 每5代显示一次，包括最后一代
                print(f"代数 {gen_info['generation']}: 适应度={gen_info['best_fitness']:.2f}")
                # 只显示权重信息
                weights = {k: v for k, v in gen_info['best_params'].items() if k.startswith('weight_')}
                for weight_key, weight_value in weights.items():
                    strategy_name = weight_key.replace('weight_', '')
                    print(f"    {strategy_name}: {weight_value:.3f}")
                print()

        # 保存优化结果到文件
        save_optimization_results(parsed_best_params, best_fitness, generation_info)
        
        return parsed_best_params, best_fitness

    except Exception as e:
        print(f"优化器运行出错: {e}")
        import traceback
        print(traceback.format_exc())
        
        # Ensure pool is terminated on error, only if it was created
        if 'pool' in locals() and pool is not None:
            pool.terminate()
        sys.exit(1)


def save_optimization_results(best_params, best_fitness, generation_info):
    """保存优化结果到文件"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建结果数据结构
        results = {
            'optimization_info': {
                'timestamp': timestamp,
                'best_fitness': float(best_fitness),
                'total_generations': len(generation_info),
                'population_size': GENETIC_OPTIMIZER_CONFIG["population_size"],
                'crossover_probability': GENETIC_OPTIMIZER_CONFIG["crossover_probability"],
                'mutation_probability': GENETIC_OPTIMIZER_CONFIG["mutation_probability"],
                'random_seed': SEED
            },
            'best_parameters': best_params,
            'generation_history': generation_info
        }
        
        # 保存为JSON文件
        json_filename = f"optimization_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n优化结果已保存到: {json_filename}")
        
        # 保存为CSV文件（只保存最佳参数）
        csv_filename = f"optimization_best_params_{timestamp}.csv"
        import csv
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value', 'Type'])
            
            for param_def in PARAMETER_DEFINITIONS:
                param_name = param_def['name']
                param_value = best_params[param_name]
                param_type = param_def['type']
                
                # 添加策略分类
                strategy_type = param_def.get('strategy', 'unknown')
                writer.writerow([param_name, param_value, f"{param_type}_{strategy_type}"])
        
        print(f"最佳参数已保存到: {csv_filename}")
        
        # 保存代数历史为CSV
        history_csv_filename = f"optimization_history_{timestamp}.csv"
        with open(history_csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Generation', 'Best_Fitness', 'Avg_Fitness'] + 
                           [p['name'] for p in PARAMETER_DEFINITIONS])
            
            for gen_info in generation_info:
                row = [
                    gen_info['generation'],
                    gen_info['best_fitness'],
                    gen_info['avg_fitness']
                ]
                # 添加所有参数值
                for param_def in PARAMETER_DEFINITIONS:
                    row.append(gen_info['best_params'].get(param_def['name'], ''))
                writer.writerow(row)
        
        print(f"代数历史已保存到: {history_csv_filename}")
        
        # 生成可读的文本报告
        txt_filename = f"optimization_report_{timestamp}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("遗传算法优化结果报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最佳适应度 (总盈亏): ${best_fitness:.2f}\n")
            f.write(f"总进化代数: {len(generation_info)}\n")
            f.write(f"种群大小: 50\n")
            f.write(f"随机种子: {SEED}\n")
            f.write("\n")
            
            f.write("最佳参数组合:\n")
            f.write("-" * 40 + "\n")
            
            # 分组显示参数
            f.write("\n【策略参数】\n")
            for param_def in PARAMETER_DEFINITIONS:
                if param_def.get('strategy') not in ['weight', 'signal', 'risk']:
                    param_name = param_def['name']
                    param_value = best_params[param_name]
                    f.write(f"  {param_name}: {param_value}\n")
            
            f.write("\n【信号阈值】\n")
            for param_def in PARAMETER_DEFINITIONS:
                if param_def.get('strategy') == 'signal':
                    param_name = param_def['name']
                    param_value = best_params[param_name]
                    f.write(f"  {param_name}: {param_value}\n")
            
            f.write("\n【风险管理参数】\n")
            for param_def in PARAMETER_DEFINITIONS:
                if param_def.get('strategy') == 'risk':
                    param_name = param_def['name']
                    param_value = best_params[param_name]
                    f.write(f"  {param_name}: {param_value}\n")
            
            f.write("\n【策略权重】\n")
            weight_sum = 0
            for param_def in PARAMETER_DEFINITIONS:
                if param_def.get('strategy') == 'weight':
                    param_name = param_def['name']
                    param_value = best_params[param_name]
                    strategy_name = param_name.replace('weight_', '')
                    weight_sum += param_value
                    f.write(f"  {strategy_name}: {param_value:.3f}\n")
            f.write(f"  权重总和: {weight_sum:.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("代数进化历史\n")
            f.write("=" * 80 + "\n")
            
            for i, gen_info in enumerate(generation_info):
                if i % 5 == 0 or i == len(generation_info) - 1:
                    f.write(f"\n代数 {gen_info['generation']}:\n")
                    f.write(f"  最佳适应度: ${gen_info['best_fitness']:.2f}\n")
                    f.write(f"  平均适应度: ${gen_info['avg_fitness']:.2f}\n")
        
        print(f"详细报告已保存到: {txt_filename}")
        
    except Exception as e:
        print(f"保存优化结果时出错: {e}")
        import traceback
        print(traceback.format_exc())


if __name__ == "__main__":
    run_optimizer()