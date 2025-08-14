"""
机器学习优化器 (基于随机探索 + 模型预测)
替代遗传算法，使用三阶段优化策略：
1. 随机探索阶段
2. 模型训练阶段  
3. 预测筛选阶段
"""
import random
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import csv
import multiprocessing
from joblib import Parallel, delayed

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

# --- Parameter Definition (Same as GA optimizer) ---
# Define the search space for strategy parameters and weights
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

def load_historical_data():
    """加载历史数据（与原优化器相同）"""
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

def generate_random_parameters():
    """生成随机参数组合"""
    params = []
    for param_def in PARAMETER_DEFINITIONS:
        if param_def['type'] == 'int':
            value = random.randint(param_def['min'], param_def['max'])
        else:  # float
            value = random.uniform(param_def['min'], param_def['max'])
        params.append(value)
    return params

def evaluate_parameters(params_array, df_data):
    """评估参数组合的适应度（与原优化器相同）"""
    parsed_params = {}
    for idx, param_def in enumerate(PARAMETER_DEFINITIONS):
        val = params_array[idx]
        parsed_params[param_def['name']] = int(round(val)) if param_def['type'] == 'int' else val

    # 构建策略参数字典（与原优化器相同）
    strategy_params_dict = {}
    for param_def in PARAMETER_DEFINITIONS:
        strategy_name = param_def.get('strategy')
        if strategy_name and strategy_name not in ['weight', 'signal', 'risk', 'market_state', 'trend_weights', 'trend_thresholds', 'confidence']:
            if strategy_name not in strategy_params_dict:
                strategy_params_dict[strategy_name] = {}
            
            param_name = param_def['name']
            # 参数名映射（与原优化器相同）
            if param_name == 'ma_cross_short_window':
                param_name = 'short_window'
            elif param_name == 'ma_cross_long_window':
                param_name = 'long_window'
            elif param_name.endswith('_period') and 'ma_cross' not in param_name:
                param_name = 'period'
            elif param_name == 'bollinger_std_dev':
                param_name = 'std_dev'
            elif param_name == 'mean_reversion_std_dev':
                param_name = 'std_dev'
            elif param_name.startswith('macd_'):
                param_name = param_name[5:]  # 移除 "macd_" 前缀
            elif param_name.startswith('wave_'):
                param_name = param_name[5:]  # 移除 "wave_" 前缀
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
        'profit_retracement_pct': parsed_params.get('profit_retracement_pct', RISK_CONFIG.get('profit_retracement_pct', 0.10)),
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

    # 回测执行（与原优化器相同）
    df = df_data.copy()
    data_provider = BacktestDataProvider(df, initial_equity=INITIAL_CAPITAL)
    
    risk_controller = RiskController(data_provider, trade_direction=BACKTEST_CONFIG['trade_direction'])
    
    for key, value in risk_params.items():
        if hasattr(risk_controller, key):
            setattr(risk_controller, key, value)
    
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
        return 0.0

    strategies_with_weights = dynamic_weight_manager.get_current_strategies_and_weights()
    
    weights_arr = []
    for name, sig in all_signals_df.items():
        w = 1.0
        for strat_instance, weight in strategies_with_weights:
            if strat_instance.name == name:
                w = weight
                break
        if w == 1.0:
            w = suggested_weights.get(name, 1.0)
        weights_arr.append(sig.values * w)

    combined = np.sum(np.column_stack(weights_arr), axis=1)
    buy_th, sell_th = signal_thresholds['buy_threshold'], signal_thresholds['sell_threshold']
    final_signals = np.where(combined > buy_th, 1, np.where(combined < sell_th, -1, 0))

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
    return total_profit_loss

def phase1_exploration(df_data, exploration_size=100):
    """第一阶段：随机探索"""
    print(f"=== 第一阶段：随机探索 ===")
    print(f"生成 {exploration_size} 组随机参数进行探索...")
    
    exploration_data = []
    
    # 使用多进程并行评估
    num_cores = multiprocessing.cpu_count()
    print(f"使用 {num_cores} 个核心进行并行评估...")
    
    # 生成所有随机参数
    all_params = [generate_random_parameters() for _ in range(exploration_size)]
    
    # 并行评估
    results = Parallel(n_jobs=num_cores)(
        delayed(evaluate_parameters)(params, df_data) 
        for params in tqdm(all_params, desc="随机探索评估")
    )
    
    # 收集结果
    for params, profit in zip(all_params, results):
        exploration_data.append({
            'params': params.copy(),
            'profit': profit
        })
    
    # 按盈利排序
    exploration_data.sort(key=lambda x: x['profit'], reverse=True)
    
    print(f"随机探索完成，最佳盈利: ${exploration_data[0]['profit']:.2f}")
    print(f"最差盈利: ${exploration_data[-1]['profit']:.2f}")
    print(f"平均盈利: ${np.mean([d['profit'] for d in exploration_data]):.2f}")
    
    return exploration_data

def phase2_model_training(exploration_data):
    """第二阶段：模型训练"""
    print(f"\n=== 第二阶段：模型训练 ===")
    
    # 准备训练数据
    X = np.array([d['params'] for d in exploration_data])
    y = np.array([d['profit'] for d in exploration_data])
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    print(f"训练数据: {len(X_train)} 组, 测试数据: {len(X_test)} 组")
    
    # 训练随机森林模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 评估模型性能
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    print(f"模型训练完成:")
    print(f"训练集 R²: {train_r2:.4f}, MSE: {train_mse:.4f}")
    print(f"测试集 R²: {test_r2:.4f}, MSE: {test_mse:.4f}")
    
    # 特征重要性分析
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]  # 最重要的10个特征
    
    print(f"\n最重要的10个参数:")
    for idx in top_features:
        param_def = PARAMETER_DEFINITIONS[idx]
        print(f"  {param_def['name']}: {feature_importance[idx]:.4f}")
    
    return model

def phase3_prediction_and_validation(model, df_data, exploration_data, prediction_size=1000, top_n=20):
    """第三阶段：预测筛选和验证"""
    print(f"\n=== 第三阶段：预测筛选 ===")
    print(f"生成 {prediction_size} 组参数进行预测，筛选前 {top_n} 组进行验证...")
    
    # 生成大量随机参数进行预测
    prediction_params = [generate_random_parameters() for _ in range(prediction_size)]
    
    # 使用模型预测盈利
    predicted_profits = model.predict(prediction_params)
    
    # 按预测盈利排序，选择前top_n组
    top_indices = np.argsort(predicted_profits)[-top_n:][::-1]
    
    print(f"预测筛选完成，预测最高盈利: ${predicted_profits[top_indices[0]]:.2f}")
    
    # 对筛选出的参数进行实际验证
    validation_results = []
    
    print(f"开始验证前 {top_n} 组参数...")
    for i, idx in enumerate(tqdm(top_indices, desc="验证参数")):
        params = prediction_params[idx]
        actual_profit = evaluate_parameters(params, df_data)
        
        validation_results.append({
            'params': params.copy(),
            'predicted_profit': predicted_profits[idx],
            'actual_profit': actual_profit,
            'prediction_error': abs(predicted_profits[idx] - actual_profit)
        })
        
        if (i + 1) % 5 == 0:
            print(f"已验证 {i + 1}/{top_n} 组，当前最佳实际盈利: ${max([r['actual_profit'] for r in validation_results[:i+1]]):.2f}")
    
    # 按实际盈利排序
    validation_results.sort(key=lambda x: x['actual_profit'], reverse=True)
    
    best_validation = validation_results[0]
    print(f"\n验证完成:")
    print(f"最佳实际盈利: ${best_validation['actual_profit']:.2f}")
    print(f"预测盈利: ${best_validation['predicted_profit']:.2f}")
    print(f"预测误差: ${best_validation['prediction_error']:.2f}")
    
    return validation_results

def run_ml_optimizer():
    """运行机器学习优化器"""
    try:
        # ML优化器配置
        EXPLORATION_SIZE = 100    # 第一阶段随机探索数量
        PREDICTION_SIZE = 1000    # 第二阶段预测数量
        VALIDATION_TOP_N = 20     # 第三阶段验证数量
        
        print("=" * 60)
        print("机器学习优化器启动")
        print("=" * 60)
        print(f"第一阶段: 随机探索 {EXPLORATION_SIZE} 组参数")
        print(f"第二阶段: 训练预测模型")
        print(f"第三阶段: 预测筛选 {PREDICTION_SIZE} 组参数，验证前 {VALIDATION_TOP_N} 组")
        
        # 加载历史数据
        df_historical_data = load_historical_data()
        print(f"历史数据加载完成，数据量: {len(df_historical_data)} 条")
        
        # 第一阶段：随机探索
        exploration_data = phase1_exploration(df_historical_data, EXPLORATION_SIZE)
        
        # 第二阶段：模型训练
        model = phase2_model_training(exploration_data)
        
        # 第三阶段：预测筛选和验证
        validation_results = phase3_prediction_and_validation(
            model, df_historical_data, exploration_data, PREDICTION_SIZE, VALIDATION_TOP_N
        )
        
        # 最终结果
        best_result = validation_results[0]
        best_params_array = best_result['params']
        best_profit = best_result['actual_profit']
        
        # 解析最佳参数
        parsed_best_params = {}
        for idx, param_def in enumerate(PARAMETER_DEFINITIONS):
            param_value = best_params_array[idx]
            if param_def['type'] == 'int':
                parsed_best_params[param_def['name']] = int(round(param_value))
            else:
                parsed_best_params[param_def['name']] = param_value
        
        # 输出结果
        print("\n" + "=" * 60)
        print("机器学习优化完成")
        print("=" * 60)
        print(f"最佳盈利: ${best_profit:.2f}")
        print(f"预测盈利: ${best_result['predicted_profit']:.2f}")
        print(f"预测误差: ${best_result['prediction_error']:.2f}")
        
        # 显示最佳参数
        print("\n=== 最佳参数组合 ===")
        print("\n策略参数:")
        for param_def in PARAMETER_DEFINITIONS:
            if param_def.get('strategy') not in ['weight', 'signal', 'risk', 'market_state', 'trend_weights', 'trend_thresholds', 'confidence']:
                print(f"  {param_def['name']}: {parsed_best_params[param_def['name']]}")
        
        print("\n策略权重:")
        weight_sum = 0
        for param_def in PARAMETER_DEFINITIONS:
            if param_def.get('strategy') == 'weight':
                strategy_name = param_def['name'].replace('weight_', '')
                weight_value = parsed_best_params[param_def['name']]
                weight_sum += weight_value
                print(f"  {strategy_name}: {weight_value:.3f}")
        print(f"\n权重总和: {weight_sum:.3f}")
        
        # 保存结果（与原优化器格式相同）
        save_ml_optimization_results(parsed_best_params, best_profit, exploration_data, validation_results)
        
        return parsed_best_params, best_profit
        
    except Exception as e:
        print(f"机器学习优化器运行出错: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

def save_ml_optimization_results(best_params, best_fitness, exploration_data, validation_results):
    """保存机器学习优化结果"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建结果数据结构
        results = {
            'optimization_info': {
                'timestamp': timestamp,
                'method': 'machine_learning',
                'best_fitness': float(best_fitness),
                'exploration_size': len(exploration_data),
                'prediction_size': 1000,
                'validation_top_n': len(validation_results),
                'random_seed': SEED
            },
            'best_parameters': best_params,
            'exploration_results': exploration_data[:50],  # 保存前50个探索结果
            'validation_results': validation_results
        }
        
        # 保存为JSON文件
        json_filename = f"ml_optimization_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n机器学习优化结果已保存到: {json_filename}")
        
        # 保存为CSV文件（只保存最佳参数）
        csv_filename = f"ml_optimization_best_params_{timestamp}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value', 'Type'])
            
            for param_def in PARAMETER_DEFINITIONS:
                param_name = param_def['name']
                param_value = best_params[param_name]
                param_type = param_def['type']
                strategy_type = param_def.get('strategy', 'unknown')
                writer.writerow([param_name, param_value, f"{param_type}_{strategy_type}"])
        
        print(f"最佳参数已保存到: {csv_filename}")
        
        # 生成可读的文本报告
        txt_filename = f"ml_optimization_report_{timestamp}.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("机器学习优化结果报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"优化方法: 随机探索 + 模型预测 + 验证筛选\n")
            f.write(f"最佳适应度 (总盈亏): ${best_fitness:.2f}\n")
            f.write(f"探索样本数: {len(exploration_data)}\n")
            f.write(f"预测样本数: 1000\n")
            f.write(f"验证样本数: {len(validation_results)}\n")
            f.write(f"随机种子: {SEED}\n")
            f.write("\n")
            
            f.write("优化阶段说明:\n")
            f.write("-" * 40 + "\n")
            f.write("1. 随机探索: 生成100组随机参数进行实际回测\n")
            f.write("2. 模型训练: 使用随机森林模型拟合参数与盈利关系\n")
            f.write("3. 预测筛选: 预测1000组参数，选择前20组进行验证\n")
            f.write("\n")
            
            f.write("最佳参数组合:\n")
            f.write("-" * 40 + "\n")
            
            # 分组显示参数
            f.write("\n【策略参数】\n")
            for param_def in PARAMETER_DEFINITIONS:
                if param_def.get('strategy') not in ['weight', 'signal', 'risk', 'market_state', 'trend_weights', 'trend_thresholds', 'confidence']:
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
            f.write("验证结果统计\n")
            f.write("=" * 80 + "\n")
            
            validation_profits = [r['actual_profit'] for r in validation_results]
            prediction_errors = [r['prediction_error'] for r in validation_results]
            
            f.write(f"验证平均盈利: ${np.mean(validation_profits):.2f}\n")
            f.write(f"验证最佳盈利: ${np.max(validation_profits):.2f}\n")
            f.write(f"验证最差盈利: ${np.min(validation_profits):.2f}\n")
            f.write(f"平均预测误差: ${np.mean(prediction_errors):.2f}\n")
            f.write(f"最大预测误差: ${np.max(prediction_errors):.2f}\n")
        
        print(f"详细报告已保存到: {txt_filename}")
        
    except Exception as e:
        print(f"保存机器学习优化结果时出错: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    run_ml_optimizer()