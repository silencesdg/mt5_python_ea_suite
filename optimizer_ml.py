
import json
import csv
from datetime import datetime
import os
import sys
import numpy as np
import random
from tqdm import tqdm
import multiprocessing
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

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
    USE_DATE_RANGE, BACKTEST_CONFIG, INITIAL_CAPITAL, SIGNAL_THRESHOLDS,
    RISK_CONFIG, GENETIC_OPTIMIZER_CONFIG
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
    return df
# --- Parameter Definition --- 
# (保持与原 optimizer.py 一致)
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
    {'name': 'stop_loss_pct',         'type': 'float', 'min': -0.02, 'max': -0.0005, 'strategy': 'risk'},
    {'name': 'profit_retracement_pct','type': 'float', 'min': 0.05, 'max': 0.20, 'strategy': 'risk'},
    {'name': 'min_profit_for_trailing','type': 'float', 'min': 0.005, 'max': 0.01, 'strategy': 'risk'},
    {'name': 'take_profit_pct',       'type': 'float', 'min': 0.10, 'max': 1.0, 'strategy': 'risk'},
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

# --- Backtest Evaluation Function ---
def evaluate_fitness(params_tuple, df_data):
    """
    执行回测并计算夏普比率作为评估指标。
    (已移除DynamicWeightManager，使用固定的优化权重)
    """
    # --- 1. 参数解析 ---
    individual = list(params_tuple)
    parsed_params = {param_def['name']: (int(round(val)) if param_def['type'] == 'int' else val) 
                     for param_def, val in zip(PARAMETER_DEFINITIONS, individual)}

    strategy_params_dict = {}
    for param_def in PARAMETER_DEFINITIONS:
        strategy_name = param_def.get('strategy')
        if strategy_name and strategy_name not in ['weight', 'signal', 'risk', 'market_state', 'trend_weights', 'trend_thresholds', 'confidence']:
            if strategy_name not in strategy_params_dict:
                strategy_params_dict[strategy_name] = {}
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
            elif param_name == 'kdj_period': param_name = 'period'
            elif param_name == 'turtle_period': param_name = 'period'
            elif param_name.startswith("wave_"):
                param_name = param_name[5:]
                if param_name == "period": param_name = "wave_period"
            elif param_name == 'daily_breakout_bars_count': param_name = 'bars_count'
            strategy_params_dict[strategy_name][param_name] = parsed_params[param_def['name']]

    strategies_for_backtest_instance = {
        config_key: strategy_class(None, SYMBOL, TIMEFRAME, **strategy_params_dict.get(strategy_class.__name__, {}))
        for config_key, strategy_class in strategy_config_key_to_class.items()
    }
    
    suggested_weights = {p['name'].replace('weight_', ''): parsed_params[p['name']] for p in PARAMETER_DEFINITIONS if p.get('strategy') == 'weight'}
    signal_thresholds = {'buy_threshold': parsed_params.get('buy_threshold', 1.0), 'sell_threshold': parsed_params.get('sell_threshold', -1.0)}
    risk_params = {p['name']: parsed_params[p['name']] for p in PARAMETER_DEFINITIONS if p.get('strategy') == 'risk'}

    # --- 2. 回测设置 ---
    df = df_data.copy()
    data_provider = BacktestDataProvider(df, initial_equity=INITIAL_CAPITAL)
    risk_controller = RiskController(data_provider, trade_direction=BACKTEST_CONFIG['trade_direction'])
    for key, value in risk_params.items():
        if hasattr(risk_controller, key): setattr(risk_controller, key, value)

    # --- 3. 信号生成 ---
    all_signals_df = pd.DataFrame(index=df.index)
    for config_key, strat_instance in strategies_for_backtest_instance.items():
        if hasattr(strat_instance, 'run_backtest'):
            all_signals_df[strat_instance.name] = strat_instance.run_backtest(df.copy())
    
    if all_signals_df.empty: return (0.0,)
    
    # --- 4. 信号加权 (使用优化器传入的固定权重) ---
    weights_arr = []
    for name, sig in all_signals_df.items():
        weight_key = name.replace('Strategy', '')
        w = suggested_weights.get(weight_key, 1.0)
        weights_arr.append(sig.values * w)

    combined = np.sum(np.column_stack(weights_arr), axis=1)
    final_signals = np.where(combined > signal_thresholds['buy_threshold'], 1, np.where(combined < signal_thresholds['sell_threshold'], -1, 0))

    # --- 5. 回测循环与权益曲线记录 ---
    equity_curve = []
    equity_dates = []
    symbol_info = data_provider.get_symbol_info(SYMBOL)
    contract_size = symbol_info['trade_contract_size'] if symbol_info else 100

    for i in range(len(df)):
        current_price = data_provider.get_current_price(SYMBOL)
        if not current_price: continue
        
        direction = "buy" if final_signals[i] == 1 else "sell" if final_signals[i] == -1 else None
        if direction:
            risk_controller.process_trading_signal(direction, current_price, abs(final_signals[i]))
        risk_controller.monitor_positions(current_price)
        
        realized_pnl = sum(t['profit_loss'] for t in risk_controller.position_manager.closed_trades)
        unrealized_pnl = sum(
            risk_controller.position_manager._calculate_pnl_pct(pos, current_price['last']) * pos['entry_price'] * pos['quantity'] * contract_size
            for pos in risk_controller.position_manager.positions
        )
        
        current_equity = INITIAL_CAPITAL + realized_pnl + unrealized_pnl
        equity_curve.append(current_equity)
        equity_dates.append(df.index[i])
        data_provider.tick()

    # --- 6. 夏普比率计算 ---
    if not equity_curve: return (0.0,)
    
    equity_series = pd.Series(equity_curve, index=pd.to_datetime(equity_dates))
    daily_equity = equity_series.resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()

    if daily_returns.empty or daily_returns.std() == 0: return (0.0,)

    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    return (sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0,)

# --- Helper Functions ---
def generate_random_params():
    """生成一组随机参数"""
    params = []
    for param_def in PARAMETER_DEFINITIONS:
        if param_def['type'] == 'int':
            params.append(random.randint(param_def['min'], param_def['max']))
        else:
            params.append(random.uniform(param_def['min'], param_def['max']))
    return tuple(params)

def run_backtest_for_params(args):
    """用于多进程回测的包装函数, 现在返回夏普比率"""
    params, df_data = args
    try:
        sharpe = evaluate_fitness(params, df_data)[0]
        return params, sharpe
    except Exception as e:
        return params, e # 返回异常对象本身

# --- Main Optimizer Logic ---
def run_optimizer(df_to_optimize=None):
    """
    主优化器函数，采用“探索-利用”策略, 基于夏普比率
    """
    try:
        if df_to_optimize is not None:
            # print("优化器正在使用传入的数据...")
            df_historical_data = df_to_optimize
        else:
            # print("优化器正在加载默认历史数据...")
            df_historical_data = load_historical_data()
        
        ML_OPTIMIZER_CONFIG = {
            "exploration_samples": 500,
            "prediction_samples": 10000,
            "top_n_candidates": 10,
            "enable_multiprocessing": True,
            "processes": multiprocessing.cpu_count() - 1 or 1,
            "model_filename": "sharpe_predictor_model.joblib"
        }

        # --- 1. 探索阶段 ---
        print(f"--- 阶段 1: 探索 (生成 {ML_OPTIMIZER_CONFIG['exploration_samples']} 组随机参数以计算夏普比率) ---")
        exploration_params = [generate_random_params() for _ in range(ML_OPTIMIZER_CONFIG['exploration_samples'])]
        
        exploration_results = []
        if ML_OPTIMIZER_CONFIG["enable_multiprocessing"]:
            pool = multiprocessing.Pool(processes=ML_OPTIMIZER_CONFIG["processes"], initializer=init_worker)
            tasks = [(p, df_historical_data) for p in exploration_params]
            with tqdm(total=len(tasks), desc="探索回测中 (计算夏普比率)") as pbar:
                for result in pool.imap_unordered(run_backtest_for_params, tasks):
                    exploration_results.append(result)
                    pbar.update()
            pool.close()
            pool.join()
        else:
            for params in tqdm(exploration_params, desc="探索回测中 (计算夏普比率)"):
                exploration_results.append(run_backtest_for_params((params, df_historical_data)))

        exploration_df = pd.DataFrame(exploration_results, columns=['params', 'sharpe'])

        # 筛选出失败和成功的结果
        failed_mask = exploration_df['sharpe'].apply(lambda x: isinstance(x, Exception))
        failed_results = exploration_df[failed_mask]
        
        if not failed_results.empty:
            print(f"\n--- 错误分析: 发现 {len(failed_results)}/{len(exploration_df)} 个失败的回测 ---")
            
            # 统计独特的错误信息
            error_counts = failed_results['sharpe'].astype(str).value_counts()
            
            print("错误类型统计:")
            for error, count in error_counts.items():
                print(f"  - [{count}次] {error}")

            print("--------------------------------------------------\n")

        exploration_df = exploration_df[~failed_mask] # 使用反向掩码筛选有效结果
        print(f"探索完成，获得 {len(exploration_df)} 个有效回测结果。")

        # --- 2. 利用阶段 (模型训练) ---
        print("\n--- 阶段 2: 利用 (训练夏普比率预测模型) ---")
        if len(exploration_df) < 2:
            print("有效数据不足，无法训练模型。")
            return

        X = np.array(list(exploration_df['params']))
        y = exploration_df['sharpe'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"模型训练完成。测试集均方误差 (MSE): {mse:.4f}")
        joblib.dump(model, ML_OPTIMIZER_CONFIG['model_filename'])
        print(f"模型已保存到: {ML_OPTIMIZER_CONFIG['model_filename']}")

        # --- 3. 优化阶段 (模型预测与验证) ---
        print(f"\n--- 阶段 3: 优化 (使用模型预测 {ML_OPTIMIZER_CONFIG['prediction_samples']} 组参数的夏普比率) ---")
        prediction_params = [generate_random_params() for _ in range(ML_OPTIMIZER_CONFIG['prediction_samples'])]
        predicted_sharpes = model.predict(np.array(prediction_params))
        
        top_n_indices = np.argsort(predicted_sharpes)[-ML_OPTIMIZER_CONFIG['top_n_candidates']:]
        top_n_candidates = [tuple(p) for p in np.array(prediction_params)[top_n_indices]]
        print(f"模型预测完成，选出 Top {ML_OPTIMIZER_CONFIG['top_n_candidates']} 组候选参数进行最终验证。")

        # --- 4. 最终验证 ---
        print("\n--- 阶段 4: 最终验证 (回测 Top N 候选参数) ---")
        final_results = []
        if ML_OPTIMIZER_CONFIG["enable_multiprocessing"]:
            pool = multiprocessing.Pool(processes=ML_OPTIMIZER_CONFIG["processes"], initializer=init_worker)
            tasks = [(p, df_historical_data) for p in top_n_candidates]
            with tqdm(total=len(tasks), desc="最终验证回测中") as pbar:
                for result in pool.imap_unordered(run_backtest_for_params, tasks):
                    final_results.append(result)
                    pbar.update()
            pool.close()
            pool.join()
        else:
            for params in tqdm(top_n_candidates, desc="最终验证回测中"):
                final_results.append(run_backtest_for_params((params, df_historical_data)))

        best_result = max(final_results, key=lambda item: item[1] if isinstance(item[1], (int, float)) else -float('inf'))
        best_params_tuple, best_sharpe = best_result
        
        best_params_dict = {param_def['name']: (int(round(val)) if param_def['type'] == 'int' else val) for idx, (param_def, val) in enumerate(zip(PARAMETER_DEFINITIONS, best_params_tuple))}

        print("\n--- 优化结束 ---")
        print(f"最佳夏普比率: {best_sharpe:.4f}")
        print("最佳参数组合:")
        for key, value in best_params_dict.items():
            print(f"  {key}: {value}")
            
        save_optimization_results(best_params_dict, best_sharpe, exploration_df, final_results)
        return best_params_dict, best_sharpe

    except Exception as e:
        print(f"优化器运行出错: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

def save_optimization_results(best_params, best_fitness, exploration_df, final_results_list):
    """保存优化结果到文件"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_results_df = pd.DataFrame(final_results_list, columns=['params', 'sharpe'])
        
        results = {
            'optimization_info': {
                'timestamp': timestamp,
                'best_fitness_metric': 'Sharpe Ratio',
                'best_fitness_value': float(best_fitness),
                'optimizer_type': 'Machine Learning Based',
                'random_seed': SEED
            },
            'best_parameters': best_params,
            'exploration_summary': {
                'count': len(exploration_df),
                'max_sharpe': exploration_df['sharpe'].max(),
                'avg_sharpe': exploration_df['sharpe'].mean()
            },
            'final_candidates': final_results_df.to_dict('records')
        }
        
        json_filename = f"ml_optimization_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating)): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            json.dump(results, f, ensure_ascii=False, indent=2, cls=NpEncoder)
        print(f"\n优化结果已保存到: {json_filename}")
        
        csv_filename = f"ml_optimization_best_params_{timestamp}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            for key, value in best_params.items():
                writer.writerow([key, value])
        print(f"最佳参数已保存到: {csv_filename}")

    except Exception as e:
        print(f"保存优化结果时出错: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    run_optimizer()
