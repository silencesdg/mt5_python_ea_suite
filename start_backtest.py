import sys
import os
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_providers import BacktestDataProvider
from core.risk import RiskController
from config import SYMBOL, TIMEFRAME, BACKTEST_COUNT, BACKTEST_START_DATE, BACKTEST_END_DATE, USE_DATE_RANGE, BACKTEST_CONFIG, INITIAL_CAPITAL
from logger import logger

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

def run_full_backtest():
    """完整的、独立的、已重构的回测流程 (支持市场状态)"""
    # 1. 加载数据
    from core.utils import get_rates, initialize, shutdown
    initialize()
    if USE_DATE_RANGE:
        rates = get_rates(SYMBOL, TIMEFRAME, BACKTEST_COUNT, BACKTEST_START_DATE, BACKTEST_END_DATE)
    else:
        rates = get_rates(SYMBOL, TIMEFRAME, BACKTEST_COUNT)
    shutdown()

    if rates is None:
        logger.error("未能获取历史数据，回测终止。")
        return

    df = pd.DataFrame(rates)
    df.set_index(pd.to_datetime(df['time'], unit='s'), inplace=True)

    # 2. 加载市场状态参数
    try:
        with open('regime_optimal_params.json', 'r') as f:
            regime_params = json.load(f)
        logger.info("成功加载市场状态参数文件。")
    except FileNotFoundError:
        logger.error("错误: regime_optimal_params.json 未找到。请先运行 regime_optimizer.py。")
        return

    # 3. 市场状态分类
    logger.info("开始为整个数据集分类市场状态...")
    regime_detector = WaveTheoryStrategy(None, SYMBOL, TIMEFRAME)
    df_with_indicators = regime_detector._calculate_indicators(df.copy())
    
    regimes = []
    adx_values = df_with_indicators['adx'].values
    ema_short_values = df_with_indicators['ema_short'].values
    ema_medium_values = df_with_indicators['ema_medium'].values
    ema_long_values = df_with_indicators['ema_long'].values
    adx_threshold = regime_detector.adx_threshold

    for i in range(len(df_with_indicators)):
        if pd.isna(adx_values[i]) or pd.isna(ema_short_values[i]):
            regimes.append("Ranging")
        elif adx_values[i] < adx_threshold:
            regimes.append("Ranging")
        elif ema_short_values[i] > ema_medium_values[i] > ema_long_values[i]:
            regimes.append("Uptrend")
        else:
            regimes.append("Downtrend")
    df['regime'] = regimes
    logger.info("市场状态分类完成。")
    logger.info(df['regime'].value_counts())

    # 4. 分状态生成信号
    logger.info("开始分状态生成所有策略信号...")
    all_signals_df = pd.DataFrame(index=df.index)
    strategy_classes = {
        "MACrossStrategy": MACrossStrategy,
        "RSIStrategy": RSIStrategy,
        "BollingerStrategy": BollingerStrategy,
        "MeanReversionStrategy": MeanReversionStrategy,
        "MomentumBreakoutStrategy": MomentumBreakoutStrategy,
        "MACDStrategy": MACDStrategy,
        "KDJStrategy": KDJStrategy,
        "TurtleStrategy": TurtleStrategy,
        "DailyBreakoutStrategy": DailyBreakoutStrategy,
        "WaveTheoryStrategy": WaveTheoryStrategy
    }

    for regime in ['Uptrend', 'Downtrend', 'Ranging']:
        logger.info(f"--- 为 {regime} 状态生成信号 ---")
        regime_df = df[df['regime'] == regime]
        if regime_df.empty:
            logger.info(f"{regime} 状态没有数据，跳过。")
            continue

        params = regime_params.get(regime, {}).get('best_parameters', {})
        if not params:
            logger.warning(f"未找到 {regime} 的参数，将无法为该状态生成信号。")
            continue

        for strat_name, strat_class in strategy_classes.items():
            instance = strat_class(None, SYMBOL, TIMEFRAME)
            strat_params = {}
            prefix_map = {
                'MACrossStrategy': 'ma_cross_',
                'RSIStrategy': 'rsi_',
                'BollingerStrategy': 'bollinger_',
                'MACDStrategy': 'macd_',
                'MeanReversionStrategy': 'mean_reversion_',
                'MomentumBreakoutStrategy': 'momentum_breakout_',
                'KDJStrategy': 'kdj_',
                'TurtleStrategy': 'turtle_',
                'DailyBreakoutStrategy': 'daily_breakout_',
                'WaveTheoryStrategy': 'wave_'
            }
            prefix = prefix_map[strat_name]
            for p_name, p_val in params.items():
                if p_name.startswith(prefix):
                    param_name = p_name.replace(prefix, '')
                    if strat_name == 'WaveTheoryStrategy' and p_name == 'wave_period':
                        param_name = 'wave_period'
                    strat_params[param_name] = p_val
            
            instance.set_params(strat_params)
            
            try:
                signals = instance.run_backtest(regime_df.copy())
                if signals is not None:
                    all_signals_df.loc[regime_df.index, strat_name] = signals
            except Exception as e:
                logger.error(f"策略 {strat_name} 在 {regime} 状态下执行失败: {e}")

    all_signals_df.fillna(0, inplace=True)

    # 5. 组合信号
    logger.info("开始组合策略信号...")
    final_signals = pd.Series(0.0, index=df.index)
    for regime in ['Uptrend', 'Downtrend', 'Ranging']:
        regime_df_indices = df[df['regime'] == regime].index
        if regime_df_indices.empty: continue

        params = regime_params.get(regime, {}).get('best_parameters', {})
        weights = {k.replace('weight_', ''): v for k, v in params.items() if k.startswith('weight_')}
        buy_threshold = params.get('buy_threshold', 1.5)
        sell_threshold = params.get('sell_threshold', -1.5)

        regime_signals = all_signals_df.loc[regime_df_indices]
        weighted_sum = pd.Series(0.0, index=regime_signals.index)
        for strat_name, weight in weights.items():
            if strat_name in regime_signals.columns:
                weighted_sum += regime_signals[strat_name] * weight
        
        # BUG FIX: Use indices to avoid alignment errors
        buy_indices = weighted_sum[weighted_sum > buy_threshold].index
        sell_indices = weighted_sum[weighted_sum < sell_threshold].index

        final_signals.loc[buy_indices] = 1
        final_signals.loc[sell_indices] = -1

    # 6. 回测主循环
    logger.info("回测主循环开始...")
    data_provider = BacktestDataProvider(df, initial_equity=INITIAL_CAPITAL)
    risk_controller = RiskController(data_provider, trade_direction=BACKTEST_CONFIG['trade_direction'])

    for i in tqdm(range(len(df)), desc="Backtesting"):
        current_price = data_provider.get_current_price(SYMBOL)
        if not current_price: continue

        current_signal = final_signals.iloc[i]
        direction = "buy" if current_signal == 1 else "sell" if current_signal == -1 else None
        
        if direction:
            risk_controller.process_trading_signal(direction, current_price, abs(current_signal))

        risk_controller.monitor_positions(current_price)
        data_provider.tick()

    # 7. 结束和报告
    logger.info("回测完成，生成性能报告...")
    summary = risk_controller.position_manager.get_trade_summary()
    logger.info("=" * 80)
    logger.info("回测性能报告")
    logger.info(f"总交易次数: {summary.get('total_trades', 0)}")
    logger.info(f"胜率: {summary.get('win_rate', 0):.2f}%")
    logger.info(f"总盈亏: ${summary.get('total_profit_loss', 0):.2f}")
    logger.info("=" * 80)
    risk_controller.position_manager.save_trade_history("backtest_trades")

def main():
    print("=" * 60)
    print("MetaTrader 5 智能交易系统 - 回测 (市场状态模式)")
    print("=" * 60)
    run_full_backtest()

if __name__ == "__main__":
    main()