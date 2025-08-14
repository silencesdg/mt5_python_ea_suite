import sys
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__name__)))

from core.data_providers import BacktestDataProvider
from core.risk import RiskController
from execution.dynamic_weights import DynamicWeightManager # Still needed for strategy list
from config import SYMBOL, TIMEFRAME, BACKTEST_COUNT, BACKTEST_START_DATE, BACKTEST_END_DATE, USE_DATE_RANGE, BACKTEST_CONFIG, INITIAL_CAPITAL, SIGNAL_THRESHOLDS, DEFAULT_WEIGHTS
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
    """完整的、独立的、已重构的回测流程"""
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

    logger.info(f"实际获取数据量: {len(rates)} 条 (请求: {BACKTEST_COUNT} 条)")
    
    # 如果实际获取的数据量超过配置，则截取配置的数据量
    if len(rates) > BACKTEST_COUNT:
        rates = rates[-BACKTEST_COUNT:]  # 取最新的数据
        logger.info(f"截取数据量为: {len(rates)} 条")

    df = pd.DataFrame(rates)
    df.set_index(pd.to_datetime(df['time'], unit='s'), inplace=True)
    
    # 2. 初始化组件 (data_provider, risk_controller)
    data_provider = BacktestDataProvider(df, initial_equity=INITIAL_CAPITAL)
    risk_controller = RiskController(data_provider, trade_direction=BACKTEST_CONFIG['trade_direction'])
    # weight_manager is not directly used for signal generation in this new architecture,
    # but its strategy_blueprints can be used to instantiate strategies.

    # 3. 策略实例化和信号预生成 (NEW STAGE 1)
    logger.info("开始预生成所有策略信号...")
    
    # Define strategy blueprints directly here or get from DynamicWeightManager if it's refactored
    # For now, let's define them directly as DynamicWeightManager is for real-time dynamic weights.
    strategy_blueprints = {
        'ma_cross': (MACrossStrategy, {}),
        'rsi': (RSIStrategy, {}),
        'bollinger': (BollingerStrategy, {}),
        'mean_reversion': (MeanReversionStrategy, {}),
        'momentum_breakout': (MomentumBreakoutStrategy, {}),
        'macd': (MACDStrategy, {}),
        'kdj': (KDJStrategy, {}),
        'turtle': (TurtleStrategy, {}),
        'daily_breakout': (DailyBreakoutStrategy, {}),
        'wave_theory': (WaveTheoryStrategy, {})
    }

    all_signals_df = pd.DataFrame(index=df.index)
    
    # Map strategy class names to config keys for weights
    strategy_name_to_config_key = {
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

    strategies_for_backtest = []
    for name, (strategy_class, params) in strategy_blueprints.items():
        # Pass None for data_provider, symbol, timeframe as run_backtest only needs df
        # But strategy __init__ expects them. So, pass dummy values.
        strategies_for_backtest.append(strategy_class(None, SYMBOL, TIMEFRAME, **params))

    for strategy in tqdm(strategies_for_backtest, desc="Generating Strategy Signals"):
        if hasattr(strategy, 'run_backtest') and callable(getattr(strategy, 'run_backtest')):
            try:
                strategy_signals = strategy.run_backtest(df.copy()) # Pass a copy to avoid modifying original df
                if strategy_signals is not None:
                    all_signals_df[strategy.name] = strategy_signals
                    logger.info(f"策略 {strategy.name} 信号生成成功")
                else:
                    logger.warning(f"策略 {strategy.name} 返回空信号")
            except Exception as e:
                logger.error(f"策略 {strategy.name} 执行失败: {str(e)}")
                # 为失败的策略创建全0信号序列
                all_signals_df[strategy.name] = pd.Series(0, index=df.index)
                continue
        else:
            logger.warning(f"策略 {strategy.name} 没有实现 'run_backtest' 方法，将被跳过。")

    # 组合信号
    logger.info("开始组合策略信号...")
    combined_weighted_signals = pd.Series(0, index=df.index)
    if not all_signals_df.empty:
        weighted_signals_list = []
        for col_name in all_signals_df.columns:
            # Extract strategy class name from column name (e.g., 'MACrossStrategy')
            strategy_class_name = col_name
            config_key = strategy_name_to_config_key.get(strategy_class_name)
            
            if config_key and config_key in DEFAULT_WEIGHTS:
                weight = DEFAULT_WEIGHTS[config_key]
                weighted_signals_list.append(all_signals_df[col_name] * weight)
            else:
                logger.warning(f"未找到策略 {strategy_class_name} 的默认权重，使用权重1.0。")
                weighted_signals_list.append(all_signals_df[col_name] * 1.0)
        
        if weighted_signals_list:
            combined_weighted_signals = pd.concat(weighted_signals_list, axis=1).sum(axis=1)
        else:
            logger.warning("没有生成任何加权信号。")

    # 应用阈值得到最终信号
    final_signals = pd.Series(0, index=df.index)
    buy_threshold = SIGNAL_THRESHOLDS.get("buy_threshold", 1.0)
    sell_threshold = SIGNAL_THRESHOLDS.get("sell_threshold", -1.0)

    final_signals[combined_weighted_signals > buy_threshold] = 1
    final_signals[combined_weighted_signals < sell_threshold] = -1

    logger.info("信号预生成和组合完成。")

    # 4. 回测主循环 (NEW STAGE 2)
    logger.info(f"回测主循环开始... 数据量: {len(df)} 条")
    
    # Reset data_provider's internal index to 0 for the loop
    data_provider.current_index = 0

    for i in tqdm(range(len(df)), desc=f"Backtesting ({len(df)} bars)"):
        try:
            # Get current price from the data_provider (which uses its internal index)
            current_price = data_provider.get_current_price(SYMBOL)
            if not current_price:
                logger.warning(f"无法获取当前价格在索引 {i}，跳过。")
                continue

            # Get the pre-generated signal for the current bar
            current_signal = final_signals.iloc[i]

            direction = None
            if current_signal == 1:
                direction = "buy"
            elif current_signal == -1:
                direction = "sell"
            
            if direction:
                risk_controller.process_trading_signal(direction, current_price, abs(current_signal))

            # Monitor and update positions
            risk_controller.monitor_positions(current_price)

            # Advance data provider to the next time step
            data_provider.tick()
        except Exception as e:
            logger.error(f"回测循环中在索引 {i} 发生错误: {str(e)}")
            # 继续下一个bar，不中断整个回测
            continue

    # 5. 结束和报告 (Keep as is)
    logger.info("回测完成，生成性能报告...")
    summary = risk_controller.position_manager.get_trade_summary()
    
    # 打印报告
    logger.info("=" * 80)
    logger.info("回测性能报告")
    logger.info(f"总交易次数: {summary.get('total_trades', 0)}")
    logger.info(f"胜率: {summary.get('win_rate', 0):.2f}%")
    logger.info(f"总盈亏: ${summary.get('total_profit_loss', 0):.2f}")
    logger.info("=" * 80)

    # 保存交易记录
    try:
        risk_controller.position_manager.save_trade_history("backtest_trades")
        logger.info("交易记录保存成功")
    except Exception as e:
        logger.error(f"保存交易记录失败: {str(e)}")
        # 尝试手动保存
        try:
            import json
            trades = risk_controller.position_manager.closed_trades
            if trades:
                with open("backtest_trades_manual.json", 'w', encoding='utf-8') as f:
                    json.dump(trades, f, ensure_ascii=False, indent=2, default=str)
                logger.info("手动保存交易记录到 backtest_trades_manual.json")
        except Exception as e2:
            logger.error(f"手动保存交易记录也失败: {str(e2)}")


def main():
    print("=" * 60)
    print("MetaTrader 5 智能交易系统 - 回测")
    print("=" * 60)
    
    try:
        run_full_backtest()
        print("\n回测完成！")
    except Exception as e:
        import traceback
        print(f"\n回测出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
