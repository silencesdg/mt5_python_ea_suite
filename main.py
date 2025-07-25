import importlib
import pandas as pd
from utils import initialize, shutdown, get_rates, close_all, send_order
from backtest import BacktestEngine
from logger import logger
from config import INITIAL_CAPITAL, STRATEGIES, BUY_THRESHOLD, SELL_THRESHOLD

# 导入优化器
from optimizer import run_optimizer

def run_realtime():
    if not initialize():
        logger.error("MT5初始化失败")
        return

    signals = []
    weights = []
    for strat, weight in STRATEGIES:
        try:
            logger.info(f"执行策略：{strat.__class__.__module__}")
            signal = strat.generate_signal()
            signals.append(signal)
            weights.append(weight)
        except Exception as e:
            logger.exception(f"运行策略 {strat.__class__.__module__} 时出错：{e}")

    weighted_signal_sum = sum(s * w for s, w in zip(signals, weights))

    if weighted_signal_sum >= BUY_THRESHOLD:
        logger.info(f"加权信号总和 ({weighted_signal_sum:.2f}) 达到买入阈值 ({BUY_THRESHOLD})，发送买入信号")
        close_all("XAUUSD")
        send_order("XAUUSD", 'buy')
    elif weighted_signal_sum <= SELL_THRESHOLD:
        logger.info(f"加权信号总和 ({weighted_signal_sum:.2f}) 达到卖出阈值 ({SELL_THRESHOLD})，发送卖出信号")
        close_all("XAUUSD")
        send_order("XAUUSD", 'sell')
    else:
        logger.info(f"加权信号总和 ({weighted_signal_sum:.2f}) 未达到交易阈值，无操作")

    shutdown()

def run_backtest():
    if not initialize():
        logger.error("MT5初始化失败")
        return

    symbol = "XAUUSD"
    timeframe = 1  # M1
    count = 50000
    rates = get_rates(symbol, timeframe, count)
    shutdown()

    if rates is None:
        logger.error("获取历史数据失败")
        return

    logger.info(f"初始资金: {INITIAL_CAPITAL}")

    df = pd.DataFrame(rates)
    engine = BacktestEngine(df)
    signals_list = []
    weights = []

    for strat, weight in STRATEGIES:
        try:
            logger.info(f"回测策略：{strat.__class__.__module__}")
            signals = engine.run_strategy(strat)
            signals_list.append(signals)
            weights.append(weight)
        except Exception as e:
            logger.exception(f"回测策略 {strat.__class__.__module__} 时出错：{e}")

    combined_signal = engine.combine_signals(signals_list, weights, BUY_THRESHOLD, SELL_THRESHOLD)
    cum_ret = engine.calc_returns(combined_signal)
    final_capital = INITIAL_CAPITAL * (1 + cum_ret.iloc[-1])
    logger.info("策略组合回测完成")
    logger.info(f"最终资金: {final_capital:.2f}")
    logger.info(cum_ret.tail())


if __name__ == "__main__":
    # --- 选择运行模式 ---
    # 1. 运行一次回测 (使用config.py中的默认权重)
    # run_backtest()

    # 2. 运行实盘交易 (使用config.py中的默认权重)
    # run_realtime()

    # 3. 运行遗传算法优化，寻找最佳权重
    run_optimizer()
