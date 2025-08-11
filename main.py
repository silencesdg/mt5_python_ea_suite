import pandas as pd
from utils import initialize, shutdown, get_rates, close_all, send_order
from backtest import BacktestEngine
from logger import logger
from config import INITIAL_CAPITAL, SYMBOL, TIMEFRAME, BACKTEST_COUNT, BACKTEST_START_DATE, BACKTEST_END_DATE, USE_DATE_RANGE
from risk_management import RiskController
from dynamic_weights import DynamicWeightManager
from trade_logger import trade_logger
from logger import logger
# 导入优化器
from optimizer import run_optimizer

def run_realtime():
    if not initialize():
        logger.error("MT5初始化失败")
        return

    # 初始化风险管理和动态权重
    risk_controller = RiskController()
    weight_manager = DynamicWeightManager(risk_controller)
    
    # 获取动态策略配置
    strategies_with_weights = weight_manager.get_current_strategies_and_weights()
    weight_info = weight_manager.get_weight_info()
    
    logger.info(f"市场状态: {weight_info['market_state']}, 置信度: {weight_info['confidence']:.2f}")
    
    # 执行策略信号生成
    signals = []
    weights = []
    for strat, weight in strategies_with_weights:
        try:
            logger.info(f"执行策略：{strat.__class__.__module__}, 权重: {weight:.2f}")
            
            # 特殊处理风险管理策略
            if strat.__class__.__module__ == 'risk_management':
                signal = strat.generate_signal_with_sync(risk_controller)
            else:
                signal = strat.generate_signal()
            
            signals.append(signal)
            weights.append(weight)
        except Exception as e:
            logger.exception(f"运行策略 {strat.__class__.__module__} 时出错：{e}")

    # 计算加权信号
    weighted_signal_sum = sum(s * w for s, w in zip(signals, weights))
    logger.info(f"加权信号总和: {weighted_signal_sum:.2f}")
    
    # 获取当前价格用于风险管理
    current_price = get_current_price(SYMBOL)
    current_time = pd.Timestamp.now()
    
    # 检查风险管理条件
    if current_price:
        risk_action, risk_reason = risk_controller.check_risk_management(current_price)
        if risk_action != "none":
            logger.info(f"触发风险管理: {risk_action}, 原因: {risk_reason}")
            # 记录平仓交易
            if trade_logger.current_position:
                trade_logger.close_position(SYMBOL, current_price, current_time, f"risk_management_{risk_action}")
            risk_controller.execute_risk_action(risk_action, risk_reason)
            shutdown()
            return
    
    # 获取当前持仓状态，取消交易间隔限制
    current_position = trade_logger.current_position
    min_trade_interval_minutes = 0  # 取消最小交易间隔限制
    
    # 取消交易间隔限制检查
    can_trade = True
    
    # 调试信息：打印当前状态
    logger.info(f"当前持仓状态: {current_position is not None}")
    if current_position:
        time_since_open = (current_time - current_position['open_time']).total_seconds() / 60
        logger.info(f"持仓信息: {current_position['direction']} @ {current_position['open_price']:.2f}, 持仓时间: {time_since_open:.1f}分钟")
    
    # 执行交易决策
    logger.info(f"加权信号总和: {weighted_signal_sum:.2f}, 买入阈值: 0, 卖出阈值: 0")
    logger.info(f"风险管理允许买入: {risk_controller.should_allow_trade('buy')}, 允许卖出: {risk_controller.should_allow_trade('sell')}")
    logger.info(f"允许交易: {can_trade}")
    
    if weighted_signal_sum > 0:
        logger.info(f"检测到买入信号: {weighted_signal_sum:.2f} > 0")
        if risk_controller.should_allow_trade("buy") and can_trade:
            # 检查是否已经有相同方向的持仓
            if current_position and current_position['direction'] == 'buy':
                logger.info(f"已持有多头仓位，信号强度: {weighted_signal_sum:.2f}，不重复开仓")
            else:
                logger.info(f"=== 准备执行买入交易 ===")
                logger.info(f"买入信号: 加权总和({weighted_signal_sum:.2f}) > 0")
                logger.info(f"当前价格: {current_price:.2f}")
                logger.info(f"风险管理允许: {risk_controller.should_allow_trade('buy')}")
                logger.info(f"交易间隔检查通过: {can_trade}")
                
                # 先平掉现有仓位（如果有）
                if current_position:
                    close_all(SYMBOL)
                    trade_logger.close_position(SYMBOL, current_price, current_time, "close_before_buy")
                
                # 开新仓
                send_order(SYMBOL, 'buy')
                trade_logger.open_position(SYMBOL, 'buy', current_price, current_time, f"weighted_signal_{weighted_signal_sum:.2f}")
                
                if current_price:
                    risk_controller.update_position_entry(current_price, "long")
        else:
            if not can_trade:
                logger.info(f"买入信号被阻止: 交易间隔限制")
            else:
                logger.info(f"买入信号被风险管理阻止: risk_controller.should_allow_trade('buy') = {risk_controller.should_allow_trade('buy')}")
                
    elif weighted_signal_sum < 0:
        logger.info(f"检测到卖出信号: {weighted_signal_sum:.2f} < 0")
        if risk_controller.should_allow_trade("sell") and can_trade:
            # 检查是否已经有相同方向的持仓
            if current_position and current_position['direction'] == 'sell':
                logger.info(f"已持有空头仓位，信号强度: {weighted_signal_sum:.2f}，不重复开仓")
            else:
                logger.info(f"=== 准备执行卖出交易 ===")
                logger.info(f"卖出信号: 加权总和({weighted_signal_sum:.2f}) < 0")
                logger.info(f"当前价格: {current_price:.2f}")
                logger.info(f"风险管理允许: {risk_controller.should_allow_trade('sell')}")
                logger.info(f"交易间隔检查通过: {can_trade}")
                
                # 先平掉现有仓位（如果有）
                if current_position:
                    close_all(SYMBOL)
                    trade_logger.close_position(SYMBOL, current_price, current_time, "close_before_sell")
                
                # 开新仓
                send_order(SYMBOL, 'sell')
                trade_logger.open_position(SYMBOL, 'sell', current_price, current_time, f"weighted_signal_{weighted_signal_sum:.2f}")
                
                if current_price:
                    risk_controller.update_position_entry(current_price, "short")
        else:
            logger.info(f"卖出信号被阻止")
    else:
        # 信号在中间区域，使用RiskController决定是否平仓
        if current_position:
            # 使用RiskController检查风险管理条件
            risk_action, risk_reason = risk_controller.check_risk_management(current_price)
            
            if risk_action != "none":
                close_all(SYMBOL)
                trade_logger.close_position(SYMBOL, current_price, current_time, f"signal_neutral_{risk_action}")
                risk_controller.execute_risk_action(risk_action, risk_reason)

    shutdown()

def get_current_price(symbol):
    """获取当前价格"""
    try:
        import MetaTrader5 as mt5
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid if tick else None
    except Exception as e:
        logger.error(f"获取当前价格失败: {e}")
        return None

def run_backtest():
    if not initialize():
        logger.error("MT5初始化失败")
        return

    # 重置交易日志记录器
    trade_logger.__init__()
    
    # 根据配置选择获取数据的方式
    if USE_DATE_RANGE:
        rates = get_rates(SYMBOL, TIMEFRAME, BACKTEST_COUNT, BACKTEST_START_DATE, BACKTEST_END_DATE)
    else:
        rates = get_rates(SYMBOL, TIMEFRAME, BACKTEST_COUNT)

    if rates is None:
        logger.error("获取历史数据失败")
        shutdown()
        return

    logger.info(f"初始资金: {INITIAL_CAPITAL}")

    df = pd.DataFrame(rates)
    engine = BacktestEngine(df)
    
    # 使用动态权重管理器（需要在MT5连接状态下获取市场状态）
    risk_controller = RiskController()
    weight_manager = DynamicWeightManager(risk_controller)
    
    # 获取当前策略配置（回测时使用固定权重或模拟动态权重）
    strategies_with_weights = weight_manager.get_current_strategies_and_weights()
    weight_info = weight_manager.get_weight_info()
    
    logger.info(f"回测使用市场状态: {weight_info['market_state']}")
    
    # 获取市场状态后关闭MT5连接
    shutdown()
    
    signals_list = []
    weights = []
    strategy_names = []

    for strat, weight in strategies_with_weights:
        try:
            logger.info(f"回测策略：{strat.__class__.__module__}, 权重: {weight:.2f}")
            signals = engine.run_strategy(strat)
            signals_list.append(signals)
            weights.append(weight)
            strategy_names.append(strat.__class__.__module__)
        except Exception as e:
            logger.exception(f"回测策略 {strat.__class__.__module__} 时出错：{e}")

    combined_signal = engine.combine_signals(signals_list, weights)
    
    # 使用带交易记录的收益率计算
    cum_ret = engine.calc_returns_with_trades(combined_signal, SYMBOL)
    final_capital = INITIAL_CAPITAL * (1 + cum_ret.iloc[-1])
    
    logger.info("策略组合回测完成")
    logger.info(f"最终资金: {final_capital:.2f}")
    logger.info(f"总收益率: {(final_capital/INITIAL_CAPITAL - 1):.2%}")
    logger.info("使用的策略权重:")
    for name, weight in zip(strategy_names, weights):
        logger.info(f"  {name}: {weight:.2f}")
    logger.info("最近收益率:")
    logger.info(cum_ret.tail())
    
    # 打印详细的交易记录
    logger.info("\n" + "="*50)
    logger.info("详细交易记录:")
    trade_logger.print_all_trades()
    
    # 保存交易记录到文件
    trade_logger.save_to_csv("backtest_trades.csv")
    trade_logger.save_to_json("backtest_trades.json")


if __name__ == "__main__":
    # --- 选择运行模式 ---
    # 1. 运行一次回测 (使用config.py中的默认权重)
    run_backtest()

    # 2. 运行实盘交易 (使用config.py中的默认权重)
    # run_realtime()

    # 3. 运行遗传算法优化，寻找最佳权重
    # run_optimizer()
