import MetaTrader5 as mt5
import pandas as pd
from logger import setup_logger
logger = setup_logger()

def initialize():
    if not mt5.initialize():
        logger.error("MT5初始化失败，错误代码：%d", mt5.last_error())
        return False
    return True

def shutdown():
    mt5.shutdown()

def get_rates(symbol, timeframe, count, start_date=None, end_date=None):
    """
    获取历史数据
    参数:
    - symbol: 交易品种
    - timeframe: 时间周期
    - count: 数据量 (当start_date和end_date都为None时使用)
    - start_date: 开始日期 (格式: "YYYY-MM-DD" 或 datetime对象)
    - end_date: 结束日期 (格式: "YYYY-MM-DD" 或 datetime对象)
    """
    # 检查MT5连接状态
    if not mt5.terminal_info():
        logger.warning("MT5终端未连接，尝试重新连接...")
        if not initialize():
            logger.error("MT5重新连接失败")
            return None
    
    # 检查交易品种是否可用
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"交易品种 {symbol} 不可用")
        return None
    
    if not symbol_info.visible:
        logger.info(f"交易品种 {symbol} 不可见，尝试启用...")
        if not mt5.symbol_select(symbol, True):
            logger.error(f"无法启用交易品种 {symbol}")
            return None
    if start_date is not None and end_date is not None:
        # 使用日期范围获取数据
        import datetime
        
        # 转换字符串为datetime对象
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        # MetaTrader5需要UTC时间，且copy_rates_range需要timezone-aware的datetime对象
        # 转换为UTC timezone
        utc_timezone = datetime.timezone.utc
        
        start_utc = start_date.replace(tzinfo=utc_timezone)
        end_utc = end_date.replace(hour=23, minute=59, second=59, tzinfo=utc_timezone)
        
        try:
            rates = mt5.copy_rates_range(symbol, timeframe, start_utc, end_utc)
            if rates is None:
                logger.info(f"获取{symbol}从{start_date.date()}到{end_date.date()}的历史数据失败")
                return None
            logger.debug(f"成功获取{symbol}从{start_date.date()}到{end_date.date()}的历史数据，共{len(rates)}条")
        except Exception as e:
            logger.error(f"使用日期范围获取数据失败: {e}")
            logger.info("回退到使用数据量获取数据")
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                logger.info(f"获取{symbol}历史数据失败")
                return None
            logger.debug(f"成功获取{symbol}历史数据，共{len(rates)}条")
    else:
        # 使用数据量获取数据
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            logger.info(f"获取{symbol}历史数据失败")
            return None
        logger.debug(f"成功获取{symbol}历史数据，共{len(rates)}条")
    
    return rates

def has_open_position(symbol):
    positions = mt5.positions_get(symbol=symbol)
    return positions is not None and len(positions) > 0

def close_all(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return
    for pos in positions:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "Close position",
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        mt5.order_send(request)

def send_order(symbol, order_type, volume=0.01):
    symbol_info_tick = mt5.symbol_info_tick(symbol)
    if symbol_info_tick is None:
        logger.error(f"无法获取{symbol}行情")
        return None

    price = symbol_info_tick.ask if order_type == "buy" else symbol_info_tick.bid
    order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type_mt5,
        "price": price,
        "deviation": 20,
        "magic": 234000,
        "comment": f"{order_type} order",
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"下单失败，retcode={result.retcode}")
        return None
    else:
        logger.info(f"下单成功: {order_type} {symbol} {volume}, ticket: {result.order}")
        return result

def close_position(ticket, symbol, volume):
    """根据ticket平掉一个特定的仓位"""
    # In MT5, you close a position by creating an opposite order.
    # We need to get the position details first.
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        logger.error(f"无法找到ticket为 {ticket} 的持仓")
        return False
    
    pos = positions[0] # positions_get returns a tuple of objects

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos.ticket,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY, # pos.type == 0 is a BUY position
        "price": mt5.symbol_info_tick(symbol).bid if pos.type == 0 else mt5.symbol_info_tick(symbol).ask,
        "deviation": 20,
        "magic": 234000,
        "comment": f"Close position {ticket}",
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"平仓失败 ticket {ticket}, retcode={result.retcode}")
        return False
    else:
        logger.info(f"平仓成功 ticket {ticket}")
        return True

def get_current_price(symbol):
    """获取当前价格"""
    try:
        # 检查MT5连接状态
        if not mt5.terminal_info():
            logger.warning("MT5终端未连接，尝试重新连接...")
            if not initialize():
                logger.error("MT5重新连接失败")
                return None
        
        # 检查交易品种是否可用
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"交易品种 {symbol} 不可用")
            return None
        
        if not symbol_info.visible:
            logger.info(f"交易品种 {symbol} 不可见，尝试启用...")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"无法启用交易品种 {symbol}")
                return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"无法获取 {symbol} 的价格信息")
            return None
            
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'time': pd.to_datetime(tick.time, unit='s')
        }
    except Exception as e:
        logger.error(f"获取当前价格失败: {e}")
        return None
