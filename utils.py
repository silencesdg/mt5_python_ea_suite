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

def get_rates(symbol, timeframe, count):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        logger.info(f"获取{symbol}历史数据失败")
        return None
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
        return

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
    else:
        logger.info(f"下单成功: {order_type} {symbol} {volume}")
