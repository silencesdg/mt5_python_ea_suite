import MetaTrader5 as mt5
import pandas as pd
from logger import logger
from core.data.abc import DataProvider


class LiveDataProvider(DataProvider):
    """实盘数据提供者 — 封装真实MT5 API调用"""

    @property
    def is_live(self):
        return True

    def initialize(self):
        if not mt5.initialize():
            logger.error("MT5初始化失败")
            return False
        logger.info("MT5连接成功")
        return True

    def shutdown(self):
        mt5.shutdown()
        logger.info("MT5连接已关闭")

    def get_current_price(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            last_price = tick.last if tick.last != 0 else (tick.bid + tick.ask) / 2
            return {
                'bid': tick.bid, 'ask': tick.ask, 'last': last_price,
                'time': pd.to_datetime(tick.time, unit='s')
            }
        logger.error(f"无法获取 {symbol} 的价格数据")
        return None

    def get_historical_data(self, symbol, timeframe, count, **kwargs):
        return mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    def get_account_info(self):
        return mt5.account_info()

    def get_positions(self, symbol):
        return mt5.positions_get(symbol=symbol)

    def get_symbol_info(self, symbol):
        return mt5.symbol_info(symbol)

    def send_order(self, symbol, order_type, volume):
        price_data = self.get_current_price(symbol)
        if not price_data:
            logger.error(f"无法获取 {symbol} 价格，无法下单")
            return None

        price = price_data['ask'] if order_type == "buy" else price_data['bid']

        if not mt5.terminal_info().trade_allowed:
            logger.error("MT5终端未启用自动交易")
            return None

        account_info = mt5.account_info()
        if account_info and not account_info.trade_allowed:
            logger.error("当前账户不允许自动交易")
            return None

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"无法获取 {symbol} 的品种信息")
            return None

        # MT5 filling_mode 是位图，用位与判断
        fm = symbol_info.filling_mode
        # MQL5 filling_mode 位图: FOK=1, IOC=2
        if fm & 1:
            filling_mode = mt5.ORDER_FILLING_FOK
        elif fm & 2:
            filling_mode = mt5.ORDER_FILLING_IOC
        else:
            filling_mode = mt5.ORDER_FILLING_RETURN

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
            "type_filling": filling_mode,
        }
        result = mt5.order_send(request)

        if result and hasattr(result, 'retcode'):
            if result.retcode == 10027:
                logger.error("自动交易被禁用")
            elif result.retcode == 10030:
                if filling_mode != mt5.ORDER_FILLING_IOC:
                    request["type_filling"] = mt5.ORDER_FILLING_IOC
                    result = mt5.order_send(request)
                if result and result.retcode == 10030 and filling_mode != mt5.ORDER_FILLING_FOK:
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result = mt5.order_send(request)
            elif result.retcode != 10009:
                logger.error(f"下单失败，错误代码: {result.retcode}")

        return result

    def close_position(self, ticket, symbol, volume):
        positions = self.get_positions(symbol)
        if not positions:
            logger.error("未找到任何持仓")
            return False

        target_position = None
        for pos in positions:
            if pos.ticket == ticket:
                target_position = pos
                break

        if not target_position:
            logger.error(f"未找到ticket为 {ticket} 的持仓")
            return False

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"无法获取 {symbol} 的当前价格")
            return False

        if target_position.type == mt5.POSITION_TYPE_BUY:
            close_price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:
            close_price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return False

        fm = symbol_info.filling_mode
        # MQL5 filling_mode 位图: FOK=1, IOC=2
        if fm & 1:
            filling_mode = mt5.ORDER_FILLING_FOK
        elif fm & 2:
            filling_mode = mt5.ORDER_FILLING_IOC
        else:
            filling_mode = mt5.ORDER_FILLING_IOC

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": target_position.ticket,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": close_price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Close position {ticket}",
            "type_filling": filling_mode,
        }

        result = mt5.order_send(request)
        if result is None:
            logger.error(f"平仓请求返回None: Ticket={ticket}")
            return False
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"平仓成功: Ticket {ticket}")
            return True
        else:
            logger.error(f"平仓失败: Ticket={ticket}, 错误码={result.retcode}")
            return False
