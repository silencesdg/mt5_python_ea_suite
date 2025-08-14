import MetaTrader5 as mt5
import pandas as pd
import random
from logger import logger
from abc import ABC, abstractmethod
from collections import namedtuple
from config import SIMULATION_CONFIG

class DataProvider(ABC):
    """数据提供者抽象基类"""
    @property
    def is_live(self):
        return False

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def get_current_price(self, symbol):
        pass

    @abstractmethod
    def get_historical_data(self, symbol, timeframe, count, **kwargs):
        pass

    @abstractmethod
    def get_account_info(self):
        pass

    @abstractmethod
    def get_positions(self, symbol):
        pass

    @abstractmethod
    def get_symbol_info(self, symbol):
        pass

    @abstractmethod
    def send_order(self, symbol, order_type, volume):
        pass

    @abstractmethod
    def close_position(self, ticket, symbol, volume):
        pass

class LiveDataProvider(DataProvider):
    """实盘数据提供者"""
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
            # 对于某些品种（如XAUUSD），last价格可能为0，使用bid/ask的平均值作为替代
            last_price = tick.last if tick.last != 0 else (tick.bid + tick.ask) / 2
            logger.info(f"获取价格数据 - {symbol}: bid={tick.bid}, ask={tick.ask}, last={tick.last}, 使用last_price={last_price}")
            return {'bid': tick.bid, 'ask': tick.ask, 'last': last_price, 'time': pd.to_datetime(tick.time, unit='s')}
        else:
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
            
        # 对于买入使用ask价格，卖出使用bid价格
        price = price_data['ask'] if order_type == "buy" else price_data['bid']
        logger.info(f"下单价格 - {symbol} {order_type}: 使用价格 {price}")
        
        # 检查终端是否允许自动交易
        if not mt5.terminal_info().trade_allowed:
            logger.error("MT5终端未启用自动交易！请在MT5中点击'自动交易'按钮或按Ctrl+E启用")
            return None
            
        # 检查账户是否允许交易
        account_info = mt5.account_info()
        if account_info and not account_info.trade_allowed:
            logger.error("当前账户不允许自动交易，请联系 broker")
            return None
        
        # 获取品种信息以确定支持的填充模式
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"无法获取 {symbol} 的品种信息")
            return None
            
        # 确定填充模式：优先使用品种支持的填充模式
        filling_mode = mt5.ORDER_FILLING_IOC  # 默认使用IOC
        if symbol_info.filling_mode == 1:  # 只支持FILLING模式
            filling_mode = mt5.ORDER_FILLING_FOK
        elif symbol_info.filling_mode == 2:  # 只支持RETURN模式
            filling_mode = mt5.ORDER_FILLING_RETURN
        elif symbol_info.filling_mode == 3:  # 支持所有模式
            filling_mode = mt5.ORDER_FILLING_RETURN  # 优先使用RETURN
            
        logger.info(f"使用填充模式: {filling_mode} (品种支持模式: {symbol_info.filling_mode})")
        
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
        
        # 检查常见错误代码
        if result and hasattr(result, 'retcode'):
            if result.retcode == 10027:
                logger.error("自动交易被禁用！请在MT5中：")
                logger.error("1. 点击工具栏的'自动交易'按钮（绿色播放图标）")
                logger.error("2. 或按快捷键 Ctrl+E")
                logger.error("3. 确保按钮变为绿色状态")
            elif result.retcode == 10030:
                logger.error("订单填充模式不支持！尝试其他填充模式...")
                # 如果RETURN模式失败，尝试IOC模式
                if filling_mode != mt5.ORDER_FILLING_IOC:
                    logger.info("尝试使用IOC填充模式...")
                    request["type_filling"] = mt5.ORDER_FILLING_IOC
                    result = mt5.order_send(request)
                # 如果IOC模式也失败，尝试FOK模式
                if result and result.retcode == 10030 and filling_mode != mt5.ORDER_FILLING_FOK:
                    logger.info("尝试使用FOK填充模式...")
                    request["type_filling"] = mt5.ORDER_FILLING_FOK
                    result = mt5.order_send(request)
            elif result.retcode != 10009:  # 10009 = 成功
                logger.error(f"下单失败，错误代码: {result.retcode}, 错误信息: {result.comment if hasattr(result, 'comment') else '未知错误'}")
        
        return result

    def close_position(self, ticket, symbol, volume):
        positions = self.get_positions(symbol)
        if not positions: 
            logger.error(f"未找到任何持仓")
            return False
        
        target_position = None
        for pos in positions:
            if pos.ticket == ticket:
                target_position = pos
                break
        
        if not target_position:
            logger.error(f"未找到ticket为 {ticket} 的持仓")
            return False
        
        # 获取当前tick价格
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"无法获取 {symbol} 的当前价格")
            return False
        
        # 根据持仓类型确定平仓价格和订单类型
        if target_position.type == mt5.POSITION_TYPE_BUY:  # 多单平仓
            close_price = tick.bid
            order_type = mt5.ORDER_TYPE_SELL
        else:  # 空单平仓
            close_price = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        
        # 获取品种信息以确定支持的填充模式
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"无法获取 {symbol} 的品种信息")
            return False
            
        # 确定填充模式：优先使用IOC（即时成交或取消）
        filling_mode = mt5.ORDER_FILLING_IOC
        if symbol_info.filling_mode == 1:  # 只支持FILLING模式
            filling_mode = mt5.ORDER_FILLING_FOK
        elif symbol_info.filling_mode == 2:  # 只支持RETURN模式
            filling_mode = mt5.ORDER_FILLING_RETURN
        elif symbol_info.filling_mode == 3:  # 支持所有模式
            filling_mode = mt5.ORDER_FILLING_IOC  # 优先使用IOC
        
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
        
        logger.info(f"发送平仓请求: Ticket={ticket}, 价格={close_price}, 类型={order_type}, 填充模式={filling_mode}")
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"平仓成功: Ticket {ticket}")
            return True
        else:
            logger.error(f"平仓失败: Ticket={ticket}, 错误码={result.retcode}, 错误信息={result.comment}")
            # 记录常见错误的具体原因
            if result.retcode == 10027:
                logger.error("自动交易被禁用！请在MT5中启用自动交易")
            elif result.retcode == 10006:
                logger.error("请求被拒绝，可能是价格变动过快")
            elif result.retcode == 10013:
                logger.error("无效请求，检查参数是否正确")
            elif result.retcode == 10016:
                logger.error("无效的成交量，检查手数是否符合要求")
            return False

class DryRunDataProvider(DataProvider):
    """
    纸上交易提供者 (Paper Trading / Dry Run)
    - 使用来自MT5的实时价格数据
    - 模拟下单和持仓，不发送真实订单
    - 考虑点差影响
    """
    def __init__(self, initial_equity=10000, leverage=100):
        self.simulated_ticket_counter = 0
        self.equity = initial_equity
        self.leverage = leverage
        # 使用LiveDataProvider作为获取市场数据的来源
        self._live_data_source = LiveDataProvider()
        # 从配置获取点差
        self.spread = SIMULATION_CONFIG.get("spread", 16)

    def initialize(self):
        logger.info("纸上交易模式初始化...")
        return self._live_data_source.initialize()

    def shutdown(self):
        logger.info("纸上交易模式关闭.")
        self._live_data_source.shutdown()

    # --- 数据获取方法 (委托给LiveDataProvider) ---
    def get_current_price(self, symbol):
        price_data = self._live_data_source.get_current_price(symbol)
        if price_data:
            # 计算双向点差值（各一半）
            spread_half = self.spread * 0.01 / 2  # XAUUSD: 1点 = 0.01，双向点差各一半
            # 使用中间价计算双向点差
            mid_price = price_data['last']
            # 返回考虑双向点差的价格
            return {
                'bid': mid_price - spread_half,  # 卖出价格（中间价 - 点差/2）
                'ask': mid_price + spread_half,  # 买入价格（中间价 + 点差/2）
                'last': mid_price,  # 最后成交价（中间价）
                'time': price_data['time']
            }
        return None

    def get_historical_data(self, symbol, timeframe, count, **kwargs):
        return self._live_data_source.get_historical_data(symbol, timeframe, count, **kwargs)

    def get_symbol_info(self, symbol):
        return self._live_data_source.get_symbol_info(symbol)

    # --- 模拟状态和交易方法 ---
    def get_account_info(self):
        # 账户信息是模拟的，因为没有真实账户活动
        return {'equity': self.equity}

    def get_positions(self, symbol):
        # 持仓信息是模拟的，由PositionManager在内部管理，这里返回空列表
        return []

    def send_order(self, symbol, order_type, volume):
        self.simulated_ticket_counter += 1
        price_data = self.get_current_price(symbol)
        price = price_data['last'] if price_data else "N/A"
        logger.info(f"[纸上交易] 模拟下单: {order_type} {volume:.2f}手 {symbol} @ {price}")
        return {'order': self.simulated_ticket_counter}

    def close_position(self, ticket, symbol, volume):
        price_data = self.get_current_price(symbol)
        price = price_data['last'] if price_data else "N/A"
        logger.info(f"[纸上交易] 模拟平仓: Ticket {ticket} @ {price}")
        return True

class BacktestDataProvider(DataProvider):
    """回测数据提供者"""
    def __init__(self, df, initial_equity=10000, leverage=100):
        self.df = df
        self.current_index = 0
        self.equity = initial_equity
        self.leverage = leverage
        self.simulated_ticket_counter = 0
        # 从配置获取点差
        self.spread = SIMULATION_CONFIG.get("spread", 16)

    def initialize(self): return True
    def shutdown(self): pass
    def get_account_info(self):
        return {'equity': self.equity}

    def get_positions(self, symbol): return []
    def get_symbol_info(self, symbol):
        return {
            'trade_contract_size': SIMULATION_CONFIG['contract_size'],
            'volume_step': SIMULATION_CONFIG['volume_step'],
            'volume_min': SIMULATION_CONFIG['volume_min'],
            'volume_max': SIMULATION_CONFIG['volume_max'],
        }

    def get_current_price(self, symbol):
        if self.current_index >= len(self.df):
            return None
        row = self.df.iloc[self.current_index]
        # 计算双向点差值（各一半）
        spread_half = self.spread * 0.01 / 2  # XAUUSD: 1点 = 0.01，双向点差各一半
        # 返回考虑双向点差的价格
        return {
            'bid': row.close - spread_half,  # 卖出价格（中间价 - 点差/2）
            'ask': row.close + spread_half,  # 买入价格（中间价 + 点差/2）
            'last': row.close,  # 最后成交价（中间价）
            'time': row.name
        }

    def get_historical_data(self, symbol, timeframe, count, **kwargs):
        if self.current_index < count:
            return None
        end_index = self.current_index + 1
        start_index = max(0, end_index - count)
        return self.df.iloc[start_index:end_index].to_dict('records')

    def send_order(self, symbol, order_type, volume):
        self.simulated_ticket_counter += 1
        logger.info(f"[回测模式] 下单: {order_type} {volume:.2f}手 {symbol}")
        return {'order': self.simulated_ticket_counter}

    def close_position(self, ticket, symbol, volume):
        logger.info(f"[回测模式] 平仓: Ticket {ticket}")
        return True

    def tick(self):
        if self.current_index < len(self.df) - 1:
            self.current_index += 1
            return True
        return False
