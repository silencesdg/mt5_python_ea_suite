import pandas as pd
from logger import logger
from core.data.abc import DataProvider
from core.data.live import LiveDataProvider
from config import SIMULATION_CONFIG


class DryRunDataProvider(DataProvider):
    """纸上交易数据提供者 — 使用实时价格但不发送真实订单"""

    def __init__(self, initial_equity=10000, leverage=100):
        self.simulated_ticket_counter = 0
        self.equity = initial_equity
        self.leverage = leverage
        self._live_data_source = LiveDataProvider()
        self.spread = SIMULATION_CONFIG.get("spread", 16)

    def initialize(self):
        logger.info("纸上交易模式初始化...")
        return self._live_data_source.initialize()

    def shutdown(self):
        logger.info("纸上交易模式关闭.")
        self._live_data_source.shutdown()

    def get_current_price(self, symbol):
        price_data = self._live_data_source.get_current_price(symbol)
        if price_data:
            spread_half = self.spread * 0.01 / 2
            mid_price = price_data['last']
            return {
                'bid': mid_price - spread_half,
                'ask': mid_price + spread_half,
                'last': mid_price,
                'time': price_data['time']
            }
        return None

    def get_historical_data(self, symbol, timeframe, count, **kwargs):
        return self._live_data_source.get_historical_data(symbol, timeframe, count, **kwargs)

    def get_symbol_info(self, symbol):
        return self._live_data_source.get_symbol_info(symbol)

    def get_account_info(self):
        return {'equity': self.equity}

    def get_positions(self, symbol):
        return []

    def send_order(self, symbol, order_type, volume, sl=None, tp=None):
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
