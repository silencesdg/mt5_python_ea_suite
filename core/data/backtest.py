import numpy as np
from logger import logger
from core.data.abc import DataProvider
from core.data.multi_tf import MultiTimeframeDataStore
from config import SIMULATION_CONFIG


class BacktestDataProvider(DataProvider):
    """回测数据提供者（重写版） — 通过 MultiTimeframeDataStore 支持多周期数据

    关键改进：
    - get_historical_data() 的 timeframe 参数现在真正生效
    - 内部持有 MultiTimeframeDataStore，按需从M1生成更高周期数据
    - current_index 推进时各周期数据指针同步
    """

    def __init__(self, multi_tf_store: MultiTimeframeDataStore,
                 initial_equity: float = 10000):
        self.multi_tf = multi_tf_store
        self.current_index = 0
        self.equity = initial_equity
        self.simulated_ticket_counter = 0
        self.spread = SIMULATION_CONFIG.get("spread", 16)

    @property
    def total_length(self) -> int:
        return self.multi_tf.length

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def get_account_info(self):
        return {'equity': self.equity}

    def get_positions(self, symbol):
        return []

    def get_symbol_info(self, symbol):
        return {
            'trade_contract_size': SIMULATION_CONFIG['contract_size'],
            'volume_step': SIMULATION_CONFIG['volume_step'],
            'volume_min': SIMULATION_CONFIG['volume_min'],
            'volume_max': SIMULATION_CONFIG['volume_max'],
        }

    def get_current_price(self, symbol):
        if self.current_index >= self.total_length:
            return None
        row = self.multi_tf.main_df.iloc[self.current_index]
        spread_half = self.spread * 0.01 / 2
        return {
            'bid': row.close - spread_half,
            'ask': row.close + spread_half,
            'last': row.close,
            'time': row.name
        }

    def get_historical_data(self, symbol, timeframe, count, **kwargs):
        """获取历史数据 — timeframe 参数现在真正生效"""
        return self.multi_tf.get_historical_data(
            symbol, timeframe, count, self.current_index
        )

    def send_order(self, symbol, order_type, volume):
        self.simulated_ticket_counter += 1
        logger.info(f"[回测模式] 下单: {order_type} {volume:.2f}手 {symbol}")
        return {'order': self.simulated_ticket_counter}

    def close_position(self, ticket, symbol, volume):
        logger.info(f"[回测模式] 平仓: Ticket {ticket}")
        return True

    def tick(self) -> bool:
        if self.current_index < self.total_length - 1:
            self.current_index += 1
            return True
        return False
