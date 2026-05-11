from abc import ABC, abstractmethod


class DataProvider(ABC):
    """数据提供者抽象基类 — 定义所有数据访问的统一接口"""

    @property
    def is_live(self):
        return False

    @abstractmethod
    def initialize(self):
        """初始化连接"""

    @abstractmethod
    def shutdown(self):
        """关闭连接"""

    @abstractmethod
    def get_current_price(self, symbol):
        """获取当前价格: {'bid', 'ask', 'last', 'time'}"""

    @abstractmethod
    def get_historical_data(self, symbol, timeframe, count, **kwargs):
        """获取历史K线数据"""

    @abstractmethod
    def get_account_info(self):
        """获取账户信息"""

    @abstractmethod
    def get_positions(self, symbol):
        """获取当前持仓"""

    @abstractmethod
    def get_symbol_info(self, symbol):
        """获取品种信息（合约规格等）"""

    @abstractmethod
    def send_order(self, symbol, order_type, volume):
        """发送订单"""

    @abstractmethod
    def close_position(self, ticket, symbol, volume):
        """平仓"""
