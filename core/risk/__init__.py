from .market_state import MarketStateAnalyzer
from .position_manager import PositionManager

class RiskController:
    """
    风险管理控制器 (已重构为依赖注入)
    """
    
    def __init__(self, data_provider, trade_direction="both"):
        self.data_provider = data_provider
        self.position_manager = PositionManager(data_provider, trade_direction)
        self.market_state_analyzer = MarketStateAnalyzer(data_provider)
        
    def process_trading_signal(self, direction, current_price, signal_strength=0.0):
        return self.position_manager.open_position(direction, current_price, signal_strength)
    
    def monitor_positions(self, current_price, dry_run=False):
        self.position_manager.monitor_positions(current_price, dry_run)

    def sync_state(self):
        """从数据源同步权益和持仓"""
        self.position_manager.update_equity()
        self.position_manager.sync_positions()

    def get_account_status(self):
        return self.position_manager.get_account_status()

    def get_positions(self):
        return self.position_manager.get_positions()

    def save_trade_history(self, base_filename):
        self.position_manager.save_trade_history(base_filename)
