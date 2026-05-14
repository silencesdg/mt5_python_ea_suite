from core.risk.market_state import MarketStateAnalyzer
from core.risk.position import PositionManager
from logger import logger


class RiskController:
    """风险管理控制器 — 门面模式，组合 PositionManager 和 MarketStateAnalyzer"""

    def __init__(self, data_provider, trade_direction="both",
                 risk_config: dict = None,
                 market_state_analyzer: MarketStateAnalyzer = None):
        self.data_provider = data_provider
        self.position_manager = PositionManager(data_provider, trade_direction, risk_config)
        self.market_state_analyzer = market_state_analyzer or MarketStateAnalyzer(data_provider)
        self.trade_direction = trade_direction

    def process_trading_signal(self, direction, current_price, signal_strength=0.0):
        return self.position_manager.open_position(direction, current_price, signal_strength)

    def monitor_positions(self, current_price, dry_run=False, weighted_signal=0.0):
        self.position_manager.monitor_positions(current_price, dry_run, weighted_signal)

    def sync_state(self):
        self.position_manager.update_equity()
        self.position_manager.sync_positions()

    def get_account_status(self):
        return {
            'equity': self.position_manager.total_equity,
            'open_positions': len(self.position_manager.positions),
            'trade_summary': self.position_manager.get_trade_summary(),
        }

    def get_positions(self):
        return self.position_manager.positions

    def save_trade_history(self, base_filename):
        self.position_manager.save_trade_history(base_filename)
