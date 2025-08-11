from .market_state import MarketStateAnalyzer
from .position_manager import PositionManager
from logger import logger
from config import RISK_CONFIG

class RiskController:
    """
    风险管理控制器，统一管理市场状态分析和仓位管理
    """
    
    def __init__(self):
        self.market_state_analyzer = MarketStateAnalyzer()
        self.position_manager = PositionManager()
        
        # 从配置文件读取风险控制参数
        self.max_position_size = RISK_CONFIG.get("max_position_size", 0.1)
        self.max_daily_loss = RISK_CONFIG.get("max_daily_loss", -0.05)
        self.daily_pnl = 0.0         # 日内盈亏
        
    def get_market_state(self):
        """
        获取当前市场状态
        """
        return self.market_state_analyzer.get_market_state()
        
    def get_dynamic_weights(self):
        """
        获取动态策略权重
        """
        market_state, confidence = self.get_market_state()
        logger.info(f"当前市场状态: {market_state}, 置信度: {confidence:.2f}")
        
        weights = self.market_state_analyzer.get_strategy_weights(market_state, confidence)
        logger.info(f"动态权重配置: {weights}")
        
        return weights, market_state, confidence
        
    def check_risk_management(self, current_price):
        """
        检查风险管理条件
        """
        return self.position_manager.check_risk_management(current_price)
        
    def execute_risk_action(self, action, reason):
        """
        执行风险管理操作
        """
        self.position_manager.execute_risk_action(action, reason)
        
    def update_position_entry(self, entry_price, position_type="long"):
        """
        更新持仓入场信息
        """
        self.position_manager.update_position_entry(entry_price, position_type)
        
    def should_allow_trade(self, signal_type):
        """
        根据风险控制决定是否允许交易
        """
        # 检查日亏损限制
        if self.daily_pnl <= self.max_daily_loss:
            logger.warning(f"日亏损已达{self.daily_pnl:.2%}，暂停交易")
            return False
            
        # 检查持仓数量限制
        current_positions = len(self.position_manager.positions)
        if current_positions >= 1:  # 限制同时只持有一个仓位
            return False
            
        return True
        
    def update_daily_pnl(self, pnl_change):
        """
        更新日内盈亏
        """
        self.daily_pnl += pnl_change
        logger.info(f"更新日内盈亏: {self.daily_pnl:.2%}")
        
    def reset_daily_stats(self):
        """
        重置日内统计
        """
        self.daily_pnl = 0.0
        logger.info("重置日内统计")
        
    def reset_all_states(self):
        """
        重置所有状态
        """
        self.market_state_analyzer.reset_state()
        self.position_manager.reset_positions()
        self.reset_daily_stats()
        logger.info("重置所有风险管理状态")