import pandas as pd
from logger import logger
from utils import has_open_position, close_all
from config import RISK_CONFIG, SYMBOL

class PositionManager:
    """
    仓位管理器，基于profit_protect逻辑
    处理止损、止盈、追踪止损等风险管理
    """
    
    def __init__(self):
        self.symbol = SYMBOL
        
        # 从配置文件读取风险管理参数
        self.stop_loss_pct = RISK_CONFIG.get("stop_loss_pct", -0.10)
        self.profit_retracement_pct = RISK_CONFIG.get("profit_retracement_pct", 0.30)
        self.min_profit_for_trailing = RISK_CONFIG.get("min_profit_for_trailing", 0.05)
        self.take_profit_pct = RISK_CONFIG.get("take_profit_pct", 0.20)
        
        # 持仓状态
        self.positions = {}  # symbol: {entry_price, entry_time, peak_profit}
        
    def check_risk_management(self, current_price):
        """
        检查是否需要触发风险管理操作
        返回: action ("close_long", "close_short", "none"), reason
        """
        # 在回测环境中，使用self.positions来判断持仓状态
        # 而不是使用has_open_position()（该函数依赖MT5实时数据）
        position_info = self.positions.get(self.symbol)
        if not position_info:
            return "none", "no_position"
            
        entry_price = position_info['entry_price']
        current_profit_pct = (current_price - entry_price) / entry_price
        
        # 更新最高利润
        self.positions[self.symbol]['peak_profit'] = max(
            position_info.get('peak_profit', 0), 
            current_profit_pct
        )
        peak_profit = self.positions[self.symbol]['peak_profit']
        
        # 检查止损条件 - 仅基于买入成本判断亏损，不考虑盈利回撤
        if current_profit_pct < 0 and current_profit_pct <= self.stop_loss_pct:
            reason = f"止损触发：当前亏损{current_profit_pct:.2%}"
            action = "close_long" if current_profit_pct < 0 else "close_short"
            return action, reason
            
        # 检查固定止盈条件
        if current_profit_pct >= self.take_profit_pct:
            reason = f"止盈触发：当前盈利{current_profit_pct:.2%}"
            action = "close_long" if current_profit_pct > 0 else "close_short"
            return action, reason
            
        # 检查追踪止损条件 - 只有在利润达到min_profit_for_trailing以上才激活
        if peak_profit > self.min_profit_for_trailing:
            retracement_from_peak = peak_profit - current_profit_pct
            if peak_profit > 0:
                retracement_pct = retracement_from_peak / peak_profit
                if retracement_pct >= self.profit_retracement_pct:
                    reason = f"追踪止损：最高盈利{peak_profit:.2%}，回撤{retracement_pct:.2%}"
                    action = "close_long" if current_profit_pct > 0 else "close_short"
                    return action, reason
        
        return "none", "no_action_needed"
    
    def execute_risk_action(self, action, reason):
        """
        执行风险管理操作
        """
        if action == "close_long":
            logger.info(f"执行平多仓：{reason}")
            close_all(self.symbol)
            self.positions.pop(self.symbol, None)
        elif action == "close_short":
            logger.info(f"执行平空仓：{reason}")
            close_all(self.symbol)
            self.positions.pop(self.symbol, None)
            
    def update_position_entry(self, entry_price, position_type="long"):
        """
        更新持仓入场信息
        """
        self.positions[self.symbol] = {
            'entry_price': entry_price,
            'entry_time': pd.Timestamp.now(),
            'position_type': position_type,
            'peak_profit': 0.0
        }
        logger.info(f"记录持仓入场：价格={entry_price:.2f}, 类型={position_type}")
        
    def get_position_info(self):
        """
        获取当前持仓信息
        """
        return self.positions.get(self.symbol, None)
        
    def reset_positions(self):
        """
        重置所有持仓状态
        """
        self.positions.clear()
        logger.info("重置所有持仓状态")