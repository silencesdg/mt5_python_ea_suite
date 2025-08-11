from strategies import ma_cross, rsi, bollinger, mean_reversion, momentum_breakout, macd, kdj, turtle, daily_breakout, wave_theory, risk_management
from config import DEFAULT_WEIGHTS
from logger import logger
class DynamicWeightManager:
    """
    动态权重管理器，根据风险管理器的建议动态调整策略权重
    """
    
    def __init__(self, risk_controller):
        self.risk_controller = risk_controller
        
        # 策略实例映射
        self.strategy_instances = {
            'ma_cross': ma_cross.Strategy(),
            'rsi': rsi.Strategy(),
            'bollinger': bollinger.Strategy(),
            'mean_reversion': mean_reversion.Strategy(),
            'momentum_breakout': momentum_breakout.Strategy(),
            'macd': macd.Strategy(),
            'kdj': kdj.Strategy(),
            'turtle': turtle.Strategy(),
            'daily_breakout': daily_breakout.Strategy(),
            'wave_theory': wave_theory.Strategy(),
            'risk_management': risk_management.Strategy()
        }
        
        # 当前权重配置
        self.current_weights = None
        self.current_market_state = "none"
        self.current_confidence = 0.0
        
    def get_current_strategies_and_weights(self):
        """
        获取当前策略实例和权重配置
        返回: [(strategy_instance, weight), ...]
        """
        # 获取动态权重
        dynamic_weights, market_state, confidence = self.risk_controller.get_dynamic_weights()
        
        # 更新当前状态
        self.current_weights = dynamic_weights
        self.current_market_state = market_state
        self.current_confidence = confidence
        
        # 构建策略列表
        strategies_with_weights = []
        for strategy_name, weight in dynamic_weights.items():
            if strategy_name in self.strategy_instances:
                strategies_with_weights.append(
                    (self.strategy_instances[strategy_name], weight)
                )
        
        return strategies_with_weights
    
    def get_weight_info(self):
        """
        获取当前权重配置信息
        """
        return {
            'market_state': self.current_market_state,
            'confidence': self.current_confidence,
            'weights': self.current_weights
        }
    
    def get_strategy_instance(self, strategy_name):
        """
        获取指定策略的实例
        """
        return self.strategy_instances.get(strategy_name)
    
    def add_custom_strategy(self, strategy_name, strategy_instance, base_weight=1.0):
        """
        添加自定义策略
        """
        self.strategy_instances[strategy_name] = strategy_instance
        logger.info(f"添加自定义策略: {strategy_name}, 基础权重: {base_weight}")
    
    def remove_strategy(self, strategy_name):
        """
        移除策略
        """
        if strategy_name in self.strategy_instances:
            del self.strategy_instances[strategy_name]
            logger.info(f"移除策略: {strategy_name}")
    
    def list_available_strategies(self):
        """
        列出所有可用策略
        """
        return list(self.strategy_instances.keys())