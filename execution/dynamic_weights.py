from strategies import ma_cross, rsi, bollinger, mean_reversion, momentum_breakout, macd, kdj, turtle, daily_breakout, wave_theory
from config import DEFAULT_WEIGHTS, SYMBOL, TIMEFRAME
from logger import logger
from core.risk.market_state import MarketStateAnalyzer

class DynamicWeightManager:
    """
    动态权重管理器
    """
    
    def __init__(self, data_provider, market_state_params=None, trend_weights=None, trend_thresholds=None, confidence_thresholds=None):
        self.data_provider = data_provider
        self.market_state_analyzer = MarketStateAnalyzer(
            data_provider, 
            market_state_params=market_state_params,
            trend_weights=trend_weights,
            trend_thresholds=trend_thresholds,
            confidence_thresholds=confidence_thresholds
        )
        
        # 策略类和它们的初始化参数的映射
        self.strategy_blueprints = {
            'ma_cross': (ma_cross.MACrossStrategy, {}),
            'rsi': (rsi.RSIStrategy, {}),
            'bollinger': (bollinger.BollingerStrategy, {}),
            'mean_reversion': (mean_reversion.MeanReversionStrategy, {}),
            'momentum_breakout': (momentum_breakout.MomentumBreakoutStrategy, {}),
            'macd': (macd.MACDStrategy, {}),
            'kdj': (kdj.KDJStrategy, {}),
            'turtle': (turtle.TurtleStrategy, {}),
            'daily_breakout': (daily_breakout.DailyBreakoutStrategy, {}),
            'wave_theory': (wave_theory.WaveTheoryStrategy, {})
        }
        
        self.strategy_instances = self._create_strategy_instances()
        
        self.current_weights = None
        self.current_market_state = "none"
        self.current_confidence = 0.0

    def _create_strategy_instances(self):
        instances = {}
        for name, (strategy_class, params) in self.strategy_blueprints.items():
            instances[name] = strategy_class(self.data_provider, SYMBOL, TIMEFRAME, **params)
        return instances
        
    def get_current_strategies_and_weights(self):
        market_state, confidence = self.market_state_analyzer.get_market_state()
        dynamic_weights = self.market_state_analyzer.get_strategy_weights(market_state, confidence)
        
        self.current_weights = dynamic_weights
        self.current_market_state = market_state
        self.current_confidence = confidence
        
        strategies_with_weights = []
        for name, weight in dynamic_weights.items():
            if name in self.strategy_instances:
                strategies_with_weights.append((self.strategy_instances[name], weight))
        
        return strategies_with_weights
    
    def get_weight_info(self):
        return {
            'market_state': self.current_market_state,
            'confidence': self.current_confidence,
            'weights': self.current_weights
        }