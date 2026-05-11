"""动态权重管理器（重写版）

关键改进：
  1. 使用 StrategyRegistry（单例）获取策略列表，消除 blueprint 重复
  2. 回测/优化器模式支持逐 bar 获取权重（get_weights_for_bar）
  3. 优化器模式接受 individual_weights 参数并传递给 MarketStateAnalyzer
"""

import strategies  # noqa: F401 — 触发策略注册
from logger import logger
from core.risk.market_state import MarketStateAnalyzer
from core.signal.registry import StrategyRegistry
from config import SYMBOL, TIMEFRAME


class DynamicWeightManager:
    """动态权重管理器"""

    def __init__(self, data_provider,
                 market_state_analyzer: MarketStateAnalyzer = None):
        self.data_provider = data_provider
        self.analyzer = market_state_analyzer or MarketStateAnalyzer(data_provider)
        self.registry = StrategyRegistry()

        # 初始化策略实例（仅实盘模式首次使用）
        self._strategies_initialized = False

    def _ensure_strategies(self):
        if not self._strategies_initialized:
            self.registry.instantiate_all(SYMBOL, TIMEFRAME,
                                          data_provider=self.data_provider)
            self._strategies_initialized = True

    # ── 实盘模式 ──

    def get_current_strategies_and_weights(self) -> list:
        """获取当前（实时）策略实例和权重列表"""
        self._ensure_strategies()
        weights = self.get_current_weights()

        result = []
        for config_key, weight in weights.items():
            instance = self.registry.get_instance(config_key)
            if instance is not None:
                result.append((instance, weight))
        return result

    def get_current_weights(self) -> dict:
        """获取当前实时权重"""
        market_state, confidence = self.analyzer.get_market_state()
        return self.analyzer.get_strategy_weights(market_state, confidence)

    # ── 回测/优化器模式 ──

    def get_weights_for_bar(self, bar_index: int,
                           individual_weights: dict = None) -> dict:
        """获取第 bar_index 个 bar 的策略权重

        Args:
            bar_index: 当前M1 bar索引
            individual_weights: 优化器传入 {config_key: weight}，非None时直接使用

        Returns:
            {config_key: weight}
        """
        market_state, confidence = self.analyzer.get_market_state(bar_index)
        return self.analyzer.get_strategy_weights(
            market_state, confidence, individual_weights
        )

    def get_weight_info(self) -> dict:
        market_state, confidence = self.analyzer.get_market_state()
        market_weights = self.analyzer.get_strategy_weights(market_state, confidence)
        return {
            'market_state': market_state,
            'confidence': confidence,
            'weights': market_weights
        }
