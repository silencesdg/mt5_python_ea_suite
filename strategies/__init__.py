"""策略模块 — 统一注册所有策略到 StrategyRegistry"""
from core.signal.registry import StrategyRegistry
from strategies.ma_cross import MACrossStrategy
from strategies.rsi import RSIStrategy
from strategies.bollinger import BollingerStrategy
from strategies.macd import MACDStrategy
from strategies.kdj import KDJStrategy
from strategies.turtle import TurtleStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.daily_breakout import DailyBreakoutStrategy
from strategies.wave_theory import WaveTheoryStrategy

_registry = StrategyRegistry()

_registry.register('ma_cross', MACrossStrategy)
_registry.register('rsi', RSIStrategy)
_registry.register('bollinger', BollingerStrategy)
_registry.register('macd', MACDStrategy)
_registry.register('kdj', KDJStrategy)
_registry.register('turtle', TurtleStrategy)
_registry.register('mean_reversion', MeanReversionStrategy)
_registry.register('momentum_breakout', MomentumBreakoutStrategy)
_registry.register('daily_breakout', DailyBreakoutStrategy)
_registry.register('wave_theory', WaveTheoryStrategy)
