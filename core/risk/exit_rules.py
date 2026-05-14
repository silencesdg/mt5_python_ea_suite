"""退出规则 — 策略模式，每个规则负责判断是否需要平仓"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExitContext:
    """退出规则所需的上下文"""
    current_profit_pct: float
    peak_profit_pct: float
    entry_time: datetime
    current_time: datetime
    symbol: str = ""
    position: dict = None


class BaseExitRule:
    """退出规则基类"""

    def __init__(self, config: dict):
        self.config = config

    def check(self, ctx: ExitContext) -> tuple[str, str]:
        """返回 (action, reason)，action 为 "close" 或 "none" """
        return "none", ""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StopLossRule(BaseExitRule):
    """固定止损"""

    def check(self, ctx: ExitContext) -> tuple[str, str]:
        if ctx.current_profit_pct <= self.config.get("stop_loss_pct", -0.10):
            return "close", "止损触发"
        return "none", ""


class TakeProfitRule(BaseExitRule):
    """固定止盈"""

    def check(self, ctx: ExitContext) -> tuple[str, str]:
        if ctx.current_profit_pct >= self.config.get("take_profit_pct", 0.20):
            return "close", "止盈触发"
        return "none", ""


class TrailingStopRule(BaseExitRule):
    """追踪止损"""

    def check(self, ctx: ExitContext) -> tuple[str, str]:
        min_profit = self.config.get("min_profit_for_trailing", 0.01)
        retracement_pct = self.config.get("profit_retracement_pct", 0.10)

        if ctx.peak_profit_pct <= min_profit:
            return "none", ""

        # ★ 回撤从峰值绝对值扣除（账户%，非相对%）：峰值+2.0%回撤1.0%→止损在+1.0%
        stop_level = ctx.peak_profit_pct - retracement_pct
        if ctx.current_profit_pct <= stop_level:
            return "close", (
                f"追踪止损触发 "
                f"(峰值 {ctx.peak_profit_pct:.2%} 回落至 {ctx.current_profit_pct:.2%})"
            )
        return "none", ""


class TimeExitRule(BaseExitRule):
    """时间退出"""

    def check(self, ctx: ExitContext) -> tuple[str, str]:
        if not self.config.get("enable_time_based_exit", False):
            return "none", ""

        max_minutes = self.config.get("max_holding_minutes", 60)
        min_profit = self.config.get("min_profit_for_time_exit", 0.001)

        holding_seconds = (ctx.current_time - ctx.entry_time).total_seconds()
        holding_minutes = holding_seconds / 60

        if holding_minutes > max_minutes and ctx.current_profit_pct < min_profit:
            return "close", f"超时平仓 (持仓超过{max_minutes}分钟且盈利未达标)"
        return "none", ""


class ExitRuleEngine:
    """退出规则引擎 — 按顺序执行所有注册的规则，返回第一个触发的退出

    注意：每日亏损上限检查在 PositionManager._check_max_daily_loss 中处理（开仓时），
    不在退出规则中重复，避免两套计算逻辑不一致。
    """

    def __init__(self, config: dict):
        self.rules = [
            StopLossRule(config),
            TakeProfitRule(config),
            TrailingStopRule(config),
            TimeExitRule(config),
        ]

    def check(self, ctx: ExitContext) -> tuple[str, str]:
        for rule in self.rules:
            action, reason = rule.check(ctx)
            if action == "close":
                return action, reason
        return "none", ""
