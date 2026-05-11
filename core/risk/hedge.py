"""
对冲管理器 — 信号对冲 + 回撤锁仓

两种对冲模式：
  1. 信号对冲：持仓方向与当前加权信号相反 → 开反向单保护
  2. 回撤锁仓：持仓亏损超过阈值 → 锁仓防进一步亏损

解锁条件：
  - 信号回到中性/同向 → 平对冲单
  - 对冲单自身止盈 → 平对冲单
  - 原始仓位平仓 → 同时平对冲单
"""

from logger import logger


class HedgeManager:
    """对冲管理器 — 与 PositionManager 协作，不直接操作仓位"""

    def __init__(self, position_manager, config: dict = None):
        self._pm = position_manager  # PositionManager 引用
        cfg = config or {}

        # ── 信号对冲 ──
        self.signal_hedge_enabled = cfg.get("signal_hedge_enabled", True)
        # 加权信号绝对值超过此阈值才触发对冲（避免噪音）
        self.signal_hedge_threshold = cfg.get("signal_hedge_threshold", 2.0)
        # 对冲比例：0.5=开一半手数，1.0=等量对冲
        self.signal_hedge_ratio = cfg.get("signal_hedge_ratio", 0.5)
        # 信号回到中性以下（绝对值<此值）时解锁
        self.signal_unhedge_threshold = cfg.get("signal_unhedge_threshold", 1.0)

        # ── 回撤锁仓 ──
        self.drawdown_hedge_enabled = cfg.get("drawdown_hedge_enabled", True)
        # 浮亏超过此比例触发锁仓
        self.drawdown_hedge_pct = cfg.get("drawdown_hedge_pct", -0.003)  # -0.3%
        # 锁仓比例
        self.drawdown_hedge_ratio = cfg.get("drawdown_hedge_ratio", 1.0)  # 100%

        # ── 对冲单元管理 ──
        # 每个对冲单元: {original_ticket, hedge_ticket, hedge_type, hedge_volume}
        self._hedge_units = []

        # ── 对冲单止盈 ──
        self.hedge_take_profit_pct = cfg.get("hedge_take_profit_pct", 0.005)  # 0.5%

        # ── 每日最大对冲次数 ──
        self.max_hedges_per_day = cfg.get("max_hedges_per_day", 5)
        self._daily_hedge_count = 0

    # ═══════════════════════════════════════════════════════════
    # 决策接口
    # ═══════════════════════════════════════════════════════════

    def evaluate(self, current_signal: float, current_price: dict) -> list:
        """
        每轮主循环调用：评估所有持仓是否需要对冲/解锁。
        返回需要执行的操作列表: [("hedge", position, reason), ("unhedge", unit, reason), ...]
        """
        actions = []

        # ① 先检查是否需要解锁已有对冲
        for unit in list(self._hedge_units):
            original = self._find_position(unit["original_ticket"])
            if original is None:
                # 原始仓位已平 → 解锁对冲
                actions.append(("unhedge", unit, "原始仓位已平仓"))
                continue

            if self._should_unhedge(unit, original, current_signal, current_price):
                actions.append(("unhedge", unit, unit.get("unhedge_reason", "解锁条件满足")))

        # ② 再检查是否需要开新对冲
        for pos in self._pm.positions:
            ticket = pos["ticket"]

            # 已有对冲的跳过
            if any(u["original_ticket"] == ticket for u in self._hedge_units):
                continue

            # 信号对冲
            if self.signal_hedge_enabled:
                reason = self._check_signal_hedge(pos, current_signal)
                if reason:
                    if self._can_hedge():
                        actions.append(("hedge", pos, reason))

            # 回撤锁仓
            if self.drawdown_hedge_enabled:
                reason = self._check_drawdown_hedge(pos, current_price)
                if reason:
                    if self._can_hedge():
                        actions.append(("hedge", pos, reason))

        return actions

    # ═══════════════════════════════════════════════════════════
    # 对冲执行
    # ═══════════════════════════════════════════════════════════

    def execute_hedge(self, position, reason: str, current_price: dict) -> bool:
        """对指定持仓开反向对冲单"""
        opposite = "sell" if position["position_type"] == "long" else "buy"

        # 确定对冲手数
        hedge_type = "signal" if "信号" in reason else "drawdown"
        ratio = self.signal_hedge_ratio if hedge_type == "signal" else self.drawdown_hedge_ratio
        hedge_volume = position["quantity"] * ratio

        # ★ 手数校验：不能低于最小手数
        symbol_info = self._pm.data_provider.get_symbol_info(self._pm.symbol)
        if symbol_info:
            vol_min = symbol_info.get("volume_min", 0.01) if isinstance(symbol_info, dict) else getattr(symbol_info, "volume_min", 0.01)
            vol_step = symbol_info.get("volume_step", 0.01) if isinstance(symbol_info, dict) else getattr(symbol_info, "volume_step", 0.01)
            hedge_volume = max(vol_min, round(hedge_volume / vol_step) * vol_step)
        else:
            hedge_volume = max(0.01, hedge_volume)

        logger.info(f"🔒 准备对冲: {opposite} {hedge_volume:.2f}手 | 原因: {reason}")

        # 调取开仓
        result = self._pm.data_provider.send_order(
            self._pm.symbol, opposite, hedge_volume
        )
        if result is None:
            logger.error(f"对冲开仓失败: {opposite} @ {hedge_volume}手")
            return False

        try:
            order_id = result["order"] if isinstance(result, dict) else result.order
        except Exception as e:
            logger.error(f"解析对冲单号失败: {e}")
            return False

        if order_id > 0:
            unit = {
                "original_ticket": position["ticket"],
                "hedge_ticket": order_id,
                "hedge_type": hedge_type,
                "hedge_volume": hedge_volume,
                "hedge_direction": opposite,
                "entry_price": current_price.get("last", 0),
                "entry_time": current_price.get("time", None),
            }
            self._hedge_units.append(unit)
            self._daily_hedge_count += 1
            logger.info(
                f"🔒 对冲开仓: Ticket {position['ticket']} → 反向 {opposite} "
                f"@{hedge_volume}手 (Ticket {order_id}) | 原因: {reason}"
            )
            return True

        logger.error(f"对冲开仓失败: order_id={order_id}")
        return False

    def execute_unhedge(self, unit: dict, reason: str) -> bool:
        """平掉对冲单，解锁"""
        success = self._pm.data_provider.close_position(
            unit["hedge_ticket"], self._pm.symbol, unit["hedge_volume"]
        )
        if success:
            self._hedge_units.remove(unit)
            logger.info(
                f"🔓 对冲解锁: Ticket {unit['hedge_ticket']} 已平仓 | 原因: {reason}"
            )
            return True
        logger.error(f"解锁平仓失败: Ticket {unit['hedge_ticket']}")
        return False

    # ═══════════════════════════════════════════════════════════
    # 内部检查
    # ═══════════════════════════════════════════════════════════

    def _check_signal_hedge(self, position, current_signal: float) -> str | None:
        """检查信号是否触发对冲"""
        is_long = position["position_type"] == "long"
        # 持多 + 信号强烈看空 → 对冲
        if is_long and current_signal < -self.signal_hedge_threshold:
            return f"信号对冲(持多, 加权={current_signal:.2f} < -{self.signal_hedge_threshold})"
        # 持空 + 信号强烈看多 → 对冲
        if not is_long and current_signal > self.signal_hedge_threshold:
            return f"信号对冲(持空, 加权={current_signal:.2f} > {self.signal_hedge_threshold})"
        return None

    def _check_drawdown_hedge(self, position, current_price: dict) -> str | None:
        """检查浮亏是否触发锁仓"""
        price = current_price.get("last", 0)
        if price <= 0:
            return None
        pnl_pct = self._pm._calculate_pnl_pct(position, price)
        if pnl_pct <= self.drawdown_hedge_pct:
            return f"回撤锁仓(浮亏={pnl_pct:.2%} <= {self.drawdown_hedge_pct:.2%})"
        return None

    def _should_unhedge(self, unit, original_pos, current_signal: float, current_price: dict) -> bool:
        """判断是否应该解锁对冲"""
        # 条件1：信号回到中性
        if abs(current_signal) < self.signal_unhedge_threshold:
            unit["unhedge_reason"] = f"信号中性({current_signal:.2f})"
            return True

        # 条件2：信号与原始仓位同向
        is_long = original_pos["position_type"] == "long"
        if is_long and current_signal > 0:
            unit["unhedge_reason"] = f"信号同向做多({current_signal:.2f})"
            return True
        if not is_long and current_signal < 0:
            unit["unhedge_reason"] = f"信号同向做空({current_signal:.2f})"
            return True

        # 条件3：对冲单自身止盈
        price = current_price.get("last", 0)
        if price > 0 and unit.get("entry_price", 0) > 0:
            if unit["hedge_direction"] == "sell":
                hedge_pnl = (unit["entry_price"] - price) / unit["entry_price"]
            else:
                hedge_pnl = (price - unit["entry_price"]) / unit["entry_price"]
            if hedge_pnl >= self.hedge_take_profit_pct:
                unit["unhedge_reason"] = f"对冲单止盈({hedge_pnl:.2%})"
                return True

        return False

    def _can_hedge(self) -> bool:
        """检查是否允许开新对冲（次数限制）"""
        if self.max_hedges_per_day <= 0:
            return True
        return self._daily_hedge_count < self.max_hedges_per_day

    def _find_position(self, ticket):
        """在持仓列表中查找指定 ticket 的仓位"""
        return next((p for p in self._pm.positions if p["ticket"] == ticket), None)

    # ═══════════════════════════════════════════════════════════
    # 状态查询
    # ═══════════════════════════════════════════════════════════

    @property
    def active_hedges(self) -> int:
        return len(self._hedge_units)

    def get_hedge_summary(self) -> dict:
        return {
            "active_hedges": len(self._hedge_units),
            "daily_hedge_count": self._daily_hedge_count,
            "units": [
                {
                    "original": u["original_ticket"],
                    "hedge": u["hedge_ticket"],
                    "type": u["hedge_type"],
                    "volume": u["hedge_volume"],
                }
                for u in self._hedge_units
            ],
        }
