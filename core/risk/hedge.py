"""
对冲管理器 — 信号对冲 + 专业锁仓管理

两种对冲模式：
  1. 信号对冲：持仓方向与当前加权信号相反 → 开反向单保护
  2. 回撤锁仓：持仓亏损超过阈值 → 锁仓防进一步亏损

锁仓解锁策略 (Lock & Walk + Trend Confirmation):
  - 锁仓组合净盈利 > 0 → 双平离场（保本）
  - 趋势确认向上(信号>1.0) → 平空留多
  - 趋势确认向下(信号<-1.0) → 平多留空（顺势反手）
  - 趋势不明 → 继续锁仓（避免两头挨打）
"""

from logger import logger


class HedgeManager:
    """对冲管理器 — 与 PositionManager 协作，不直接操作仓位"""

    def __init__(self, position_manager, config: dict = None):
        self._pm = position_manager  # PositionManager 引用
        cfg = config or {}

        # ── 信号对冲 ──
        self.signal_hedge_enabled = cfg.get("signal_hedge_enabled", True)
        self.signal_hedge_threshold = cfg.get("signal_hedge_threshold", 2.0)
        self.signal_hedge_ratio = cfg.get("signal_hedge_ratio", 0.5)
        self.signal_unhedge_threshold = cfg.get("signal_unhedge_threshold", 1.0)

        # ── 回撤锁仓 ──
        self.drawdown_hedge_enabled = cfg.get("drawdown_hedge_enabled", True)
        self.drawdown_hedge_pct = cfg.get("drawdown_hedge_pct", -0.003)
        self.drawdown_hedge_ratio = cfg.get("drawdown_hedge_ratio", 1.0)

        # ── 锁仓管理（新） ──
        # 锁仓组合盈亏阈值：净盈超过此比例→双平离场
        self.lock_net_profit_pct = cfg.get("lock_net_profit_pct", 0.0)  # 打平即可
        # 趋势确认信号阈值（方向确认用，复用 signal_unhedge_threshold）

        # ── 对冲单元管理 ──
        self._hedge_units = []

        # ── 对冲单止盈 ──
        self.hedge_take_profit_pct = cfg.get("hedge_take_profit_pct", 0.005)

        # ── 每日最大对冲次数 ──
        self.max_hedges_per_day = cfg.get("max_hedges_per_day", 5)
        self._daily_hedge_count = 0

    # ═══════════════════════════════════════════════════════════
    # 决策接口
    # ═══════════════════════════════════════════════════════════

    def evaluate(self, current_signal: float, current_price: dict) -> list:
        """
        每轮主循环调用。
        返回操作列表: action 类型见下
          ("hedge", position, reason)       — 开新对冲
          ("unhedge_only", unit, reason)    — 只平对冲单，保留原始
          ("close_original", unit, reason)  — 平原始单，保留对冲（顺势反手）
          ("close_both", unit, reason)      — 双平离场
          ("hold", unit, reason)            — 继续锁仓（仅日志用途，不执行）
        """
        actions = []

        # ① 锁仓决策
        for unit in list(self._hedge_units):
            original = self._find_position(unit["original_ticket"])
            if original is None:
                actions.append(("close_both", unit, "原始仓位已消失"))
                continue

            action, reason = self._resolve_lock(unit, original, current_signal, current_price)
            if action != "hold":
                actions.append((action, unit, reason))

        # ② 开新对冲
        for pos in self._pm.positions:
            ticket = pos["ticket"]

            if any(u["original_ticket"] == ticket for u in self._hedge_units):
                continue

            hedge_reason = None
            if self.drawdown_hedge_enabled:
                hedge_reason = self._check_drawdown_hedge(pos, current_price)
            if not hedge_reason and self.signal_hedge_enabled:
                hedge_reason = self._check_signal_hedge(pos, current_signal)

            if hedge_reason and self._can_hedge():
                actions.append(("hedge", pos, hedge_reason))

        return actions

    # ═══════════════════════════════════════════════════════════
    # 锁仓决策核心（Lock & Walk + Trend Confirmation）
    # ═══════════════════════════════════════════════════════════

    def _resolve_lock(self, unit, original_pos, current_signal: float, current_price: dict):
        """
        专业锁仓决策：
        1. 锁仓组合整体盈利 → 双平离场
        2. 趋势确认方向 → 跟趋势走，平亏损方留盈利方
        3. 趋势不明 → 继续锁仓
        """
        price = current_price.get("last", 0)
        if price <= 0:
            return ("hold", "价格无效")

        # 计算双方盈亏
        original_pnl = self._pm._calculate_pnl_pct(original_pos, price)
        hedge_price = unit.get("entry_price", 0)
        if hedge_price <= 0:
            return ("close_both", "对冲单价格异常")

        if unit["hedge_direction"] == "sell":
            hedge_pnl = (hedge_price - price) / hedge_price
        else:
            hedge_pnl = (price - hedge_price) / hedge_price

        net_pnl = original_pnl + hedge_pnl

        # ① 锁仓组合净盈利 → 双平（绝不亏钱离场）
        if net_pnl >= self.lock_net_profit_pct:
            return ("close_both",
                    f"组合盈利(原={original_pnl:+.2%}+对={hedge_pnl:+.2%}={net_pnl:+.2%})")

        # ② 趋势确认 → 跟趋势
        is_long = original_pos["position_type"] == "long"
        thresh = self.signal_unhedge_threshold

        if current_signal > thresh:
            # 信号看多
            if is_long:
                return ("unhedge_only", f"趋势看多(信号={current_signal:.2f})→平空留多")
            else:
                return ("close_original", f"趋势看多(信号={current_signal:.2f})→平空留多顺势")

        if current_signal < -thresh:
            # 信号看空
            if is_long:
                return ("close_original", f"趋势看空(信号={current_signal:.2f})→平多留空顺势")
            else:
                return ("unhedge_only", f"趋势看空(信号={current_signal:.2f})→平多留空")

        # ③ 对冲单止盈（即使趋势不明，对冲单赚够了也解锁）
        if hedge_pnl >= self.hedge_take_profit_pct:
            return ("unhedge_only", f"对冲单止盈({hedge_pnl:.2%}≥{self.hedge_take_profit_pct:.1%})")

        # ④ 趋势不明 → 继续锁
        return ("hold", f"趋势不明(信号={current_signal:.2f})→继续锁仓")

    # ═══════════════════════════════════════════════════════════
    # 执行方法
    # ═══════════════════════════════════════════════════════════

    def execute_hedge(self, position, reason: str, current_price: dict) -> bool:
        """开反向对冲单"""
        opposite = "sell" if position["position_type"] == "long" else "buy"
        hedge_type = "signal" if "信号" in reason else "drawdown"
        ratio = self.signal_hedge_ratio if hedge_type == "signal" else self.drawdown_hedge_ratio
        hedge_volume = position["quantity"] * ratio

        # 手数校验
        symbol_info = self._pm.data_provider.get_symbol_info(self._pm.symbol)
        if symbol_info:
            vol_min = symbol_info.get("volume_min", 0.01) if isinstance(symbol_info, dict) else getattr(symbol_info, "volume_min", 0.01)
            vol_step = symbol_info.get("volume_step", 0.01) if isinstance(symbol_info, dict) else getattr(symbol_info, "volume_step", 0.01)
            hedge_volume = max(vol_min, round(hedge_volume / vol_step) * vol_step)
        else:
            hedge_volume = max(0.01, hedge_volume)

        logger.info(f"🔒 准备对冲: {opposite} {hedge_volume:.2f}手 | 原因: {reason}")

        result = self._pm.data_provider.send_order(self._pm.symbol, opposite, hedge_volume)
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
            logger.info(f"🔒 对冲开仓: Ticket {position['ticket']} → {opposite} @{hedge_volume}手 (Ticket {order_id})")
            return True

        logger.error(f"对冲开仓失败: order_id={order_id}")
        return False

    def execute_unhedge_only(self, unit: dict, reason: str) -> bool:
        """只平对冲单，保留原始仓位"""
        return self._close_hedge_position(unit, reason)

    def execute_close_original(self, unit: dict, reason: str) -> bool:
        """平原始仓位，保留对冲单（顺势反手）"""
        success = self._pm.data_provider.close_position(
            unit["original_ticket"], self._pm.symbol, unit["hedge_volume"]
        )
        if success:
            self._hedge_units.remove(unit)
            logger.info(f"🔄 顺势反手: 平原始 Ticket {unit['original_ticket']}，留对冲 Ticket {unit['hedge_ticket']} | {reason}")
            return True
        logger.error(f"反手平仓失败: Ticket {unit['original_ticket']}")
        return False

    def execute_close_both(self, unit: dict, reason: str) -> bool:
        """双平离场"""
        ok1 = self._pm.data_provider.close_position(
            unit["original_ticket"], self._pm.symbol, unit["hedge_volume"]
        )
        ok2 = self._pm.data_provider.close_position(
            unit["hedge_ticket"], self._pm.symbol, unit["hedge_volume"]
        )
        if ok1 and ok2:
            self._hedge_units.remove(unit)
            logger.info(f"✅ 双平离场: Ticket {unit['original_ticket']}+{unit['hedge_ticket']} 已平 | {reason}")
            return True
        logger.error(f"双平失败: 原={ok1} 对={ok2}")
        return False

    def _close_hedge_position(self, unit: dict, reason: str) -> bool:
        """平对冲单（保留原始）"""
        success = self._pm.data_provider.close_position(
            unit["hedge_ticket"], self._pm.symbol, unit["hedge_volume"]
        )
        if success:
            self._hedge_units.remove(unit)
            logger.info(f"🔓 解锁(留原始): Ticket {unit['hedge_ticket']} 已平 | {reason}")
            return True
        logger.error(f"解锁失败: Ticket {unit['hedge_ticket']}")
        return False

    # ═══════════════════════════════════════════════════════════
    # 内部检查
    # ═══════════════════════════════════════════════════════════

    def _check_signal_hedge(self, position, current_signal: float) -> str | None:
        is_long = position["position_type"] == "long"
        if is_long and current_signal < -self.signal_hedge_threshold:
            return f"信号对冲(持多, 加权={current_signal:.2f} < -{self.signal_hedge_threshold})"
        if not is_long and current_signal > self.signal_hedge_threshold:
            return f"信号对冲(持空, 加权={current_signal:.2f} > {self.signal_hedge_threshold})"
        return None

    def _check_drawdown_hedge(self, position, current_price: dict) -> str | None:
        price = current_price.get("last", 0)
        if price <= 0:
            return None
        pnl_pct = self._pm._calculate_pnl_pct(position, price)
        if pnl_pct <= self.drawdown_hedge_pct:
            return f"回撤锁仓(浮亏={pnl_pct:.2%} <= {self.drawdown_hedge_pct:.2%})"
        return None

    def _can_hedge(self) -> bool:
        if self.max_hedges_per_day <= 0:
            return True
        return self._daily_hedge_count < self.max_hedges_per_day

    def _find_position(self, ticket):
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
                {"original": u["original_ticket"], "hedge": u["hedge_ticket"],
                 "type": u["hedge_type"], "volume": u["hedge_volume"]}
                for u in self._hedge_units
            ],
        }
