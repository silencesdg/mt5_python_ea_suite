import pandas as pd
import numpy as np
import json
import os
from logger import logger
from config import (
    RISK_CONFIG, SYMBOL, INITIAL_CAPITAL, CAPITAL_ALLOCATION,
    RISK_CONFIG_CONST, SIMULATION_CONFIG
)
from core.risk.exit_rules import ExitRuleEngine, ExitContext
from core.risk.hedge import HedgeManager


class PositionManager:
    """持仓管理器（重写版）

    关键改进：
      1. update_equity() 包含浮盈浮亏
      2. max_daily_loss 真正落地生效
      3. 退出规则委托给 ExitRuleEngine（策略模式）
    """

    def __init__(self, data_provider, trade_direction="both", risk_config: dict = None,
                 persist_peaks: bool = True):
        self.data_provider = data_provider
        self.symbol = SYMBOL
        self.trade_direction = trade_direction
        self._persist_peaks = persist_peaks

        # 风险管理参数
        self._risk_config = risk_config or RISK_CONFIG
        risk = self._risk_config
        self.stop_loss_pct = risk.get("stop_loss_pct", -0.10)
        self.profit_retracement_pct = risk.get("profit_retracement_pct", 0.10)
        self.min_profit_for_trailing = risk.get("min_profit_for_trailing", 0.01)
        self.take_profit_pct = risk.get("take_profit_pct", 0.20)
        self.max_holding_minutes = risk.get("max_holding_minutes", 60)
        self.min_profit_for_time_exit = risk.get("min_profit_for_time_exit", 0.001)
        self.enable_time_based_exit = RISK_CONFIG_CONST.get("enable_time_based_exit", False)
        self.max_daily_loss = risk.get("max_daily_loss", -0.30)

        # 资金管理
        self.initial_capital = INITIAL_CAPITAL
        self.long_capital_pct = CAPITAL_ALLOCATION.get("long_pct", 0.5)
        self.short_capital_pct = CAPITAL_ALLOCATION.get("short_pct", 0.5)

        # 持仓和交易记录
        self.positions = []
        self.closed_trades = []
        self.total_equity = self.initial_capital

        # 冷却期：平仓后N根K线内不重新开仓（防反复进出）
        self.cooldown_bars = risk.get("cooldown_bars", 30)
        self._cooldown_counter = 0

        # 退出规则引擎
        exit_config = {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "min_profit_for_trailing": self.min_profit_for_trailing,
            "profit_retracement_pct": self.profit_retracement_pct,
            "enable_time_based_exit": self.enable_time_based_exit,
            "max_holding_minutes": self.max_holding_minutes,
            "min_profit_for_time_exit": self.min_profit_for_time_exit,
            "max_daily_loss": self.max_daily_loss,
        }
        self.exit_engine = ExitRuleEngine(exit_config)

        # 峰值数据持久化（优化器中禁用文件I/O避免多进程竞争）
        self.peak_data_file = "position_peaks.json"
        if self._persist_peaks:
            self._load_peak_data()

        # 对冲管理器
        from config import HEDGE_CONFIG
        self.hedge_manager = HedgeManager(self, HEDGE_CONFIG)

    # ── 仓位计算 ──

    def _calculate_position_size(self, capital_to_allocate, current_price):
        price = current_price['last']
        if not isinstance(price, (int, float)) or price == 0:
            logger.error(f"价格无效: {price}")
            return 0.0

        symbol_info = self.data_provider.get_symbol_info(self.symbol)
        if not symbol_info:
            logger.error(f"无法获取 {self.symbol} 的合约信息")
            return 0.0

        is_dict = isinstance(symbol_info, dict)
        contract_size = symbol_info['trade_contract_size'] if is_dict else symbol_info.trade_contract_size
        volume_step = symbol_info['volume_step'] if is_dict else symbol_info.volume_step
        min_volume = symbol_info['volume_min'] if is_dict else symbol_info.volume_min
        max_volume = symbol_info['volume_max'] if is_dict else symbol_info.volume_max

        value_of_one_lot = price * contract_size
        if value_of_one_lot == 0:
            return 0.0

        volume = capital_to_allocate / value_of_one_lot
        volume = round(volume / volume_step) * volume_step
        volume = max(min_volume, min(volume, max_volume))
        return volume

    # ── 开仓 ──

    def open_position(self, direction, current_price, signal_strength=0.0, dry_run=False):
        # 冷却期检查
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return False

        position_type = 'long' if direction == 'buy' else 'short'

        # 开仓前更新权益，确保仓位大小基于最新资产（含浮盈浮亏）
        self.update_equity()

        if dry_run:
            execution_price = current_price['ask'] if direction == 'buy' else current_price['bid']
        else:
            execution_price = current_price['last']

        # 交易方向限制
        if self.trade_direction == "long" and direction == "sell":
            logger.info("当前配置只允许做多，忽略卖出信号")
            return False
        elif self.trade_direction == "short" and direction == "buy":
            logger.info("当前配置只允许做空，忽略买入信号")
            return False

        # ★ 每日亏损检查（幽灵代码落地）
        current_time = current_price.get('time', pd.Timestamp.now())
        if self._check_max_daily_loss(current_time):
            logger.warning("当日亏损已达上限，禁止开新仓")
            return False

        # 最大持仓数检查 — 优先使用 risk_config 传入值，回退到 REALTIME_CONFIG
        max_key = f'max_{position_type}_positions'
        max_positions = self._risk_config.get(max_key, None)
        if max_positions is None:
            from config import REALTIME_CONFIG
            max_positions = REALTIME_CONFIG.get(max_key, 1)
        current_count = len([p for p in self.positions if p['position_type'] == position_type])
        if 0 < max_positions <= current_count:
            logger.info(f"已达到最大{position_type}持仓数 ({max_positions})，忽略信号")
            return False

        # 资金分配
        capital_pct = self.long_capital_pct if direction == 'buy' else self.short_capital_pct
        capital_for_trade = self.total_equity * capital_pct

        position_volume = self._calculate_position_size(capital_for_trade, {'last': execution_price})
        if position_volume <= 0:
            logger.info("仓位大小为0，无法开仓")
            return False

        order_result = self.data_provider.send_order(self.symbol, direction, position_volume)
        if order_result is None:
            logger.error("订单返回None")
            return False

        try:
            order_id = order_result['order'] if isinstance(order_result, dict) else order_result.order
        except Exception as e:
            logger.error(f"解析订单ID失败: {e}")
            return False

        if order_result and order_id > 0:
            new_position = {
                'ticket': order_id,
                'symbol': self.symbol,
                'entry_price': execution_price,
                'entry_time': current_time,
                'position_type': position_type,
                'quantity': position_volume,
                'peak_profit_pct': 0.0,
            }
            self.positions.append(new_position)
            logger.info(f"开仓成功: {direction} @ {execution_price:.2f}, 手数: {position_volume:.2f}, Ticket: {order_id}")
            return True
        else:
            logger.error(f"开仓失败: {direction} @ {execution_price:.2f}")
            return False

    # ── 持仓监控 ──

    def monitor_positions(self, current_price, dry_run=False, weighted_signal=0.0):
        if not self.positions:
            return

        current_time = current_price.get('time', pd.Timestamp.now())

        positions_to_remove = []
        for position in self.positions:
            # 确定平仓执行价格
            if dry_run:
                close_price = current_price['bid'] if position['position_type'] == 'long' else current_price['ask']
            else:
                close_price = current_price['last']

            # 用执行价格计算盈亏（与平仓时一致，避免中间价偏差）
            pnl_pct = self._calculate_pnl_pct(position, close_price)
            old_peak = position.get('peak_profit_pct', 0)
            new_peak = max(old_peak, pnl_pct)
            position['peak_profit_pct'] = new_peak

            if new_peak > old_peak:
                self._save_peak_data()

            # 退出规则检查
            ctx = ExitContext(
                current_profit_pct=pnl_pct,
                peak_profit_pct=new_peak,
                entry_time=position['entry_time'],
                current_time=current_time,
                symbol=self.symbol,
                position=position,
            )
            action, reason = self.exit_engine.check(ctx)

            if action == "close":
                logger.info(f"平仓信号触发 (Ticket: {position['ticket']}): {reason}")
                success = self.data_provider.close_position(
                    position['ticket'], position['symbol'], position['quantity']
                )
                if success:
                    self._record_closed_trade(position, close_price, reason)
                    positions_to_remove.append(position)
                else:
                    logger.error(f"平仓失败, Ticket: {position['ticket']}")

        if positions_to_remove:
            self.positions = [p for p in self.positions if p not in positions_to_remove]
            self._cooldown_counter = self.cooldown_bars
            self.update_equity()
            self.cleanup_peak_data()

        # ── 对冲评估 ──
        if self.positions and self.hedge_manager:
            hedge_actions = self.hedge_manager.evaluate(weighted_signal, current_price)
            for action, target, reason in hedge_actions:
                if action == "hedge":
                    self.hedge_manager.execute_hedge(target, reason, current_price)
                elif action == "unhedge":
                    self.hedge_manager.execute_unhedge(target, reason)

    # ── 盈亏计算 ──

    def _calculate_pnl_pct(self, position, current_price_value):
        entry_price = position['entry_price']
        if position['position_type'] == 'long':
            return (current_price_value - entry_price) / entry_price if entry_price != 0 else 0.0
        else:
            return (entry_price - current_price_value) / entry_price if entry_price != 0 else 0.0

    def _calculate_unrealized_pnl(self) -> float:
        """★ 新增：计算所有持仓的浮动盈亏"""
        if not self.positions:
            return 0.0
        current_price_data = self.data_provider.get_current_price(self.symbol)
        if not current_price_data:
            return 0.0
        current_price = current_price_data['last']
        symbol_info = self.data_provider.get_symbol_info(self.symbol)
        contract_size = (symbol_info['trade_contract_size']
                         if symbol_info and isinstance(symbol_info, dict)
                         else 100) if symbol_info else 100
        total = 0.0
        for pos in self.positions:
            if pos['position_type'] == 'long':
                pnl = (current_price - pos['entry_price']) * pos['quantity'] * contract_size
            else:
                pnl = (pos['entry_price'] - current_price) * pos['quantity'] * contract_size
            total += pnl
        return total

    def _record_closed_trade(self, position, close_price, close_reason):
        symbol_info = self.data_provider.get_symbol_info(position['symbol'])
        contract_size = (symbol_info['trade_contract_size']
                         if symbol_info and isinstance(symbol_info, dict)
                         else 100) if symbol_info else 100
        if position['position_type'] == 'long':
            pnl = (close_price - position['entry_price']) * position['quantity'] * contract_size
        else:
            pnl = (position['entry_price'] - close_price) * position['quantity'] * contract_size

        trade_record = position.copy()
        trade_record.update({
            'status': 'closed',
            'close_price': close_price,
            'close_time': pd.Timestamp.now(),
            'close_reason': close_reason,
            'profit_loss': pnl
        })
        self.closed_trades.append(trade_record)
        logger.info(f"平仓记录 #{position['ticket']}: 盈亏: ${pnl:.2f}")

    # ── 权益管理 ──

    def update_equity(self):
        """★ 修正：包含浮盈浮亏"""
        if self.data_provider.is_live:
            account_info = self.data_provider.get_account_info()
            if account_info:
                equity = account_info if isinstance(account_info, dict) else account_info.equity
                if isinstance(equity, dict):
                    equity = equity.get('equity', self.total_equity)
                self.total_equity = equity
        else:
            realized_pnl = sum(t['profit_loss'] for t in self.closed_trades)
            unrealized_pnl = self._calculate_unrealized_pnl()
            self.total_equity = self.initial_capital + realized_pnl + unrealized_pnl

    def _check_max_daily_loss(self, current_time) -> bool:
        """★ 新增：检查当日最大亏损是否触发"""
        if self.max_daily_loss >= 0:
            return False
        today = current_time.date() if hasattr(current_time, 'date') else pd.Timestamp(current_time).date()
        daily_pnl = sum(
            t['profit_loss'] for t in self.closed_trades
            if hasattr(t.get('close_time'), 'date') and t['close_time'].date() == today
            or hasattr(t.get('entry_time'), 'date') and t['entry_time'].date() == today
        )
        daily_loss_pct = daily_pnl / self.initial_capital if self.initial_capital > 0 else 0
        return daily_loss_pct <= self.max_daily_loss

    # ── 同步 ──

    def sync_positions(self):
        if not self.data_provider.is_live:
            return
        live_positions = self.data_provider.get_positions(self.symbol)
        if live_positions is None:
            return
        saved_peaks = self._load_peak_data()
        existing_peaks = {pos['ticket']: pos.get('peak_profit_pct', 0.0) for pos in self.positions}
        merged_peaks = {**existing_peaks, **saved_peaks}
        self.positions.clear()
        for pos in live_positions:
            restored_peak = merged_peaks.get(pos.ticket, 0.0)
            new_position = {
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'entry_price': pos.price_open,
                'entry_time': pd.to_datetime(pos.time, unit='s'),
                'position_type': 'long' if pos.type == 0 else 'short',
                'quantity': pos.volume,
                'peak_profit_pct': restored_peak
            }
            self.positions.append(new_position)
            logger.info(f"同步持仓 {pos.ticket}: 恢复峰值={restored_peak:.6%}")
        logger.info(f"持仓已从MT5同步: {len(self.positions)}个")
        self.update_equity()

    # ── 交易摘要 ──

    def get_trade_summary(self) -> dict:
        if not self.closed_trades:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_profit_loss': 0.0,
                'avg_profit_loss': 0.0, 'max_profit': 0.0, 'max_loss': 0.0
            }
        profits = [t['profit_loss'] for t in self.closed_trades]
        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len([p for p in profits if p > 0]),
            'losing_trades': len([p for p in profits if p < 0]),
            'win_rate': (len([p for p in profits if p > 0]) / len(profits) * 100) if profits else 0,
            'total_profit_loss': sum(profits),
            'avg_profit_loss': float(np.mean(profits)) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0
        }

    # ── 持久化 ──

    def save_trade_history(self, base_filename):
        if not self.closed_trades:
            return
        df = pd.DataFrame(self.closed_trades)
        df.to_csv(f"{base_filename}.csv", index=False, encoding='utf-8-sig')
        with open(f"{base_filename}.json", 'w', encoding='utf-8') as f:
            json.dump(self.closed_trades, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"交易记录已保存到 {base_filename}.csv/.json")

    def _load_peak_data(self):
        try:
            if os.path.exists(self.peak_data_file):
                with open(self.peak_data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载峰值数据失败: {e}")
        return {}

    def _save_peak_data(self):
        try:
            if not self._persist_peaks:
                return
            peak_data = {pos['ticket']: pos.get('peak_profit_pct', 0.0) for pos in self.positions}
            with open(self.peak_data_file, 'w', encoding='utf-8') as f:
                json.dump(peak_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存峰值数据失败: {e}")

    def cleanup_peak_data(self):
        try:
            if not self._persist_peaks:
                return
            if os.path.exists(self.peak_data_file):
                with open(self.peak_data_file, 'r', encoding='utf-8') as f:
                    peak_data = json.load(f)
                current_tickets = {pos['ticket'] for pos in self.positions}
                cleaned = {t: p for t, p in peak_data.items() if t in current_tickets}
                with open(self.peak_data_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"清理峰值数据失败: {e}")
