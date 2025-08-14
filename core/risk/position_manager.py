import pandas as pd
import numpy as np
import json
import os
from logger import logger
from config import RISK_CONFIG, SYMBOL, INITIAL_CAPITAL, CAPITAL_ALLOCATION, RISK_CONFIG_CONST

import pandas as pd
import numpy as np
import json
import os
from logger import logger
from config import RISK_CONFIG, SYMBOL, INITIAL_CAPITAL, CAPITAL_ALLOCATION, RISK_CONFIG_CONST, USE_MEMORY_FOR_PEAK_DATA

class PositionManager:
    """
    持仓管理器 - 统一管理所有持仓操作和交易记录 (已重构为依赖注入)
    """
    
    def __init__(self, data_provider, trade_direction="both"):
        self.data_provider = data_provider
        self.symbol = SYMBOL
        self.trade_direction = trade_direction
        self.use_memory_storage = USE_MEMORY_FOR_PEAK_DATA
        
        # 风险管理参数
        self.stop_loss_pct = RISK_CONFIG.get("stop_loss_pct", -0.10)
        self.profit_retracement_pct = RISK_CONFIG.get("profit_retracement_pct", 0.10)
        self.min_profit_for_trailing = RISK_CONFIG.get("min_profit_for_trailing", 0.0003)
        self.take_profit_pct = RISK_CONFIG.get("take_profit_pct", 0.20)
        self.enable_time_based_exit = RISK_CONFIG_CONST.get("enable_time_based_exit", False)
        self.max_holding_minutes = RISK_CONFIG.get("max_holding_minutes", 60)
        self.min_profit_for_time_exit = RISK_CONFIG.get("min_profit_for_time_exit", 0.0005)

        # 资金管理参数
        self.initial_capital = INITIAL_CAPITAL
        self.long_capital_pct = CAPITAL_ALLOCATION.get("long_pct", 0.5)
        self.short_capital_pct = CAPITAL_ALLOCATION.get("short_pct", 0.5)
        self.max_daily_loss = RISK_CONFIG.get("max_daily_loss", -0.10)
        
        # 持仓和交易记录
        self.positions = []
        self.closed_trades = []
        self.total_equity = self.initial_capital
        
        # 根据配置选择存储方式
        if self.use_memory_storage:
            self.peak_data = {}
            logger.info("使用内存变量存储持仓峰值数据")
        else:
            self.peak_data_file = f"position_peaks_{os.getpid()}.json"
            logger.info(f"使用文件 {self.peak_data_file} 存储持仓峰值数据")
            self._load_peak_data() # 初始化时加载一次

    def _calculate_position_size(self, capital_to_allocate, current_price):
        price = current_price['last']
        if not isinstance(price, (int, float)) or price == 0: return 0.0
        symbol_info = self.data_provider.get_symbol_info(self.symbol)
        if not symbol_info: return 0.0
        is_dict = isinstance(symbol_info, dict)
        contract_size = symbol_info['trade_contract_size'] if is_dict else symbol_info.trade_contract_size
        volume_step = symbol_info['volume_step'] if is_dict else symbol_info.volume_step
        min_volume = symbol_info['volume_min'] if is_dict else symbol_info.volume_min
        max_volume = symbol_info['volume_max'] if is_dict else symbol_info.volume_max
        value_of_one_lot = price * contract_size
        if value_of_one_lot == 0: return 0.0
        volume = capital_to_allocate / value_of_one_lot
        volume = round(volume / volume_step) * volume_step
        return max(min_volume, min(volume, max_volume))

    def open_position(self, direction, current_price, signal_strength=0.0, dry_run=False):
        position_type_to_open = 'long' if direction == 'buy' else 'short'
        execution_price = current_price['ask'] if dry_run and direction == 'buy' else current_price['bid'] if dry_run else current_price['last']
        if (self.trade_direction == "long" and direction == "sell") or (self.trade_direction == "short" and direction == "buy"): return False
        from config import REALTIME_CONFIG
        max_positions = REALTIME_CONFIG.get(f'max_{position_type_to_open}_positions', 1)
        current_positions = [p for p in self.positions if p['position_type'] == position_type_to_open]
        if max_positions > 0 and len(current_positions) >= max_positions: return False
        capital_pct = self.long_capital_pct if direction == 'buy' else self.short_capital_pct
        capital_for_this_trade = self.total_equity * capital_pct
        position_volume = self._calculate_position_size(capital_for_this_trade, {'last': execution_price})
        if position_volume <= 0: return False
        order_result = self.data_provider.send_order(self.symbol, direction, position_volume)
        if order_result is None: return False
        try:
            order_id = order_result['order'] if isinstance(order_result, dict) else order_result.order
        except Exception: return False
        if order_result and order_id > 0:
            new_position = {
                'ticket': order_id, 'symbol': self.symbol, 'entry_price': execution_price,
                'entry_time': current_price['time'], 'position_type': position_type_to_open,
                'quantity': position_volume, 'peak_profit_pct': 0.0,
            }
            self.positions.append(new_position)
            return True
        return False

    def monitor_positions(self, current_price, dry_run=False):
        if not self.positions: return
        positions_to_remove = []
        for position in self.positions:
            pnl_pct = self._calculate_pnl_pct(position, current_price['last'])
            old_peak = position.get('peak_profit_pct', 0)
            new_peak = max(old_peak, pnl_pct)
            position['peak_profit_pct'] = new_peak
            if new_peak > old_peak: self._save_peak_data()
            action, reason = self._check_risk_conditions(pnl_pct, new_peak, position['entry_time'], current_price['time'], position)
            if action == "close":
                close_price = current_price['last']
                if dry_run:
                    close_price = current_price['bid'] if position['position_type'] == 'long' else current_price['ask']
                if self.data_provider.close_position(position['ticket'], position['symbol'], position['quantity']):
                    self._record_closed_trade(position, close_price, reason)
                    positions_to_remove.append(position)
        if positions_to_remove:
            self.positions = [p for p in self.positions if p not in positions_to_remove]
            self.update_equity()
            self.cleanup_peak_data()

    def _calculate_pnl_pct(self, position, current_price_value):
        entry_price = position['entry_price']
        if entry_price == 0: return 0.0
        return (current_price_value - entry_price) / entry_price if position['position_type'] == 'long' else (entry_price - current_price_value) / entry_price

    def _record_closed_trade(self, position, close_price, close_reason):
        symbol_info = self.data_provider.get_symbol_info(position['symbol'])
        contract_size = (symbol_info['trade_contract_size'] if isinstance(symbol_info, dict) else symbol_info.trade_contract_size) if symbol_info else 100
        pnl = (close_price - position['entry_price']) * position['quantity'] * contract_size if position['position_type'] == 'long' else (position['entry_price'] - close_price) * position['quantity'] * contract_size
        trade_record = position.copy()
        trade_record.update({'status': 'closed', 'close_price': close_price, 'close_time': pd.Timestamp.now(), 'close_reason': close_reason, 'profit_loss': pnl})
        self.closed_trades.append(trade_record)

    def _check_risk_conditions(self, current_profit_pct, peak_profit_pct, entry_time, current_time, position=None):
        if current_profit_pct <= self.stop_loss_pct: return "close", f"止损触发"
        if current_profit_pct >= self.take_profit_pct: return "close", f"止盈触发"
        if peak_profit_pct > self.min_profit_for_trailing:
            stop_level = peak_profit_pct * (1 - self.profit_retracement_pct)
            if current_profit_pct <= stop_level:
                return "close", f"追踪止损触发"
        if self.enable_time_based_exit and (current_time - entry_time).total_seconds() / 60 > self.max_holding_minutes and current_profit_pct < self.min_profit_for_time_exit:
            return "close", f"超时平仓"
        return "none", ""

    def update_equity(self):
        if self.data_provider.is_live:
            account_info = self.data_provider.get_account_info()
            if account_info: self.total_equity = account_info.equity
        else:
            self.total_equity = self.initial_capital + sum(t['profit_loss'] for t in self.closed_trades)

    def sync_positions(self):
        if not self.data_provider.is_live: return
        live_positions = self.data_provider.get_positions(self.symbol)
        if live_positions is None: return
        saved_peaks = self._load_peak_data()
        self.positions.clear()
        for pos in live_positions:
            restored_peak = saved_peaks.get(str(pos.ticket), 0.0)
            self.positions.append({
                'ticket': pos.ticket, 'symbol': pos.symbol, 'entry_price': pos.price_open,
                'entry_time': pd.to_datetime(pos.time, unit='s'), 'position_type': 'long' if pos.type == 0 else 'short',
                'quantity': pos.volume, 'peak_profit_pct': restored_peak
            })

    def get_trade_summary(self):
        if not self.closed_trades: return {}
        profits = [t['profit_loss'] for t in self.closed_trades]
        return {
            'total_trades': len(self.closed_trades), 'winning_trades': len([p for p in profits if p > 0]),
            'losing_trades': len([p for p in profits if p < 0]), 'win_rate': (len([p for p in profits if p > 0]) / len(profits) * 100) if profits else 0,
            'total_profit_loss': sum(profits), 'avg_profit_loss': np.mean(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0, 'max_loss': min(profits) if profits else 0
        }

    def save_trade_history(self, base_filename):
        if not self.closed_trades: return
        pd.DataFrame(self.closed_trades).to_csv(f"{base_filename}.csv", index=False, encoding='utf-8-sig')

    def _load_peak_data(self):
        if self.use_memory_storage:
            return self.peak_data
        else:
            try:
                if os.path.exists(self.peak_data_file):
                    with open(self.peak_data_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                return {}
            except Exception as e:
                logger.error(f"从文件加载峰值数据失败: {e}")
                return {}

    def _save_peak_data(self):
        if self.use_memory_storage:
            try:
                self.peak_data.update({str(pos['ticket']): pos.get('peak_profit_pct', 0.0) for pos in self.positions})
            except Exception as e:
                logger.error(f"更新内存峰值数据失败: {e}")
        else:
            try:
                peak_data = {str(pos['ticket']): pos.get('peak_profit_pct', 0.0) for pos in self.positions}
                with open(self.peak_data_file, 'w', encoding='utf-8') as f:
                    json.dump(peak_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"保存峰值数据到文件失败: {e}")

    def cleanup_peak_data(self):
        current_tickets = {str(pos['ticket']) for pos in self.positions}
        if self.use_memory_storage:
            try:
                self.peak_data = {ticket: peak for ticket, peak in self.peak_data.items() if ticket in current_tickets}
            except Exception as e:
                logger.error(f"清理内存峰值数据失败: {e}")
        else:
            try:
                if os.path.exists(self.peak_data_file):
                    with open(self.peak_data_file, 'r', encoding='utf-8') as f:
                        peak_data = json.load(f)
                    cleaned_peak_data = {ticket: peak for ticket, peak in peak_data.items() if ticket in current_tickets}
                    with open(self.peak_data_file, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_peak_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"清理文件峰值数据失败: {e}")
