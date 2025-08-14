import pandas as pd
import numpy as np
import json
import os
from logger import logger
from config import RISK_CONFIG, SYMBOL, INITIAL_CAPITAL, CAPITAL_ALLOCATION, RISK_CONFIG_CONST

class PositionManager:
    """
    持仓管理器 - 统一管理所有持仓操作和交易记录 (已重构为依赖注入)
    """
    
    def __init__(self, data_provider, trade_direction="both"):
        self.data_provider = data_provider
        self.symbol = SYMBOL
        self.trade_direction = trade_direction
        
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
        
        # 持久化文件路径
        self.peak_data_file = "position_peaks.json"
        
        # 初始化时加载峰值数据
        self._load_peak_data()

    def _calculate_position_size(self, capital_to_allocate, current_price):
        price = current_price['last']
        logger.info(f"当前价格: {price}")
        if not isinstance(price, (int, float)) or price == 0: 
            logger.error(f"价格无效: {price}")
            return 0.0

        symbol_info = self.data_provider.get_symbol_info(self.symbol)
        logger.info(f"合约信息: {symbol_info}")
        if not symbol_info:
            logger.error(f"无法获取 {self.symbol} 的合约信息")
            return 0.0

        is_dict = isinstance(symbol_info, dict)
        contract_size = symbol_info['trade_contract_size'] if is_dict else symbol_info.trade_contract_size
        volume_step = symbol_info['volume_step'] if is_dict else symbol_info.volume_step
        min_volume = symbol_info['volume_min'] if is_dict else symbol_info.volume_min
        max_volume = symbol_info['volume_max'] if is_dict else symbol_info.volume_max
        
        logger.info(f"合约大小: {contract_size}, 步长: {volume_step}, 最小: {min_volume}, 最大: {max_volume}")

        value_of_one_lot = price * contract_size
        logger.info(f"一手价值: {value_of_one_lot}")
        if value_of_one_lot == 0: 
            logger.error("一手价值为0")
            return 0.0

        volume = capital_to_allocate / value_of_one_lot
        logger.info(f"原始手数: {volume}")
        volume = round(volume / volume_step) * volume_step
        logger.info(f"调整后手数: {volume}")
        volume = max(min_volume, min(volume, max_volume))
        logger.info(f"最终手数: {volume}")
        return volume

    def open_position(self, direction, current_price, signal_strength=0.0, dry_run=False):
        logger.info(f"开始处理{direction}开仓请求")
        position_type_to_open = 'long' if direction == 'buy' else 'short'
        
        # 根据交易方向选择合适的成交价格
        if dry_run:
            # 在回测/模拟模式中，考虑点差
            if direction == 'buy':
                execution_price = current_price['ask']  # 买入用卖方价（ask）
            else:
                execution_price = current_price['bid']  # 卖出用买方价（bid）
        else:
            # 实盘模式使用最后价格
            execution_price = current_price['last']
        
        # 检查交易方向限制
        if self.trade_direction == "long" and direction == "sell":
            logger.info("当前配置只允许做多，忽略卖出信号")
            return False
        elif self.trade_direction == "short" and direction == "buy":
            logger.info("当前配置只允许做空，忽略买入信号")
            return False
        
        # 从配置中获取最大持仓限制
        from config import REALTIME_CONFIG
        max_positions = REALTIME_CONFIG.get(f'max_{position_type_to_open}_positions', 1)
        logger.info(f"配置文件中的max_{position_type_to_open}_positions: {REALTIME_CONFIG.get(f'max_{position_type_to_open}_positions', 'NOT_FOUND')}")
        logger.info(f"最大{position_type_to_open}持仓数限制: {max_positions}")
        
        # 检查当前同向持仓数量
        current_positions = [p for p in self.positions if p['position_type'] == position_type_to_open]
        logger.info(f"当前{position_type_to_open}持仓数: {len(current_positions)}")
        
        # 如果配置为0或负数，表示无限制
        if max_positions <= 0:
            logger.info(f"{position_type_to_open}持仓数无限制")
        elif len(current_positions) >= max_positions:
            logger.info(f"已达到最大{position_type_to_open}持仓数 ({max_positions})，忽略信号")
            return False

        capital_pct = self.long_capital_pct if direction == 'buy' else self.short_capital_pct
        capital_for_this_trade = self.total_equity * capital_pct
        logger.info(f"分配资金: {capital_for_this_trade:.2f} (总权益: {self.total_equity:.2f}, 比例: {capital_pct:.2%})")
        
        position_volume = self._calculate_position_size(capital_for_this_trade, {'last': execution_price})
        logger.info(f"计算仓位大小: {position_volume:.2f}")
        if position_volume <= 0:
            logger.info("仓位大小为0，无法开仓")
            return False

        logger.info(f"发送订单: {direction} {position_volume:.2f}手 {self.symbol}")
        order_result = self.data_provider.send_order(self.symbol, direction, position_volume)
        logger.info(f"订单结果: {order_result}")
        
        if order_result is None:
            logger.error("订单返回None，可能是数据提供者问题")
            return False
        
        try:
            order_id = order_result['order'] if isinstance(order_result, dict) else order_result.order
            logger.info(f"订单ID: {order_id}")
        except Exception as e:
            logger.error(f"解析订单ID失败: {e}")
            return False

        if order_result and order_id > 0:
            new_position = {
                'ticket': order_id,
                'symbol': self.symbol,
                'entry_price': execution_price,
                'entry_time': current_price['time'],
                'position_type': position_type_to_open,
                'quantity': position_volume,
                'peak_profit_pct': 0.0,
            }
            self.positions.append(new_position)
            logger.info(f"开仓成功: {direction} @ {execution_price:.2f}, 手数: {position_volume:.2f}, Ticket: {order_id}")
            logger.info(f"持仓 {order_id}: 初始化峰值盈利为 0.0%")
            return True
        else:
            logger.error(f"开仓失败: {direction} @ {current_price['last']:.2f}, 订单结果: {order_result}")
            return False

    def monitor_positions(self, current_price, dry_run=False):
        if not self.positions: return

        # 打印所有持仓的当前状态
        logger.info(f"当前持仓数量: {len(self.positions)}")
        for pos in self.positions:
            # 计算持仓时间
            holding_time = current_price['time'] - pos['entry_time']
            holding_minutes = holding_time.total_seconds() / 60
            logger.info(f"持仓 {pos['ticket']}: 持仓时间={holding_minutes:.1f}分钟, 当前峰值={pos.get('peak_profit_pct', 0):.6%}")

        positions_to_remove = []
        for position in self.positions:
            pnl_pct = self._calculate_pnl_pct(position, current_price['last'])
            old_peak = position.get('peak_profit_pct', 0)
            
            # 确保正确更新峰值
            new_peak = max(old_peak, pnl_pct)
            position['peak_profit_pct'] = new_peak
            
            # 打印详细的追踪止损计算信息
            # logger.info(f"持仓 {position['ticket']}: 当前盈利={pnl_pct:.6%}, 原峰值={old_peak:.6%}, 新峰值={new_peak:.6%}")
            
            # 如果有新的峰值，打印详细信息并保存到文件
            if new_peak > old_peak:
                logger.info(f"持仓 {position['ticket']}: 新的峰值盈利: {new_peak:.6%}")
                # 确认保存到持仓对象
                logger.info(f"持仓 {position['ticket']}: 确认保存的峰值: {position.get('peak_profit_pct', 0):.6%}")
                # 保存到文件
                self._save_peak_data()
            
            action, reason = self._check_risk_conditions(
                pnl_pct, 
                new_peak,
                position['entry_time'],
                current_price['time'],
                position
            )
            
            if action == "close":
                logger.info(f"平仓信号触发 (Ticket: {position['ticket']}): {reason}")
                
                # 在dry_run模式下，计算考虑点差的平仓价格
                if dry_run:
                    # 从配置获取点差
                    from config import SIMULATION_CONFIG
                    spread_points = SIMULATION_CONFIG.get("spread", 16)
                    spread_value = spread_points * 0.01  # XAUUSD: 1点 = 0.01
                    
                    if position['position_type'] == 'long':
                        # 多头平仓用bid价（卖出价）
                        close_price = current_price['bid']
                    else:
                        # 空头平仓用ask价（买入价）
                        close_price = current_price['ask']
                else:
                    close_price = current_price['last']
                
                success = self.data_provider.close_position(position['ticket'], position['symbol'], position['quantity'])
                if success:
                    self._record_closed_trade(position, close_price, reason)
                    positions_to_remove.append(position)
                else:
                    logger.error(f"平仓失败, Ticket: {position['ticket']}")

        if positions_to_remove:
            # 记录平仓的持仓ticket
            closed_tickets = [pos['ticket'] for pos in positions_to_remove]
            logger.info(f"平仓持仓: {closed_tickets}")
            
            self.positions = [p for p in self.positions if p not in positions_to_remove]
            self.update_equity()
            
            # 清理已平仓持仓的峰值数据
            self.cleanup_peak_data()

    def _calculate_pnl_pct(self, position, current_price_value):
        entry_price = position['entry_price']
        if position['position_type'] == 'long':
            return (current_price_value - entry_price) / entry_price if entry_price != 0 else 0.0
        else:
            return (entry_price - current_price_value) / entry_price if entry_price != 0 else 0.0

    def _record_closed_trade(self, position, close_price, close_reason):
        symbol_info = self.data_provider.get_symbol_info(position['symbol'])
        contract_size = (symbol_info['trade_contract_size'] if isinstance(symbol_info, dict) 
                       else symbol_info.trade_contract_size) if symbol_info else 100
        
        pnl = 0
        if position['position_type'] == 'long':
            pnl = (close_price - position['entry_price']) * position['quantity'] * contract_size
        else:
            pnl = (position['entry_price'] - close_price) * position['quantity'] * contract_size

        trade_record = position.copy()
        trade_record.update({
            'status': 'closed', 'close_price': close_price, 'close_time': pd.Timestamp.now(),
            'close_reason': close_reason, 'profit_loss': pnl
        })
        self.closed_trades.append(trade_record)
        logger.info(f"平仓记录 #{position['ticket']}: 盈亏: ${pnl:.2f}")

    def _check_risk_conditions(self, current_profit_pct, peak_profit_pct, entry_time, current_time, position=None):
        # 记录详细的追踪止损计算过程
        logger.debug(f"追踪止损计算: 当前盈利={current_profit_pct:.4%}, 峰值盈利={peak_profit_pct:.4%}, 阈值={self.min_profit_for_trailing:.4%}")
        
        if current_profit_pct <= self.stop_loss_pct: 
            logger.debug(f"触发止损: {current_profit_pct:.4%} <= {self.stop_loss_pct:.4%}")
            return "close", f"止损触发"
        if current_profit_pct >= self.take_profit_pct: 
            logger.debug(f"触发止盈: {current_profit_pct:.4%} >= {self.take_profit_pct:.4%}")
            return "close", f"止盈触发"
        
        # 修正后的追踪止损逻辑
        if peak_profit_pct > self.min_profit_for_trailing:
            stop_level = peak_profit_pct * (1 - self.profit_retracement_pct)
            retracement_amount = peak_profit_pct - current_profit_pct
            retracement_pct = (retracement_amount / peak_profit_pct) if peak_profit_pct > 0 else 0
            
            # 计算实际回撤金额
            if position:
                symbol_info = self.data_provider.get_symbol_info(self.symbol)
                contract_size = (symbol_info['trade_contract_size'] if isinstance(symbol_info, dict) 
                               else symbol_info.trade_contract_size) if symbol_info else 100
                # 使用持仓的entry_price和quantity来计算实际回撤金额
                position_value = position['entry_price'] * position['quantity'] * contract_size
                actual_retracement_amount = position_value * retracement_amount
            else:
                actual_retracement_amount = 0
            
            logger.info(f"追踪止损详细计算:")
            logger.info(f"  峰值盈利: {peak_profit_pct:.4%}")
            logger.info(f"  当前盈利: {current_profit_pct:.4%}")
            logger.info(f"  回撤阈值: {self.profit_retracement_pct:.2%}")
            logger.info(f"  计算止损位: {peak_profit_pct:.4%} × (1 - {self.profit_retracement_pct:.2%}) = {stop_level:.4%}")
            logger.info(f"  实际回撤: {retracement_pct:.2%} (回撤金额: ${actual_retracement_amount:.2f})")
            logger.info(f"  触发条件: 当前盈利 {current_profit_pct:.4%} <= 止损位 {stop_level:.4%} = {current_profit_pct <= stop_level}")
            
            if current_profit_pct <= stop_level:
                logger.info(f"触发追踪止损: 当前盈利{current_profit_pct:.4%} <= 止损位{stop_level:.4%}")
                return "close", f"追踪止损触发 (盈利从 {peak_profit_pct:.2%} 回落至 {current_profit_pct:.2%}, 回撤{retracement_pct:.2%})"
            else:
                logger.info(f"未触发追踪止损: 当前盈利{current_profit_pct:.4%} > 止损位{stop_level:.4%}")
        else:
            pass  # 未达到追踪止损条件

        if self.enable_time_based_exit:
            holding_duration = current_time - entry_time
            if (holding_duration.total_seconds() / 60) > self.max_holding_minutes:
                if current_profit_pct < self.min_profit_for_time_exit:
                    return "close", f"超时平仓 (持仓超过{self.max_holding_minutes}分钟且盈利未达标)"

        return "none", ""

    def update_equity(self):
        # In live mode, always trust the broker's account info
        if self.data_provider.is_live:
            account_info = self.data_provider.get_account_info()
            if account_info:
                self.total_equity = account_info.equity
        # In simulation/backtest mode, calculate equity based on trade history
        else:
            self.total_equity = self.initial_capital + sum(t['profit_loss'] for t in self.closed_trades)

    def sync_positions(self):
        # 只在实盘模式下执行同步
        if not self.data_provider.is_live:
            return

        live_positions = self.data_provider.get_positions(self.symbol)
        if live_positions is None: return

        # 从文件加载保存的峰值数据
        saved_peaks = self._load_peak_data()
        logger.info(f"从文件加载的峰值数据: {saved_peaks}")

        # 保存现有的峰值数据（内存中的）
        existing_peaks = {pos['ticket']: pos.get('peak_profit_pct', 0.0) for pos in self.positions}
        
        # 合并数据：优先使用文件中的数据，如果没有则使用内存中的数据
        merged_peaks = {**existing_peaks, **saved_peaks}
        # logger.info(f"合并后的峰值数据: {merged_peaks}")

        self.positions.clear()
        for pos in live_positions:
            # 获取该持仓之前记录的峰值，优先从文件中获取
            restored_peak = merged_peaks.get(pos.ticket, 0.0)
            
            new_position = {
                'ticket': pos.ticket, 'symbol': pos.symbol, 'entry_price': pos.price_open,
                'entry_time': pd.to_datetime(pos.time, unit='s'),
                'position_type': 'long' if pos.type == 0 else 'short',
                'quantity': pos.volume, 'peak_profit_pct': restored_peak
            }
            self.positions.append(new_position)
            
            logger.info(f"同步持仓 {pos.ticket}: 恢复峰值={restored_peak:.6%}")
        
        logger.info(f"持仓已从MT5同步: {len(self.positions)}个")

    def get_trade_summary(self):
        if not self.closed_trades: return {}
        profits = [t['profit_loss'] for t in self.closed_trades]
        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len([p for p in profits if p > 0]),
            'losing_trades': len([p for p in profits if p < 0]),
            'win_rate': (len([p for p in profits if p > 0]) / len(profits) * 100) if profits else 0,
            'total_profit_loss': sum(profits),
            'avg_profit_loss': np.mean(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0
        }

    def save_trade_history(self, base_filename):
        if not self.closed_trades: return
        df = pd.DataFrame(self.closed_trades)
        df.to_csv(f"{base_filename}.csv", index=False, encoding='utf-8-sig')
        with open(f"{base_filename}.json", 'w', encoding='utf-8') as f:
            json.dump(self.closed_trades, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"交易记录已保存到 {base_filename}.csv/.json")

    def _load_peak_data(self):
        """从文件加载峰值数据"""
        try:
            if os.path.exists(self.peak_data_file):
                with open(self.peak_data_file, 'r', encoding='utf-8') as f:
                    peak_data = json.load(f)
                # logger.info(f"从文件加载峰值数据: {peak_data}")
                return peak_data
            else:
                logger.info("峰值数据文件不存在，使用空数据")
                return {}
        except Exception as e:
            logger.error(f"加载峰值数据失败: {e}")
            return {}

    def _save_peak_data(self):
        """保存峰值数据到文件"""
        try:
            peak_data = {pos['ticket']: pos.get('peak_profit_pct', 0.0) for pos in self.positions}
            with open(self.peak_data_file, 'w', encoding='utf-8') as f:
                json.dump(peak_data, f, ensure_ascii=False, indent=2)
            #logger.info(f"峰值数据已保存到文件: {peak_data}")
        except Exception as e:
            logger.error(f"保存峰值数据失败: {e}")

    def cleanup_peak_data(self):
        """清理已平仓持仓的峰值数据"""
        try:
            if os.path.exists(self.peak_data_file):
                with open(self.peak_data_file, 'r', encoding='utf-8') as f:
                    peak_data = json.load(f)
                
                # 获取当前持仓的ticket列表
                current_tickets = {pos['ticket'] for pos in self.positions}
                
                # 清理已平仓持仓的数据
                cleaned_peak_data = {ticket: peak for ticket, peak in peak_data.items() 
                                   if ticket in current_tickets}
                
                # 保存清理后的数据
                with open(self.peak_data_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_peak_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"清理峰值数据完成，保留 {len(cleaned_peak_data)} 个持仓数据")
        except Exception as e:
            logger.error(f"清理峰值数据失败: {e}")
