import pandas as pd
from trade_logger import trade_logger
from risk_management import RiskController
from config import SIGNAL_THRESHOLDS
from logger import logger
class BacktestEngine:
    def __init__(self, df):
        """
        df: 包含历史k线的DataFrame，至少包括open, high, low, close字段
        """
        self.df = df

    def run_strategy(self, strategy):
        """
        执行策略的run_backtest，得到信号序列
        """
        signals = strategy.run_backtest(self.df)
        # 记录策略信号统计
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        neutral_count = (signals == 0).sum()
        logger.info(f"策略 {strategy.__class__.__module__} 信号统计 - 买入: {buy_count}, 卖出: {sell_count}, 中性: {neutral_count}")
        return signals

    def combine_signals(self, signals_list, weights):
        """
        多策略信号加权合成，并根据阈值生成最终信号
        返回合成信号序列
        """
        df_signals = pd.concat(signals_list, axis=1).fillna(0)
        weighted_signals = df_signals * weights
        combined = weighted_signals.sum(axis=1)

        def apply_threshold(score):
            if score > SIGNAL_THRESHOLDS["buy_threshold"]:
                return 1
            elif score < SIGNAL_THRESHOLDS["sell_threshold"]:
                return -1
            else:
                return 0

        combined_signal = combined.apply(apply_threshold)
        
        # 记录信号统计
        buy_signals = (combined_signal == 1).sum()
        sell_signals = (combined_signal == -1).sum()
        neutral_signals = (combined_signal == 0).sum()
        logger.info(f"信号统计 - 买入: {buy_signals}, 卖出: {sell_signals}, 中性: {neutral_signals}")
        
        return combined_signal

    def calc_returns(self, signals):
        """
        根据信号计算策略回测收益率（简化版）
        """
        df = self.df.copy()
        df['signal'] = signals.shift(1).fillna(0)  # 防止未来函数
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'] * df['returns']
        cum_ret = (1 + df['strategy_returns']).cumprod() - 1
        return cum_ret
    
    def calc_returns_with_trades(self, signals, symbol="XAUUSD"):
        """
        根据信号计算策略回测收益率，并记录每一笔交易
        使用RiskController进行风险管理，防止频繁交易
        """
        df = self.df.copy()
        df['signal'] = signals.shift(1).fillna(0)  # 防止未来函数
        
        # 初始化风险管理器
        risk_controller = RiskController()
        
        # 记录交易
        in_position = False
        position_direction = None
        position_price = 0
        position_time = None
        last_trade_time = None
        min_trade_interval = 0  # 取消最小交易间隔限制
        
        for i in range(1, len(df)):
            current_signal = df['signal'].iloc[i]
            current_price = df['close'].iloc[i]
            current_time = df.index[i] if hasattr(df.index, 'to_pydatetime') else i
            
            # 取消交易间隔限制检查
            can_trade = True
            
            # 定期打印进度（每1000根K线）
            if i % 1000 == 0:
                logger.info(f"回测进度: {i}/{len(df)-1}")
            
            # 开仓信号
            if current_signal != 0 and not in_position and can_trade:
                direction = 'buy' if current_signal == 1 else 'sell'
                
                # 检查风险管理器是否允许交易
                if risk_controller.should_allow_trade(direction):
                    trade_logger.open_position(symbol, direction, current_price, current_time, f"signal_{current_signal}")
                    in_position = True
                    position_direction = direction
                    position_price = current_price
                    position_time = current_time
                    last_trade_time = current_time
                    
                    # 更新RiskController的持仓信息
                    position_type = "long" if direction == 'buy' else "short"
                    risk_controller.update_position_entry(current_price, position_type)
            
            # 持仓时检查风险管理条件（无论信号如何）
            if in_position:
                # 使用RiskController检查风险管理条件
                risk_action, risk_reason = risk_controller.check_risk_management(current_price)
                
                if risk_action != "none":
                    trade_logger.close_position(symbol, current_price, current_time, f"risk_management_{risk_action}")
                    risk_controller.execute_risk_action(risk_action, risk_reason)
                    in_position = False
                    continue  # 跳过后续信号处理
            
            # 中性信号处理
            elif current_signal == 0 and in_position:
                trade_logger.close_position(symbol, current_price, current_time, "signal_to_neutral")
                in_position = False
            
            # 反向信号（取消交易间隔限制）
            elif current_signal != 0 and in_position and can_trade:
                # 只有当信号明显反向时才平仓并反向
                if (current_signal == 1 and position_direction == 'sell') or \
                   (current_signal == -1 and position_direction == 'buy'):
                    # 先平仓
                    trade_logger.close_position(symbol, current_price, current_time, "signal_reverse")
                    in_position = False
                    
                    # 立即开新仓（取消等待间隔）
                    direction = 'buy' if current_signal == 1 else 'sell'
                    if risk_controller.should_allow_trade(direction):
                        trade_logger.open_position(symbol, direction, current_price, current_time, f"signal_{current_signal}")
                        in_position = True
                        position_direction = direction
                        position_price = current_price
                        position_time = current_time
                        last_trade_time = current_time
                        
                        # 更新RiskController的持仓信息
                        position_type = "long" if direction == 'buy' else "short"
                        risk_controller.update_position_entry(current_price, position_type)
            
            # 持仓到最后
            elif in_position and i == len(df) - 1:
                trade_logger.close_position(symbol, current_price, current_time, "end_of_backtest")
                in_position = False
        
        # 计算收益率
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['signal'] * df['returns']
        cum_ret = (1 + df['strategy_returns']).cumprod() - 1
        return cum_ret
