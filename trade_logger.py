import pandas as pd
import numpy as np
from datetime import datetime
from logger import logger
import json

class TradeLogger:
    """
    交易日志记录器，跟踪每一笔交易的详情
    """
    
    def __init__(self):
        self.trades = []  # 存储所有交易记录
        self.current_position = None  # 当前持仓信息
        self.trade_id = 0  # 交易ID计数器
        
    def open_position(self, symbol, direction, price, timestamp, strategy_signal=""):
        """
        开仓
        """
        if self.current_position is not None:
            return False
            
        self.trade_id += 1
        self.current_position = {
            'trade_id': self.trade_id,
            'symbol': symbol,
            'direction': direction,  # 'buy' 或 'sell'
            'open_price': price,
            'open_time': timestamp,
            'strategy_signal': strategy_signal,
            'status': 'open'
        }
        
        logger.info(f"开仓 #{self.trade_id}: {direction} {symbol} @ {price:.2f} 时间: {timestamp}")
        return True
    
    def close_position(self, symbol, price, timestamp, close_reason="signal"):
        """
        平仓
        """
        if self.current_position is None:
            return False
            
        if self.current_position['symbol'] != symbol:
            return False
            
        # 计算盈亏
        if self.current_position['direction'] == 'buy':
            profit_loss = price - self.current_position['open_price']
            profit_loss_pct = profit_loss / self.current_position['open_price'] * 100
        else:  # sell
            profit_loss = self.current_position['open_price'] - price
            profit_loss_pct = profit_loss / self.current_position['open_price'] * 100
        
        # 完成交易记录
        trade_record = self.current_position.copy()
        trade_record.update({
            'close_price': price,
            'close_time': timestamp,
            'close_reason': close_reason,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'status': 'closed'
        })
        
        self.trades.append(trade_record)
        
        logger.info(f"平仓 #{self.trade_id}: {trade_record['direction']} {symbol} @ {price:.2f} | "
                   f"盈亏: {profit_loss:.2f} ({profit_loss_pct:+.2f}%) | 时间: {timestamp}")
        
        self.current_position = None
        return True
    
    def _calculate_profit_loss_pct(self, current_price, position):
        """
        计算当前持仓的盈亏百分比
        """
        if not position:
            return 0
        
        if position['direction'] == 'buy':
            profit_loss = current_price - position['open_price']
        else:  # sell
            profit_loss = position['open_price'] - current_price
        
        return profit_loss / position['open_price'] * 100
    
    def get_trade_summary(self):
        """
        获取交易统计摘要
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit_loss': 0,
                'avg_profit_loss': 0,
                'max_profit': 0,
                'max_loss': 0
            }
        
        closed_trades = [t for t in self.trades if t['status'] == 'closed']
        if not closed_trades:
            return {
                'total_trades': len(self.trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit_loss': 0,
                'avg_profit_loss': 0,
                'max_profit': 0,
                'max_loss': 0
            }
        
        profits = [t['profit_loss'] for t in closed_trades]
        winning_trades = len([p for p in profits if p > 0])
        losing_trades = len([p for p in profits if p < 0])
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / len(closed_trades) * 100 if closed_trades else 0,
            'total_profit_loss': sum(profits),
            'avg_profit_loss': np.mean(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0
        }
    
    def print_all_trades(self):
        """
        打印所有交易记录
        """
        if not self.trades:
            logger.info("暂无交易记录")
            return
        
        logger.info("=" * 80)
        logger.info("交易记录详情")
        logger.info("=" * 80)
        
        for trade in self.trades:
            if trade['status'] == 'closed':
                logger.info(f"交易 #{trade['trade_id']}:")
                logger.info(f"  {trade['direction'].upper()} {trade['symbol']}")
                logger.info(f"  开仓: {trade['open_price']:.2f} @ {trade['open_time']}")
                logger.info(f"  平仓: {trade['close_price']:.2f} @ {trade['close_time']}")
                logger.info(f"  盈亏: {trade['profit_loss']:.2f} ({trade['profit_loss_pct']:+.2f}%)")
                logger.info(f"  原因: {trade['close_reason']}")
                logger.info("-" * 40)
        
        # 打印统计摘要
        summary = self.get_trade_summary()
        logger.info("交易统计摘要:")
        logger.info(f"  总交易次数: {summary['total_trades']}")
        logger.info(f"  盈利次数: {summary['winning_trades']}")
        logger.info(f"  亏损次数: {summary['losing_trades']}")
        logger.info(f"  胜率: {summary['win_rate']:.1f}%")
        logger.info(f"  总盈亏: {summary['total_profit_loss']:.2f}")
        logger.info(f"  平均盈亏: {summary['avg_profit_loss']:.2f}")
        logger.info(f"  最大盈利: {summary['max_profit']:.2f}")
        logger.info(f"  最大亏损: {summary['max_loss']:.2f}")
        logger.info("=" * 80)
    
    def save_to_csv(self, filename="trades.csv"):
        """
        保存交易记录到CSV文件
        """
        if not self.trades:
            logger.info("无交易记录可保存")
            return
        
        df = pd.DataFrame(self.trades)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"交易记录已保存到 {filename}")
    
    def save_to_json(self, filename="trades.json"):
        """
        保存交易记录到JSON文件
        """
        if not self.trades:
            logger.info("无交易记录可保存")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.trades, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"交易记录已保存到 {filename}")

# 全局交易日志记录器实例
trade_logger = TradeLogger()