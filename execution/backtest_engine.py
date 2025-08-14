import pandas as pd
from tqdm import tqdm
from core.risk import RiskController
from config import SIGNAL_THRESHOLDS, BACKTEST_CONFIG, SPREAD
from logger import logger

class BacktestEngine:
    def __init__(self, df, trade_direction="both"):
        self.df = df
        self.trade_direction = trade_direction
        self.spread = BACKTEST_CONFIG.get("spread", SPREAD)
        
    def run_strategy(self, strategy):
        signals = strategy.run_backtest(self.df)
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        logger.info(f"策略 {strategy.__class__.__module__} 信号统计 - 买入: {buy_count}, 卖出: {sell_count}")
        return signals

    def combine_signals(self, signals_list, weights):
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
        buy_signals = (combined_signal == 1).sum()
        sell_signals = (combined_signal == -1).sum()
        logger.info(f"组合信号统计 - 买入: {buy_signals}, 卖出: {sell_signals}")
        return combined_signal

    def run_backtest(self, signals, symbol="XAUUSD"):
        df = self.df.copy()
        df['signal'] = signals.shift(1).fillna(0)
        
        risk_controller = RiskController(self.trade_direction)
        logger.info(f"回测开始: 交易方向={self.trade_direction}")
        
        # 使用tqdm创建进度条
        for i in tqdm(range(1, len(df)), desc=f"Backtesting ({len(df)} bars)"):
            current_signal = df['signal'].iloc[i]
            # 计算考虑双向点差的买卖价格
            close_price = df['close'].iloc[i]
            spread_points = self.spread
            spread_half = spread_points * 0.01 / 2  # XAUUSD: 1点 = 0.01，双向点差各一半
            
            current_price = {
                'bid': close_price - spread_half,  # 卖出价格（中间价 - 点差/2）
                'ask': close_price + spread_half,  # 买入价格（中间价 + 点差/2）
                'last': close_price  # 最后成交价（中间价）
            }
            
            direction = None
            if current_signal == 1:
                direction = "buy"
            elif current_signal == -1:
                direction = "sell"
            
            if direction:
                risk_controller.process_trading_signal(
                    direction, current_price, abs(current_signal), dry_run=True
                )
            
            risk_controller.monitor_positions(current_price, dry_run=True)
        
        risk_controller.position_manager.force_close_all_positions(df['close'].iloc[-1], dry_run=True)
        
        logger.info("回测完成，生成性能报告...")
        summary = risk_controller.position_manager.get_trade_summary()
        
        logger.info("=" * 80)
        logger.info("回测性能报告")
        logger.info("=" * 80)
        logger.info(f"    总交易次数: {summary['total_trades']}")
        logger.info(f"    盈利次数: {summary['winning_trades']}")
        logger.info(f"    亏损次数: {summary['losing_trades']}")
        logger.info(f"    胜率: {summary['win_rate']:.2f}%")
        logger.info("-" * 40)
        logger.info(f"    总盈亏: ${summary['total_profit_loss']:.2f}")
        logger.info(f"    平均每笔交易盈亏: ${summary['avg_profit_loss']:.2f}")
        logger.info(f"    最大盈利: ${summary['max_profit']:.2f}")
        logger.info(f"    最大亏损: ${summary['max_loss']:.2f}")
        logger.info("=" * 80)

        risk_controller.position_manager.save_to_csv("backtest_trades.csv")
        risk_controller.position_manager.save_to_json("backtest_trades.json")

        return summary
