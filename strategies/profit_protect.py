
import pandas as pd
from logger import logger

class Strategy:
    def __init__(self):
        self.symbol = "XAUUSD"
        # --- 策略核心参数 ---
        # 固定止损线：亏损10%则卖出
        self.stop_loss_pct = -0.10
        # 利润回撤百分比：从最高利润点回撤30%则卖出
        self.profit_retracement_pct = 0.30
        # 追踪止损的激活阈值：当利润超过5%后，才开始启动追踪止损逻辑
        self.min_profit_for_trailing = 0.05

    def generate_signal(self):
        """
        此策略为资金管理和退出策略，不产生独立的买入信号。
        实盘逻辑应与其他策略结合，此处仅为框架完整性。
        """
        logger.warning("ProfitProtect策略是一个退出策略，不应单独用于实盘产生信号。")
        return 0

    def run_backtest(self, df):
        """
        盈利保护策略回测：
        - 固定止损：亏损10%卖出。
        - 追踪止损：利润超过5%后启动，从最高利润点回撤30%卖出。
        为了独立回测，本策略会在一开始买入，然后应用退出逻辑。
        """
        df = df.copy()
        signals = pd.Series(0, index=df.index)

        if len(df) < 2:
            return signals

        # --- 回测状态变量 ---
        position_open = False
        entry_price = 0.0
        peak_profit_pct = 0.0 # 记录达到的最高利润百分比

        for i in range(len(df)):
            # 如果没有持仓，就在第一个机会买入（用于独立回测）
            if not position_open:
                position_open = True
                entry_price = df['close'].iloc[i]
                signals.iat[i] = 1 # 买入信号
                peak_profit_pct = 0.0 # 重置最高利润
                continue

            # 如果有持仓，则执行退出逻辑
            if position_open:
                current_price = df['close'].iloc[i]
                current_profit_pct = (current_price - entry_price) / entry_price

                # 1. 更新最高利润点
                peak_profit_pct = max(peak_profit_pct, current_profit_pct)

                # 2. 检查固定止损条件
                if current_profit_pct <= self.stop_loss_pct:
                    logger.info(f"索引 {i}: 触发固定止损。入场价: {entry_price:.2f}, 当前价: {current_price:.2f}, 亏损: {current_profit_pct:.2%}")
                    signals.iat[i] = -1 # 卖出信号
                    position_open = False # 平仓
                    continue

                # 3. 检查追踪止损条件
                # 只有当最高利润超过了激活阈值，才开始计算回撤
                if peak_profit_pct > self.min_profit_for_trailing:
                    retracement_from_peak = (peak_profit_pct - current_profit_pct)
                    
                    # 避免除以零或负数的情况
                    if peak_profit_pct > 0:
                        retracement_pct = retracement_from_peak / peak_profit_pct
                        if retracement_pct >= self.profit_retracement_pct:
                            logger.info(f"索引 {i}: 触发追踪止损。最高利润: {peak_profit_pct:.2%}, 当前利润: {current_profit_pct:.2%}, 回撤超过30%")
                            signals.iat[i] = -1 # 卖出信号
                            position_open = False # 平仓
                            continue
        return signals
