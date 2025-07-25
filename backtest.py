import pandas as pd

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
        return strategy.run_backtest(self.df)

    def combine_signals(self, signals_list, weights, buy_threshold, sell_threshold):
        """
        多策略信号加权合成，并根据阈值生成最终信号
        返回合成信号序列
        """
        df_signals = pd.concat(signals_list, axis=1).fillna(0)
        weighted_signals = df_signals * weights
        combined = weighted_signals.sum(axis=1)

        def apply_threshold(score):
            if score >= buy_threshold:
                return 1
            elif score <= sell_threshold:
                return -1
            else:
                return 0

        combined_signal = combined.apply(apply_threshold)
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
