import pandas as pd


class SignalCombiner:
    """信号组合器 — 统一加权求和和阈值过滤逻辑

    消除 start_backtest.py / optimizer.py / backtest_engine.py 中的重复。
    """

    @staticmethod
    def combine_vectorized(signals_df: pd.DataFrame,
                           weights: dict,
                           buy_threshold: float,
                           sell_threshold: float) -> pd.Series:
        """向量化信号组合 — 用于回测预生成阶段

        Args:
            signals_df: 每列是一个策略的回测信号序列，列名为策略类名
            weights: {config_key 或 class_name: 权重}
            buy_threshold: 加权和大于此值 → 买入信号(1)
            sell_threshold: 加权和小于此值 → 卖出信号(-1)

        Returns:
            最终信号序列 (-1/0/1)
        """
        combined = pd.Series(0.0, index=signals_df.index)
        for col in signals_df.columns:
            # 权重匹配：先尝试类名，再尝试config_key
            w = weights.get(col, 1.0)
            combined += signals_df[col] * w

        final = pd.Series(0, index=combined.index)
        final[combined > buy_threshold] = 1
        final[combined < sell_threshold] = -1
        return final

    @staticmethod
    def combine_at_bar(signals: dict,
                       weights: dict,
                       buy_threshold: float,
                       sell_threshold: float) -> int:
        """单 bar 信号组合 — 用于回测/优化器逐 bar 循环

        Args:
            signals: {策略标识: signal_value (-1/0/1 或 0.0~1.0)}
            weights: {策略标识: 权重}
            buy_threshold: 加权和买入阈值
            sell_threshold: 加权和卖出阈值

        Returns:
            -1/0/1
        """
        weighted_sum = sum(
            signals.get(name, 0) * weights.get(name, 1.0)
            for name in set(signals) | set(weights)
        )
        if weighted_sum > buy_threshold:
            return 1
        elif weighted_sum < sell_threshold:
            return -1
        return 0
