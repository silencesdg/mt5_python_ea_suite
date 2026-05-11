"""统一回测引擎 — 消除 start_backtest.py 和 optimizer.py 中的回测循环重复"""

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from logger import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import MultiTimeframeDataStore, BacktestDataProvider
from core.risk.controller import RiskController
from core.risk.market_state import MarketStateAnalyzer
from core.signal.registry import StrategyRegistry
from core.signal.combiner import SignalCombiner
from execution.weights import DynamicWeightManager
from config import (
    SYMBOL, TIMEFRAME, INITIAL_CAPITAL, SIGNAL_THRESHOLDS, DEFAULT_WEIGHTS,
    BACKTEST_CONFIG, RISK_CONFIG,
    MARKET_STATE_CONFIG, TREND_INDICATOR_WEIGHTS, TREND_THRESHOLDS, CONFIDENCE_THRESHOLDS
)
from utils.constants import PERIOD_H1


class BacktestEngine:
    """统一回测引擎

    支持两种模式：
      - standard: 使用 DEFAULT_WEIGHTS 的固定权重回测
      - dynamic: 使用 MarketStateAnalyzer 的动态权重回测
    """

    def __init__(self, symbol: str = SYMBOL, timeframe: int = TIMEFRAME,
                 initial_capital: float = INITIAL_CAPITAL,
                 use_dynamic_weights: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.use_dynamic_weights = use_dynamic_weights

        # 延迟初始化
        self.multi_tf = None
        self.data_provider = None
        self.risk_controller = None
        self.registry = StrategyRegistry()

    def load_data(self, rates: np.ndarray) -> None:
        """加载M1数据并初始化组件"""
        if rates is None or len(rates) == 0:
            raise ValueError("无法加载历史数据")

        self.multi_tf = MultiTimeframeDataStore()
        self.multi_tf.load_m1_data(rates)
        self.data_provider = BacktestDataProvider(self.multi_tf, self.initial_capital)

        # 预计算市场状态（动态权重模式）
        analyzer = None
        if self.use_dynamic_weights:
            analyzer = MarketStateAnalyzer(
                timeframe=PERIOD_H1,
                market_state_params=MARKET_STATE_CONFIG,
                trend_weights=TREND_INDICATOR_WEIGHTS,
                trend_thresholds=TREND_THRESHOLDS,
                confidence_thresholds=CONFIDENCE_THRESHOLDS,
            )
            analyzer.precompute_from_multitf(self.multi_tf)

        self.risk_controller = RiskController(
            self.data_provider,
            trade_direction=BACKTEST_CONFIG.get('trade_direction', 'both'),
            risk_config=RISK_CONFIG,
            market_state_analyzer=analyzer,
        )

        self.weight_manager = DynamicWeightManager(
            self.data_provider,
            market_state_analyzer=analyzer or MarketStateAnalyzer(),
        )

    def _instantiate_strategies(self, params_dict: dict = None) -> dict:
        """使用给定参数实例化所有策略"""
        return self.registry.instantiate_all(self.symbol, self.timeframe, params_dict)

    def precompute_signals(self, strategies: dict) -> pd.DataFrame:
        """预生成所有策略信号"""
        logger.info("开始预生成所有策略信号...")
        signals_df = pd.DataFrame(index=self.multi_tf.main_df.index)

        for config_key, strategy in tqdm(strategies.items(), desc="生成策略信号"):
            try:
                sig = strategy.run_backtest(self.multi_tf.main_df)
                if sig is not None and len(sig) > 0:
                    signals_df[config_key] = sig
                    logger.debug(f"策略 {config_key} 信号生成成功")
                else:
                    logger.warning(f"策略 {config_key} 返回空信号")
                    signals_df[config_key] = pd.Series(0, index=signals_df.index)
            except Exception as e:
                logger.error(f"策略 {config_key} 执行失败: {e}")
                signals_df[config_key] = pd.Series(0, index=signals_df.index)

        logger.info(f"信号预生成完成: {len(signals_df.columns)} 个策略")
        return signals_df

    def run(self, rates: np.ndarray,
            strategy_params: dict = None,
            weights: dict = None,
            buy_threshold: float = None,
            sell_threshold: float = None) -> dict:
        """执行完整回测

        Args:
            rates: MT5原始M1数据（numpy数组）
            strategy_params: {config_key: {param_name: value}} 策略参数覆写
            weights: {config_key: weight} 策略权重，None则使用 DEFAULT_WEIGHTS
            buy_threshold: 买入阈值
            sell_threshold: 卖出阈值

        Returns:
            dict: 交易摘要
        """
        self.load_data(rates)

        buy_th = buy_threshold or SIGNAL_THRESHOLDS.get('buy_threshold', 1.5)
        sell_th = sell_threshold or SIGNAL_THRESHOLDS.get('sell_threshold', -1.5)

        # 实例化策略并预计算信号
        strategies = self._instantiate_strategies(strategy_params)
        signals_df = self.precompute_signals(strategies)

        # 信号组合
        if weights is not None:
            use_weights = weights
        else:
            use_weights = DEFAULT_WEIGHTS

        combined_signals = SignalCombiner.combine_vectorized(
            signals_df, use_weights, buy_th, sell_th
        )

        logger.info("回测主循环开始...")
        total_bars = len(self.multi_tf.main_df)

        for bar_index in tqdm(range(total_bars), desc="回测执行"):
            try:
                current_price = self.data_provider.get_current_price(self.symbol)
                if not current_price:
                    self.data_provider.tick()
                    continue

                # 动态权重模式：每bar重新计算权重并重新组合信号
                if self.use_dynamic_weights and self.weight_manager.analyzer._precomputed_states is not None:
                    bar_weights = self.weight_manager.get_weights_for_bar(bar_index)
                    bar_signals = {col: signals_df[col].iloc[bar_index] for col in signals_df.columns}
                    current_signal = SignalCombiner.combine_at_bar(
                        bar_signals, bar_weights, buy_th, sell_th
                    )
                else:
                    current_signal = combined_signals.iloc[bar_index]
                if current_signal == 1:
                    self.risk_controller.process_trading_signal("buy", current_price, 1.0)
                elif current_signal == -1:
                    self.risk_controller.process_trading_signal("sell", current_price, 1.0)

                self.risk_controller.monitor_positions(current_price, dry_run=True)
                self.data_provider.tick()

            except Exception as e:
                logger.error(f"回测循环在索引 {bar_index} 出错: {e}")
                self.data_provider.tick()
                continue

        # 生成报告
        summary = self.risk_controller.position_manager.get_trade_summary()
        self._print_report(summary)

        # 保存交易记录
        try:
            self.risk_controller.save_trade_history("backtest_trades")
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")

        return summary

    def _print_report(self, summary: dict) -> None:
        logger.info("=" * 80)
        logger.info("回测性能报告")
        logger.info(f"总交易次数: {summary.get('total_trades', 0)}")
        logger.info(f"胜率: {summary.get('win_rate', 0):.2f}%")
        logger.info(f"总盈亏: ${summary.get('total_profit_loss', 0):.2f}")
        logger.info("=" * 80)
