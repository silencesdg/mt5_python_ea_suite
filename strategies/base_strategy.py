#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略基类
"""

from logger import logger

class BaseStrategy:
    """策略基类 (已重构为依赖注入)"""
    
    def __init__(self, data_provider, symbol, timeframe):
        self.data_provider = data_provider
        self.symbol = symbol
        self.timeframe = timeframe
        self.name = self.__class__.__name__
        
    def set_params(self, params):
        """动态设置策略参数"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"{self.name}: 参数 '{key}' 已更新为 {value}")
            else:
                logger.warning(f"{self.name}: 尝试设置不存在的参数 '{key}'")

    def _log_signal(self, signal, reason=None):
        """记录信号日志"""
        if signal != 0:
            direction = "买入" if signal > 0 else "卖出"
            logger.info(f"{self.name}: 生成 {direction} 信号 (强度: {signal:.2f}), 原因: {reason}")
        else:
            logger.debug(f"{self.name}: 无信号, 原因: {reason}")
    
    def generate_signal(self):
        """生成交易信号 - 子类必须实现此方法"""
        raise NotImplementedError("子类必须实现 generate_signal 方法")
    
    def run_backtest(self, df):
        """回测方法 - 子类必须实现此方法"""
        raise NotImplementedError("子类必须实现 run_backtest 方法")
