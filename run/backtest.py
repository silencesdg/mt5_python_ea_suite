#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回测入口 — 支持远程/本地两种数据源"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime

from core.data.remote import RemoteDataProvider
from execution.backtest import BacktestEngine
from config import (
    SYMBOL, TIMEFRAME, BACKTEST_COUNT,
    BACKTEST_START_DATE, BACKTEST_END_DATE, USE_DATE_RANGE,
    REMOTE_SERVER_HOST, REMOTE_SERVER_PORT, DATA_PROVIDER_MODE,
)
from logger import logger

# MT5 rates 结构化数组 dtype
MT5_RATES_DTYPE = np.dtype([
    ('time', 'i8'),
    ('open', 'f8'),
    ('high', 'f8'),
    ('low', 'f8'),
    ('close', 'f8'),
    ('tick_volume', 'i8'),
    ('spread', 'i4'),
    ('real_volume', 'i8'),
])


def fetch_remote_rates(symbol, timeframe, count):
    """通过远程 API 获取历史 K 线，转成 MT5 兼容的 numpy 数组"""
    logger.info(f"使用远程数据源 {REMOTE_SERVER_HOST}:{REMOTE_SERVER_PORT}")
    remote = RemoteDataProvider(host=REMOTE_SERVER_HOST, port=REMOTE_SERVER_PORT)
    if not remote.initialize():
        logger.error("远程API初始化失败")
        return None

    rates_data = remote.get_historical_data(symbol, timeframe, count)
    remote.shutdown()

    if not rates_data:
        logger.error(f"远程API获取 {symbol} TF{timeframe} 历史数据失败")
        return None

    # 转成 MT5 的 numpy 结构化数组格式
    records = []
    for r in rates_data:
        records.append((
            r['time'],
            r['open'],
            r['high'],
            r['low'],
            r['close'],
            r.get('tick_volume', 0),
            r.get('spread', 0),
            r.get('real_volume', 0),
        ))
    rates = np.array(records, dtype=MT5_RATES_DTYPE)
    logger.info(f"远程获取数据: {len(rates)} 条")
    return rates


def main():
    print("=" * 60)
    print("MT5 智能交易系统 - 回测")
    print(f"数据源: {DATA_PROVIDER_MODE.upper()}")
    print(f"品种: {SYMBOL}  |  周期: M{TIMEFRAME}  |  数据量: {BACKTEST_COUNT}")
    print("=" * 60)

    try:
        if DATA_PROVIDER_MODE == "remote":
            rates = fetch_remote_rates(SYMBOL, TIMEFRAME, BACKTEST_COUNT)
        else:
            from core.utils import initialize, shutdown, get_rates
            initialize()
            if USE_DATE_RANGE:
                rates = get_rates(SYMBOL, TIMEFRAME, BACKTEST_COUNT,
                                  BACKTEST_START_DATE, BACKTEST_END_DATE)
            else:
                rates = get_rates(SYMBOL, TIMEFRAME, BACKTEST_COUNT)
            shutdown()

        if rates is None or len(rates) == 0:
            logger.error("未能获取历史数据，回测终止。")
            return

        logger.info(f"获取数据: {len(rates)} 条")

        engine = BacktestEngine(use_dynamic_weights=False)
        summary = engine.run(rates)
        print(f"\n回测完成！总盈亏: ${summary.get('total_profit_loss', 0):.2f}")

    except Exception as e:
        import traceback
        logger.error(f"回测出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
