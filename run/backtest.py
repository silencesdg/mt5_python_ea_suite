#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回测入口 — python -m run.backtest"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import get_rates, initialize, shutdown
from execution.backtest import BacktestEngine
from config import SYMBOL, TIMEFRAME, BACKTEST_COUNT, BACKTEST_START_DATE, BACKTEST_END_DATE, USE_DATE_RANGE
from logger import logger


def main():
    print("=" * 60)
    print("MT5 智能交易系统 - 回测")
    print("=" * 60)

    try:
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
