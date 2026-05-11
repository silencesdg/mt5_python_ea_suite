#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时交易入口 — python -m run.realtime"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.realtime_trader import RealtimeTrader
from core.data import LiveDataProvider, DryRunDataProvider
from config import REALTIME_CONFIG, INITIAL_CAPITAL
from logger import setup_logger


def main():
    log_level = REALTIME_CONFIG.get('logging_level', 'INFO')
    setup_logger(log_level)

    is_dry_run = REALTIME_CONFIG.get('dry_run', True)

    print("=" * 60)
    print("MT5 智能交易系统 - 实时交易")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: {'模拟运行' if is_dry_run else '实盘交易'}")
    print("=" * 60)

    if is_dry_run:
        data_provider = DryRunDataProvider(initial_equity=INITIAL_CAPITAL)
    else:
        print("WARNING: 即将启动实盘交易模式！")
        confirm = input("确认启动实盘交易？(输入 'YES' 继续): ")
        if confirm != 'YES':
            print("已取消启动。")
            return
        data_provider = LiveDataProvider()

    trader = RealtimeTrader(data_provider, update_interval=REALTIME_CONFIG['update_interval'])
    print("\n正在启动交易系统... (按 Ctrl+C 可安全停止)")
    trader.start()


if __name__ == "__main__":
    main()
