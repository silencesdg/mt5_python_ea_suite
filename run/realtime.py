#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时交易入口 — 支持远程/本地两种数据源"""

import sys
import os
import fcntl
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.realtime_trader import RealtimeTrader
from core.data import LiveDataProvider, DryRunDataProvider, RemoteDataProvider
from config import (
    REALTIME_CONFIG, INITIAL_CAPITAL,
    REMOTE_SERVER_HOST, REMOTE_SERVER_PORT, DATA_PROVIDER_MODE,
)
from logger import setup_logger

LOCK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.ea.lock')


def main():
    # ★ 进程锁：防止多实例同时运行
    lock_fd = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("❌ 已有 EA 实例在运行，退出")
        os.close(lock_fd)
        return
    log_level = REALTIME_CONFIG.get('logging_level', 'INFO')
    setup_logger(log_level)

    is_dry_run = REALTIME_CONFIG.get('dry_run', True)

    print("=" * 60)
    print("MT5 智能交易系统 - 实时交易")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据源: {DATA_PROVIDER_MODE.upper()}")
    print(f"运行模式: {'模拟运行' if is_dry_run else '实盘交易'}")
    print("=" * 60)

    if DATA_PROVIDER_MODE == "remote":
        data_provider = RemoteDataProvider(host=REMOTE_SERVER_HOST, port=REMOTE_SERVER_PORT)
        if not data_provider.initialize():
            print("❌ 远程 MT5 API 连接失败，请检查 Windows 端服务是否运行")
            return
        print(f"✅ 已连接远程 MT5 {REMOTE_SERVER_HOST}:{REMOTE_SERVER_PORT}")
        acct = data_provider.get_account_info()
        if acct:
            print(f"   账号: {acct.get('login','?')}  余额: ${acct.get('balance','?'):.2f}")
    elif is_dry_run:
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

    # 写入 PID 文件供重启脚本使用
    pid_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.ea_pid')
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    trader.start()


if __name__ == "__main__":
    main()
