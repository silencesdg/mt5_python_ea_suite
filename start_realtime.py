#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时交易启动脚本 (已重构)
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution.realtime_trader import RealtimeTrader
from core.data_providers import LiveDataProvider, DryRunDataProvider
from config import REALTIME_CONFIG, INITIAL_CAPITAL
from logger import setup_logger

def main():
    # 设置日志级别
    log_level = REALTIME_CONFIG.get('logging_level', 'INFO')
    setup_logger(log_level)
    
    # 从配置决定运行模式
    is_dry_run = REALTIME_CONFIG.get('dry_run', True)
    
    print("=" * 60)
    print("MetaTrader 5 智能交易系统")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: {'模拟运行' if is_dry_run else '实盘交易'}")
    print("=" * 60)

    # 根据模式选择数据提供者
    if is_dry_run:
        data_provider = DryRunDataProvider(initial_equity=INITIAL_CAPITAL)
    else:
        # 安全确认
        print("⚠️  警告：即将启动实盘交易模式！")
        confirm = input("确认启动实盘交易？(输入 'YES' 继续): ")
        if confirm != 'YES':
            print("已取消启动。")
            return
        data_provider = LiveDataProvider()

    # 创建并启动交易器实例
    trader = RealtimeTrader(data_provider, update_interval=REALTIME_CONFIG['update_interval'])
    
    print("\n正在启动交易系统... (按 Ctrl+C 可安全停止)")
    trader.start()

if __name__ == "__main__":
    main()