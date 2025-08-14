import pandas as pd
from core.utils import initialize, shutdown, get_rates
from logger import logger
from optimizer import run_optimizer

# Note: The run_backtest and run_realtime functions have been moved to 
# start_backtest.py and start_realtime.py respectively.
# This file is kept as a reference for the optimizer entry point.

if __name__ == "__main__":
    # --- 选择运行模式 ---
    # 1. 运行回测, 请执行 python start_backtest.py
    # run_backtest() 

    # 2. 运行实盘交易, 请执行 python start_realtime.py
    # from realtime_trader import RealtimeTrader
    # trader = RealtimeTrader()
    # trader.start()

    # 3. 运行遗传算法优化，寻找最佳权重
    # run_optimizer()
    pass