
import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from time import sleep

# --- 配置 ---
# 要获取的交易品种
SYMBOL = "XAUUSD"

# K线周期
TIMEFRAME = mt5.TIMEFRAME_M1

# 需要获取的数据的总起始日期
TOTAL_START_DATE = "2024-01-01"

# 输出文件名
OUTPUT_FILE = "full_historical_data.parquet"

# MT5单次请求的最大K线数 (保守起见，略低于10万)
CHUNK_SIZE = 90000

# --- 主函数 ---
def download_historical_data():
    """连接MT5，分块下载历史数据，并保存到Parquet文件。"""
    print("--- 开始历史数据下载任务 ---")

    # 连接到MetaTrader 5
    if not mt5.initialize():
        print("MT5初始化失败, 请检查终端是否开启。")
        mt5.shutdown()
        return

    print(f"已成功连接到MT5: {mt5.terminal_info()}")

    # 将字符串日期转换为datetime对象
    start_date_limit = datetime.strptime(TOTAL_START_DATE, '%Y-%m-%d')
    
    # 我们从现在开始，往回获取数据
    end_date_of_chunk = datetime.now()

    all_rates_df = []
    total_bars_fetched = 0

    while end_date_of_chunk > start_date_limit:
        print(f"正在获取 {end_date_of_chunk.strftime('%Y-%m-%d')} 之前的 {CHUNK_SIZE} 条数据...")
        
        # 从指定日期开始向前获取数据
        try:
            rates = mt5.copy_rates_from(SYMBOL, TIMEFRAME, end_date_of_chunk, CHUNK_SIZE)
        except Exception as e:
            print(f"从MT5获取数据时发生错误: {e}")
            rates = None

        if rates is None or len(rates) == 0:
            print("没有更多数据返回，或与服务器通信失败。结束下载。")
            break

        # 将元组列表转换为DataFrame
        rates_df = pd.DataFrame(rates)
        all_rates_df.append(rates_df)
        
        # 计算下一次请求的结束日期
        # 新的结束点是当前获取到的数据块中最早的时间点
        earliest_time_in_chunk = rates_df['time'].iloc[0]
        end_date_of_chunk = pd.to_datetime(earliest_time_in_chunk, unit='s')
        
        total_bars_fetched += len(rates)
        print(f"已获取 {len(rates)} 条数据。最早的数据点: {end_date_of_chunk.strftime('%Y-%m-%d')}, 总计已获取: {total_bars_fetched}")

        # 如果获取到的最早日期已经越过了我们的总起始日期，就停止
        if end_date_of_chunk <= start_date_limit:
            print("已达到设定的总起始日期，下载完成。")
            break
        
        # 短暂休眠，避免过于频繁地请求服务器
        sleep(0.5)

    # --- 数据处理 ---
    if not all_rates_df:
        print("未能获取到任何数据，程序退出。")
        mt5.shutdown()
        return

    print("\n--- 开始数据合并与清洗 ---")
    # 合并所有数据块
    full_df = pd.concat(all_rates_df, ignore_index=True)
    print(f"合并后总行数: {len(full_df)}")

    # 去除重复数据（基于时间戳）
    full_df.drop_duplicates(subset='time', inplace=True)
    print(f"去除重复后总行数: {len(full_df)}")

    # 将时间戳转换为datetime对象，并设置为索引
    full_df['time'] = pd.to_datetime(full_df['time'], unit='s')
    
    # 按时间排序
    full_df.sort_values('time', inplace=True)
    print("数据已按时间排序。")

    # 将time列设为索引
    full_df.set_index('time', inplace=True)

    # --- 保存文件 ---
    try:
        full_df.to_parquet(OUTPUT_FILE)
        print(f"\n--- 任务成功 ---")
        print(f"数据已成功保存到: {os.path.abspath(OUTPUT_FILE)}")
        print(f"数据范围: 从 {full_df.index[0]} 到 {full_df.index[-1]}")
        print(f"总计K线数量: {len(full_df)}")
    except Exception as e:
        print(f"保存到Parquet文件失败: {e}")

    # 关闭与MetaTrader 5的连接
    mt5.shutdown()

if __name__ == "__main__":
    download_historical_data()
