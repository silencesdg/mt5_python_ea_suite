
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# --- 本地模块导入 ---
# 导入波浪理论策略以使用其指标计算和状态判断逻辑
from strategies.wave_theory import WaveTheoryStrategy

# 导入机器学习优化器的核心函数
# (注意：我们可能需要对optimizer_ml.py稍作修改，使其核心功能可以被调用)
from optimizer_ml import run_optimizer as run_ml_optimizer, load_historical_data

# --- 配置 ---
DATA_FILE = "full_historical_data.parquet"
CLASSIFIED_DATA_FILE = "full_historical_data_classified.parquet"
RESULTS_FILE = "regime_optimal_params.json"

# --- 核心功能 ---

def classify_market_regimes(df):
    """
    使用WaveTheoryStrategy中的逻辑为整个数据集打上市场状态标签。
    返回一个带有 'regime' 列的 DataFrame。
    """
    print("--- 开始为数据集分类市场状态 ---")
    
    # 使用默认参数实例化策略，我们只借用它的指标计算能力
    # 这里的参数值不影响状态判断的逻辑本身
    strategy = WaveTheoryStrategy(data_provider=None, symbol="XAUUSD", timeframe=None)
    
    # 1. 计算所有需要的指标
    # 我们需要确保传入一个副本，避免修改原始df
    df_with_indicators = strategy._calculate_indicators(df.copy())
    
    # 2. 逐行判断市场状态
    regimes = []
    # 使用.values可以稍微加速循环
    adx_values = df_with_indicators['adx'].values
    ema_short_values = df_with_indicators['ema_short'].values
    ema_medium_values = df_with_indicators['ema_medium'].values
    ema_long_values = df_with_indicators['ema_long'].values
    
    # 从实例中获取阈值
    adx_threshold = strategy.adx_threshold
    
    for i in range(len(df_with_indicators)):
        # 处理NaN值的情况，在数据初期指标未生成时，设为None
        if np.isnan(adx_values[i]) or np.isnan(ema_short_values[i]) or np.isnan(ema_medium_values[i]) or np.isnan(ema_long_values[i]):
            regimes.append(None)
            continue

        # 判断震荡
        # 注意：原始逻辑中还有一个range_pct的判断，这里为简化，暂时只用ADX和均线，后续可完善
        if adx_values[i] < adx_threshold:
            regimes.append("Ranging")
        # 判断上涨趋势
        elif ema_short_values[i] > ema_medium_values[i] > ema_long_values[i]:
            regimes.append("Uptrend")
        # 其他情况视为下跌趋势
        else:
            regimes.append("Downtrend")
            
    df_with_indicators['regime'] = regimes
    
    # 打印各类别的统计信息
    print("\n市场状态分类统计:")
    print(df_with_indicators['regime'].value_counts())
    
    # 删除那些没有成功分类的行
    df_with_indicators.dropna(subset=['regime'], inplace=True)
    print(f"\n去除无法分类的初期数据后，剩余总行数: {len(df_with_indicators)}")
    
    return df_with_indicators

def run_regime_based_optimization():
    """
    主函数：加载数据，分类状态，并为每个状态分别运行优化器。
    """
    # 1. 加载或生成带有市场状态标签的数据
    if os.path.exists(CLASSIFIED_DATA_FILE):
        print(f"正在从已分类的数据文件 {CLASSIFIED_DATA_FILE} 加载...")
        df_classified = pd.read_parquet(CLASSIFIED_DATA_FILE)
        print(f"已分类数据加载完成，共 {len(df_classified)} 条记录。")
    else:
        print(f"未找到已分类的数据文件，将执行初次分类...")
        if not os.path.exists(DATA_FILE):
            print(f"错误: 原始数据文件 {DATA_FILE} 不存在。请先运行 data_downloader.py。")
            return
        
        print(f"正在从 {DATA_FILE} 加载原始数据...")
        full_df = pd.read_parquet(DATA_FILE)
        print(f"原始数据加载完成，共 {len(full_df)} 条记录。")

        # 为整个数据集打上状态标签
        df_classified = classify_market_regimes(full_df)
        
        # 保存已分类的数据以备将来使用
        print(f"正在将分类后的数据保存到 {CLASSIFIED_DATA_FILE}...")
        try:
            df_classified.to_parquet(CLASSIFIED_DATA_FILE)
            print("保存完成。")
        except Exception as e:
            print(f"保存已分类数据时出错: {e}")
    
    # 3. 按状态分组，并分别进行优化
    all_regime_params = {}
    regimes_to_optimize = ['Uptrend', 'Downtrend', 'Ranging']
    
    for regime in regimes_to_optimize:
        print(f"\n{'='*80}\n--- 开始为【{regime}】市场状态进行参数优化 ---\n{'='*80}")
        
        regime_df = df_classified[df_classified['regime'] == regime].copy()
        
        if len(regime_df) < 5000: # 如果某个状态的数据太少，优化可能无意义
            print(f"警告: {regime} 状态的数据量过少 ({len(regime_df)}条)，跳过优化。")
            continue
            
        # 调用ML优化器，并把特定状态的数据传递给它
        # *** 注意：这需要我们修改 optimizer_ml.py 中的 run_optimizer 函数，使其能接收一个DataFrame作为参数 ***
        # *** 目前，我们先假设这个修改已经完成 ***
        try:
            best_params, best_sharpe = run_ml_optimizer(regime_df)
            all_regime_params[regime] = {
                'best_sharpe': best_sharpe,
                'best_parameters': best_params
            }
            print(f"\n---【{regime}】状态优化完成！最佳夏普比率: {best_sharpe:.4f} ---")
        except Exception as e:
            print(f"为 {regime} 状态优化时发生严重错误: {e}")
            import traceback
            print(traceback.format_exc())

    # 4. 保存最终结果
    if all_regime_params:
        print(f"\n{'='*80}\n--- 所有市场状态优化全部完成！---\n{'='*80}")
        print("最终结果:")
        print(json.dumps(all_regime_params, indent=2))
        
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_regime_params, f, ensure_ascii=False, indent=4)
            
        print(f"\n结果已保存到: {os.path.abspath(RESULTS_FILE)}")
    else:
        print("\n没有完成任何状态的优化。")

if __name__ == "__main__":
    run_regime_based_optimization()
