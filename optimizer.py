
import random
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing

import pandas as pd
from utils import initialize, shutdown, get_rates
from backtest import BacktestEngine
from logger import logger
from config import INITIAL_CAPITAL, SYMBOL, TIMEFRAME, OPTIMIZER_COUNT, OPTIMIZER_START_DATE, OPTIMIZER_END_DATE, USE_DATE_RANGE
from risk_management import RiskController
from dynamic_weights import DynamicWeightManager

# 1. 定义适应度函数 (已优化)
def evaluate_fitness(individual, df_data):
    """
    评估函数现在接收预先加载的DataFrame作为参数，避免了重复IO。
    使用新的动态权重架构进行评估。
    输入:
    - individual: 一个代表策略权重的列表。
    - df_data: 包含历史K线数据的Pandas DataFrame。
    输出: 一个元组，包含适应度分数（最终资金）。
    """
    weights = individual

    # 使用新的架构进行评估
    engine = BacktestEngine(df_data)
    
    # 创建风险管理器和权重管理器
    risk_controller = RiskController()
    weight_manager = DynamicWeightManager(risk_controller)
    
    # 获取策略实例
    strategies_with_weights = weight_manager.get_current_strategies_and_weights()
    
    # 使用优化器提供的权重替换动态权重
    signals_list = []
    strategy_names = []
    
    for i, (strat, _) in enumerate(strategies_with_weights):
        if i < len(weights):
            signals = engine.run_strategy(strat)
            signals_list.append(signals)
            strategy_names.append(strat.__class__.__module__)
    
    # 使用优化器权重进行组合
    combined_signal = engine.combine_signals(signals_list, weights[:len(signals_list)])
    cum_ret = engine.calc_returns(combined_signal)
    final_capital = INITIAL_CAPITAL * (1 + cum_ret.iloc[-1])

    # 在优化过程中，可以注释掉这行日志以提高速度，因为它会大量输出
    # logger.info(f"评估权重: {[f'{w:.2f}' for w in weights[:len(signals_list)]]} -> 最终资金: {final_capital:.2f}")

    return (final_capital,)

# 2. 设置遗传算法 (已优化)
def run_optimizer():
    """
    配置并运行遗传算法
    """
    # --- 数据预加载 ---
    logger.info("--- 开始遗传算法优化 --- ")
    logger.info("步骤 1/4: 初始化MT5并预加载历史数据...")
    if not initialize():
        logger.error("MT5初始化失败，无法开始优化")
        return

    # 根据配置选择获取数据的方式
    if USE_DATE_RANGE:
        rates = get_rates(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT, OPTIMIZER_START_DATE, OPTIMIZER_END_DATE)
    else:
        rates = get_rates(SYMBOL, TIMEFRAME, OPTIMIZER_COUNT)
    shutdown() # 获取数据后即可关闭连接

    if rates is None:
        logger.error("获取历史数据失败，优化终止")
        return

    df_historical_data = pd.DataFrame(rates)
    logger.info(f"历史数据加载完成，共 {len(df_historical_data)} 条记录。")

    # --- DEAP 设置 ---
    logger.info("步骤 2/4: 配置遗传算法...")
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.1, 2.0)
    
    # 使用动态权重管理器获取策略数量
    risk_controller = RiskController()
    weight_manager = DynamicWeightManager(risk_controller)
    num_strategies = len(weight_manager.list_available_strategies())
    
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_strategies)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_fitness, df_data=df_historical_data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- 并行计算设置 ---
    logger.info("步骤 3/4: 配置并行计算和统计...")
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # --- 统计功能设置 ---
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # --- 运行算法 ---
    population = toolbox.population(n=50)
    ngen = 20
    cxpb = 0.5
    mutpb = 0.2

    logger.info(f"步骤 4/4: 开始并行进化... (种群大小: {len(population)}, 进化代数: {ngen})")
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=stats, verbose=True)

    pool.close()

    # --- 结果 ---
    best_individual = tools.selBest(population, k=1)[0]
    best_fitness = best_individual.fitness.values[0]

    logger.info("--- 遗传算法优化结束 ---")
    logger.info(f"找到的最佳权重: {[f'{w:.2f}' for w in best_individual]}")
    logger.info(f"对应的最佳最终资金: {best_fitness:.2f}")

    return best_individual, best_fitness
