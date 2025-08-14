# 交易配置
SYMBOL = "XAUUSD"
INTERVAL = 60  # 秒
INITIAL_CAPITAL = 20000  # 初始资金
SPREAD = 32  # 点差（点数）,买卖合起来的总共点差成本

# 时间配置
TIMEFRAME = 1# M1 (1分钟图) - MT5常量值

# 回测时间范围 (格式: "YYYY-MM-DD")
# 注意：这些日期需要确保在MT5服务器上有可用数据
BACKTEST_START_DATE = "2025-05-01"
BACKTEST_END_DATE = "2025-08-01"

# 优化器时间范围 (格式: "YYYY-MM-DD")
# 注意：确保与回测时间不重叠
OPTIMIZER_START_DATE = "2025-04-01"
OPTIMIZER_END_DATE = "2025-05-01"

# 安全设置：如果日期获取失败，自动回退到数据量模式
USE_DATE_RANGE = False  # 设置为False可强制使用数据量模式

# 兼容性配置 (如果日期配置不可用，则使用数据量)
BACKTEST_COUNT = 30000  # 回测数据量
OPTIMIZER_COUNT = 15000  # 优化器数据量

RISK_CONFIG_CONST = {
    'enable_time_based_exit': True
}

# 资金分配配置
CAPITAL_ALLOCATION = {
    "long_pct": 0.7,   # 多头持仓分配资金比例
    "short_pct": 0.3,  # 空头持仓分配资金比例
}

# 模拟交易特定配置 (用于dry_run模式)
SIMULATION_CONFIG = {
    "leverage": 100,              # 模拟杠杆
    "contract_size": 1,         # XAUUSD的合约大小
    "volume_step": 0.01,          # 交易手数步长
    "volume_min": 0.01,           # 最小交易手数
    "volume_max": 100.0,          # 最大交易手数
    "spread": 16,                 # 点差（点数）
}

# 回测配置
BACKTEST_CONFIG = {
    "trade_direction": "both",       # 交易方向: "long"(只做多), "short"(只做空), "both"(多空都支持)
    "spread": 16,                    # 点差（点数）
}


# 实时交易配置
REALTIME_CONFIG = {
    "update_interval": 5,           # 更新间隔（秒）
    "daily_reset_time": "00:00",     # 每日重置时间
    "max_long_positions": 3,        # 最大多头持仓数（增加为3个）
    "max_short_positions": 3,       # 最大空头持仓数（增加为3个）
    "min_trade_interval": 0,         # 最小交易间隔（分钟），0表示无限制
    "enable_auto_trading": True,     # 是否启用自动交易
    "dry_run": False,                 # 是否为模拟运行（不实际下单）
    "logging_level": "DEBUG",         # 日志级别
    "trade_direction": "both",       # 交易方向: "long"(只做多), "short"(只做空), "both"(多空都支持)
}


# 数据获取配置
DATA_CONFIG = {
    "m1_bars_count": 5000,        # 1分钟K线数据获取数量
}


# 遗传算法优化器配置
GENETIC_OPTIMIZER_CONFIG = {
    # 算法参数
    "population_size": 200,        # 种群大小
    "generations": 50,           # 进化代数
    "crossover_probability": 0.7, # 交叉概率
    "mutation_probability": 0.3,  # 变异概率
    
    # 选择算法参数
    "tournament_size": 3,        # 锦标赛选择大小
    
    # 变异算法参数
    "mutation_mu": 0,            # 变异均值
    "mutation_sigma": 0.1,       # 变异标准差
    "mutation_indpb": 0.1,       # 变异概率（每个基因）
    
    # 并行处理
    "enable_multiprocessing": True,  # 启用多进程
    "processes": None,           # 进程数，None表示自动检测
    
    # 输出控制
    "verbose": True,             # 详细输出
    "save_generation_info": True, # 保存代数信息
}


'''
此处上面的是固定的参数，可手动调整
--------------------------------------------------------
此处下面所有参数，都将进入优化器进行优化
'''

# 信号阈值配置
SIGNAL_THRESHOLDS = {
    "buy_threshold": 1.5,     # 买入信号阈值 (优化后)
    "sell_threshold": -0.87   # 卖出信号阈值 (优化后)
}


# 风险管理参数
RISK_CONFIG = {
    "stop_loss_pct": -0.01,        # 固定止损：亏损1% (优化后)
    "profit_retracement_pct": 0.1,  # 利润回撤10%止盈 (优化后)
    "min_profit_for_trailing": 0.01,  # 追踪止损激活阈值：利润0.1% (降低阈值以激活追踪止损)
    "take_profit_pct": 0.001,         # 固定止盈：盈利0.1% (优化后)
    "max_position_size": 1,         # 最大仓位100%
    "max_daily_loss": -0.3,          # 最大日亏损30%
    "max_holding_minutes": 140,      # 持仓超过140分钟 (优化后)
    "min_profit_for_time_exit": 0.01, # 且盈利未达到0.1%则平仓 (优化后)
}

# 市场状态分析参数
MARKET_STATE_CONFIG = {
    "trend_period": 44,
    "retracement_tolerance": 0.1382775008306345,
    "volume_period": 13,
    "volume_ma_period": 12,
}

# 策略参数配置 (从优化器中提取的优化参数)
STRATEGY_CONFIG = {
    # MACrossStrategy 参数
    "ma_cross": {
        "short_window": 5,
        "long_window": 39,
    },
    
    # RSIStrategy 参数
    "rsi": {
        "period": 22,
        "overbought": 80,
        "oversold": 24,
    },
    
    # BollingerStrategy 参数
    "bollinger": {
        "period": 10,
        "std_dev": 2.8916513144581044,
    },
    
    # MACDStrategy 参数
    "macd": {
        "fast_ema": 17,
        "slow_ema": 22,
        "signal_period": 13,
    },
    
    # MeanReversionStrategy 参数
    "mean_reversion": {
        "period": 40,
        "std_dev": 2.917216289407233,
    },
    
    # MomentumBreakoutStrategy 参数
    "momentum_breakout": {
        "period": 27,
    },
    
    # KDJStrategy 参数
    "kdj": {
        "period": 6,
    },
    
    # TurtleStrategy 参数
    "turtle": {
        "period": 16,
    },
    
    # DailyBreakoutStrategy 参数
    "daily_breakout": {
        "bars_count": 807,
    },
    
    # WaveTheoryStrategy 参数
    "wave_theory": {
        "ema_short": 3,
        "ema_medium": 15,
        "ema_long": 36,
        "wave_period": 28,
        "range_period": 30,
        "adx_period": 21,
        "momentum_period": 13,
        "range_threshold": -0.02028040971155598,
        "adx_threshold": 22,
    },
}

# 市场趋势判断权重配置
TREND_INDICATOR_WEIGHTS = {
    "price_breakout": 0.03552729699860058,      # 价格突破权重
    "volume_confirmation": 0.2564191814903339, # 成交量确认权重
    "momentum oscillator": 0.48690789892001296, # 动量震荡指标权重
    "moving_average": 0.3549274308680269,     # 移动平均线权重
}

# 趋势判断阈值
TREND_THRESHOLDS = {
    "strong_trend": 0.24063407379490956,         # 强趋势阈值
    "weak_trend": 0.3708545035763192,           # 弱趋势阈值
    "volume_spike": 1.7781348694952452,         # 成交量突增倍数
    "oversold": 29,              # 超卖阈值 (RSI)
    "overbought": 69,            # 超买阈值 (RSI)
}


# 动态权重配置（经过优化器优化的最佳权重）
DEFAULT_WEIGHTS = {
    "ma_cross": 0.44428771408324463,
    "rsi": 1.6518277343219148,
    "bollinger": 0.6014474871460991,
    "mean_reversion": 0.04509729271515517,
    "momentum_breakout": 1.0768245883204248,
    "macd": 0.8909161638775971,
    "kdj": 1.095784002572773,
    "turtle": 1.7942934575029192,
    "daily_breakout": 2.4237777692491713,
    "wave_theory": 0.7354173414675755,
}

# 市场状态策略权重配置
MARKET_STATE_WEIGHTS = {
    "uptrend": {
        "ma_cross": 1.50,
        "momentum_breakout": 1.20,
        "turtle": 0.25,
        "macd": 0.35,
        "daily_breakout": 1.50,
        "rsi": 1.00,
        "bollinger": 1.00,
        "kdj": 0.40,
        "mean_reversion": 0.80,
        "wave_theory": 0.20
    },
    "downtrend": {
        "ma_cross": 1.50,
        "momentum_breakout": 1.20,
        "turtle": 0.25,
        "macd": 0.35,
        "daily_breakout": 1.50,
        "rsi": 1.00,
        "bollinger": 1.00,
        "kdj": 0.40,
        "mean_reversion": 0.80,
        "wave_theory": 0.20
    },
    "ranging": {
        "rsi": 1.60,
        "bollinger": 1.70,
        "mean_reversion": 1.50,
        "kdj": 1.00,
        "wave_theory": 0.50,
        "ma_cross": 0.70,
        "macd": 0.20,
        "turtle": 0.10,
        "momentum_breakout": 0.50,
        "daily_breakout": 0.90
    },
    "none": DEFAULT_WEIGHTS
}

# 市场趋势置信度阈值配置
CONFIDENCE_THRESHOLDS = {
    "high_confidence": 0.941049792261965,       # 高置信度阈值
    "medium_confidence": 0.8881796011658835,     # 中等置信度阈值
}

