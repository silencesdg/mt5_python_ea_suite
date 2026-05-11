# MT5 代理服务配置
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5555

# 远端客户端配置（迁移到其他电脑时填写 MT5 机器的 IP）
REMOTE_SERVER_HOST = "192.168.1.5"
REMOTE_SERVER_PORT = 5555

# 数据提供者模式: "remote" (远程HTTP API) / "local" (本机MT5)
DATA_PROVIDER_MODE = "remote"

# 交易配置
SYMBOL = "XAUUSDz"
INITIAL_CAPITAL = 1944  # 初始资金（2026-05-12 实盘余额 $1944.27）

# 时间配置
TIMEFRAME = 1# M1 (1分钟图) - MT5常量值

# 回测时间范围 (格式: "YYYY-MM-DD")
BACKTEST_START_DATE = "2025-05-01"
BACKTEST_END_DATE = "2025-08-01"

# 优化器时间范围 (格式: "YYYY-MM-DD")
OPTIMIZER_START_DATE = "2025-04-01"
OPTIMIZER_END_DATE = "2025-05-01"

# 安全设置：如果日期获取失败，自动回退到数据量模式
USE_DATE_RANGE = False  # 设置为False可强制使用数据量模式

# 兼容性配置 (如果日期配置不可用，则使用数据量)
BACKTEST_COUNT = 30000  # 回测数据量
OPTIMIZER_COUNT = 50000  # 优化器数据量

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

# 对冲配置（信号对冲 + 回撤锁仓）
HEDGE_CONFIG = {
    # ── 信号对冲 ──
    "signal_hedge_enabled": True,       # 启用信号对冲
    "signal_hedge_threshold": 2.0,      # 加权信号绝对值超此值触发对冲
    "signal_hedge_ratio": 0.5,          # 对冲手数比例 (0.5=半仓对冲)
    "signal_unhedge_threshold": 1.0,    # 信号回到此值以下解锁

    # ── 回撤锁仓 ──
    "drawdown_hedge_enabled": True,     # 启用回撤锁仓
    "drawdown_hedge_pct": -0.003,       # 浮亏超-0.3%触发锁仓
    "drawdown_hedge_ratio": 1.0,        # 锁仓比例 (1.0=全额锁仓)

    # ── 对冲单止盈 ──
    "hedge_take_profit_pct": 0.005,     # 对冲单自身盈利0.5%止盈

    # ── 风控限制 ──
    "max_hedges_per_day": 5,            # 每日最多对冲5次
}


# 数据获取配置
DATA_CONFIG = {
    "m1_bars_count": 5000,        # 1分钟K线数据获取数量
}


# 遗传算法优化器配置
GENETIC_OPTIMIZER_CONFIG = {
    # 算法参数
    "population_size": 50,         # 种群大小
    "generations": 10,           # 进化代数
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

# 信号阈值配置（优化器结果 2026-05-10，5万根M1数据）
SIGNAL_THRESHOLDS = {
    "buy_threshold": 1.344,
    "sell_threshold": -2.980
}


# 风险管理参数（优化器结果 2026-05-10，5万根M1数据）
RISK_CONFIG = {
    "stop_loss_pct": -0.046,
    "profit_retracement_pct": 0.070,
    "min_profit_for_trailing": 0.009,
    "take_profit_pct": 0.246,
    "max_daily_loss": -0.3,
    "max_holding_minutes": 133,
    "min_profit_for_time_exit": 0.010,
    "cooldown_bars": 30
}

# 市场状态分析参数
MARKET_STATE_CONFIG = {
    "trend_period": 24,
    "retracement_tolerance": 0.425,
    "volume_period": 21,
    "volume_ma_period": 12
}

# 策略参数配置（优化器结果 2026-05-10，5万根M1数据）
STRATEGY_CONFIG = {
    "ma_cross": {
        "short_window": 12,
        "long_window": 30
    },
    "rsi": {
        "period": 21,
        "overbought": 75,
        "oversold": 26
    },
    "bollinger": {
        "period": 20,
        "std_dev": 2.162
    },
    "macd": {
        "fast_ema": 16,
        "slow_ema": 34,
        "signal_period": 12
    },
    "mean_reversion": {
        "period": 29,
        "std_dev": 2.149
    },
    "momentum_breakout": {
        "period": 15,
        "momentum_period": 17
    },
    "kdj": {
        "period": 21
    },
    "turtle": {
        "period": 42
    },
    "daily_breakout": {
        "bars_count": 746
    },
    "wave_theory": {
        "ema_short": 3,
        "ema_medium": 16,
        "ema_long": 26,
        "wave_period": 32,
        "range_period": 30,
        "adx_period": 23,
        "momentum_period": 10,
        "range_threshold": 0.002,
        "adx_threshold": 23
    }
}

# 市场趋势判断权重配置
TREND_INDICATOR_WEIGHTS = {
    "price_breakout": -0.0949,
    "volume_confirmation": 0.5277,
    "momentum oscillator": 0.3677,
    "moving_average": 0.1506
}

# 趋势判断阈值
TREND_THRESHOLDS = {
    "strong_trend": 0.4182,
    "weak_trend": 0.2164,
    "volume_spike": 1.5843,
    "oversold": 24,
    "overbought": 80
}


# 动态权重配置（优化器结果 2026-05-10，5万根M1数据）
DEFAULT_WEIGHTS = {
    "ma_cross": 0.809,
    "rsi": 1.141,
    "bollinger": 0.389,
    "mean_reversion": 1.106,
    "momentum_breakout": 0.147,
    "macd": 1.181,
    "kdj": 1.525,
    "turtle": 0.559,
    "daily_breakout": 1.846,
    "wave_theory": 1.361
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
    "high_confidence": 0.8474,
    "medium_confidence": 0.4964
}
