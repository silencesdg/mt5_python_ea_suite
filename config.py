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
INITIAL_CAPITAL = 2054  # 初始资金（5月12日 实盘余额 $2054.23）

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
    'enable_time_based_exit': False  # 关闭超时平仓，让止盈/止损/跟踪止损接管
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
    "max_long_positions": 1,        # 同方向只持一单，避免重复开仓
    "max_short_positions": 1,       # 同方向只持一单
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

    # ── 锁仓管理 ──
    "lock_net_profit_pct": 0.0,         # 锁仓组合净盈利>0→双平离场

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
    "buy_threshold": 1.0315663601353515,
    "sell_threshold": -2.312867704316165
}


# 风险管理参数（优化器结果 2026-05-10，5万根M1数据）
RISK_CONFIG = {
    # ★ 以下百分比均为保证金%（开仓成本%），非金价涨跌%、非账户%
    #    risk_leverage 控制计算杠杆：设100则0.01手按$47保证金算%
    #    例：亏损$25 / 保证金$47 = -53%，而不是 $25/账户$1682=-1.5%
    "risk_leverage": 100,               # 计算杠杆（实际杠杆不变，仅影响%基准）
    "stop_loss_pct": -0.50,            # -50% 保证金（亏一半保证金就平）
    "profit_retracement_pct": 0.35,    # 35% 保证金（拖尾回撤容忍）
    "min_profit_for_trailing": 0.70,   # +70% 保证金后激活拖尾
    "take_profit_pct": 1.00,           # +100% 保证金（赚一倍保证金就平）
    "max_daily_loss": -0.3,
    "max_holding_minutes": 133,
    "min_profit_for_time_exit": 0.010,
    "cooldown_bars": 30,
    # ★ 硬止损倍率：MT5 服务器端 SL/TP = 软止损 × 倍率（兜底，仅 EA 挂掉时触发）
    "hard_sl_multiplier": 1.5,   # 硬 SL = 软 SL × 1.5（-1.5%→-2.25%账户）
    "hard_tp_multiplier": 1.3,   # 硬 TP = 软 TP × 1.3（+3.0%→+3.9%账户）
}

# 市场状态分析参数
MARKET_STATE_CONFIG = {
    "trend_period": 83,
    "retracement_tolerance": 0.15567164907853615,
    "volume_period": 29,
    "volume_ma_period": 23
}

# 策略参数配置（优化器结果 2026-05-10，5万根M1数据）
STRATEGY_CONFIG = {
    "ma_cross": {
        "short_window": 13,
        "long_window": 17
    },
    "rsi": {
        "period": 30,
        "overbought": 68,
        "oversold": 32
    },
    "bollinger": {
        "period": 20,
        "std_dev": 2.4663422649047333
    },
    "macd": {
        "fast_ema": 14,
        "slow_ema": 24,
        "signal_period": 9
    },
    "mean_reversion": {
        "period": 29,
        "std_dev": 2.149
    },
    "momentum_breakout": {
        "period": 15,
        "momentum_period": 13
    },
    "kdj": {
        "period": 21
    },
    "turtle": {
        "period": 42
    },
    "daily_breakout": {
        "bars_count": 2008
    },
    "wave_theory": {
        "ema_short": 3,
        "ema_medium": 11,
        "ema_long": 46,
        "wave_period": 11,
        "range_period": 35,
        "adx_period": 20,
        "momentum_period": 10,
        "range_threshold": 0.01,
        "adx_threshold": 21
    }
}

# 市场趋势判断权重配置
TREND_INDICATOR_WEIGHTS = {
    "price_breakout": 0.1504199442999585,
    "volume_confirmation": 0.12782205952949638,
    "momentum oscillator": 0.3677,
    "moving_average": 0.4092273363154768
}

# 趋势判断阈值
TREND_THRESHOLDS = {
    "strong_trend": 0.7940886082643032,
    "weak_trend": 0.4565953163045441,
    "volume_spike": 2.7329673335105396,
    "oversold": 32,
    "overbought": 68
}


# 动态权重配置（优化器结果 2026-05-10，5万根M1数据）
DEFAULT_WEIGHTS = {
    "ma_cross": 0.32048324544087786,
    "rsi": 1.906009222972592,
    "bollinger": 0.9232444456662969,
    "mean_reversion": 1.1649306516234597,
    "momentum_breakout": 0.8769099086178171,
    "macd": 0.748930985323352,
    "kdj": 1.4647470623067596,
    "turtle": 1.637931136008917,
    "daily_breakout": 1.478598649389487,
    "wave_theory": 0.818283180489803
}

# 市场状态策略权重配置
MARKET_STATE_WEIGHTS = {
    "uptrend": {
        "ma_cross": 0.32048324544087786,
        "momentum_breakout": 0.8769099086178171,
        "turtle": 1.637931136008917,
        "macd": 0.748930985323352,
        "daily_breakout": 1.478598649389487,
        "rsi": 1.906009222972592,
        "bollinger": 0.9232444456662969,
        "kdj": 1.4647470623067596,
        "mean_reversion": 1.1649306516234597,
        "wave_theory": 0.818283180489803
    },
    "downtrend": {
        "ma_cross": 0.32048324544087786,
        "momentum_breakout": 0.8769099086178171,
        "turtle": 1.637931136008917,
        "macd": 0.748930985323352,
        "daily_breakout": 1.478598649389487,
        "rsi": 1.906009222972592,
        "bollinger": 0.9232444456662969,
        "kdj": 1.4647470623067596,
        "mean_reversion": 1.1649306516234597,
        "wave_theory": 0.818283180489803
    },
    "ranging": {
        "rsi": 1.906009222972592,
        "bollinger": 0.9232444456662969,
        "mean_reversion": 1.1649306516234597,
        "kdj": 1.4647470623067596,
        "wave_theory": 0.818283180489803,
        "ma_cross": 0.32048324544087786,
        "macd": 0.748930985323352,
        "turtle": 1.637931136008917,
        "momentum_breakout": 0.8769099086178171,
        "daily_breakout": 1.478598649389487
    },
    "none": DEFAULT_WEIGHTS
}

# 市场趋势置信度阈值配置
CONFIDENCE_THRESHOLDS = {
    "high_confidence": 0.6813641209473742,
    "medium_confidence": 0.6336441706563001
}
