# MT5 代理服务配置
import importlib as _importlib, sys as _sys

def reload():
    """★ 热重载配置 — 运行时调用，无需重启 EA"""
    _importlib.reload(_sys.modules[__name__])

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
OPTIMIZER_COUNT = 6000  # 优化器数据量（约4天M1，更快响应行情）

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
    "max_long_positions": 10,        # 同方向最多10单
    "max_short_positions": 10,       # 同方向最多10单
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
    "population_size": 20,         # 种群大小（精简化，6000bar快速收敛）
    "generations": 6,              # 进化代数
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
    "buy_threshold": 2.087211936631902,
    "sell_threshold": -0.6594468309213586
}


# 风险管理参数（手动设定，优化器不宜优化 — 保证金% 基础值，杠杆自适应缩放）
RISK_CONFIG = {
    # ★ 以下百分比均为保证金%（100x基准），实盘按 (MT5杠杆/100) 缩放
    #    2000x下：-0.70 → -14×杠杆 → -1400%保证金 = -$32 ≈ 32点
    "risk_leverage": 100,              # 100x基准计算保证金%（不改实际杠杆）
    "stop_loss_pct": -0.10,                           # -10% 保证金 ←手动设定，不进优化器
    "profit_retracement_pct": 0.03,              # ★ 3% 回撤容忍（相对！3%×峰值利润） ←手动设定
    "retracement_mode": "relative",              # ★ 回撤模式从绝对值改为相对（利润的3%）
    "min_profit_for_trailing": 0.10,                  # +10% 后激活拖尾 ←手动设定
    "take_profit_pct": 0.15,                          # +15% 止盈 ←手动设定
    "max_daily_loss": -0.3,
    "max_holding_minutes": 0,                         # 0=不启用时间平仓
    "min_profit_for_time_exit": 0.05,
    "cooldown_bars": 30,
    # ★ 同向加仓门槛递增：第N单需要的信号 = 基础阈值 × α^(N-1)
    "entry_escalation_alpha": 1.2,    # 递增系数（1.2=每单信号强20%）
    # ★ 波动率自适应拖尾：激活点数 = 基础点数 × (当前ATR / 长周期ATR)
    "vol_adaptive_trailing": True,     # 启动波动率自适应拖尾
    "trailing_atr_period": 14,         # ATR短周期
    "trailing_atr_baseline": 100,      # ATR长周期（基准）
    # ★ 硬止损倍率：MT5 服务器端 SL/TP = 软止损 × 倍率（兜底，仅 EA 挂掉时触发）
    "hard_sl_multiplier": 1.5,   # 硬 SL = 软 SL × 1.5
    "hard_tp_multiplier": 1.3,   # 硬 TP = 软 TP × 1.3
    # ★ 回测适应度门槛：低于此值不开新单（适应度≈回测总盈亏$）
    "min_backtest_fitness": 95,      # 适应度<95 → 不开仓（低于95说明市场难做）
}

# ★ 最近优化适应度（daily_optimize.py 写入，EA 热加载读取）
LAST_OPTIMIZATION_FITNESS = 58.63

# 市场状态分析参数
MARKET_STATE_CONFIG = {
    "trend_period": 60,
    "retracement_tolerance": 0.3857733104065197,
    "volume_period": 42,
    "volume_ma_period": 30
}

# 策略参数配置（优化器结果 2026-05-10，5万根M1数据）
STRATEGY_CONFIG = {
    "ma_cross": {
        "short_window": 7,
        "long_window": 18
    },
    "rsi": {
        "period": 19,
        "overbought": 75,
        "oversold": 31
    },
    "bollinger": {
        "period": 20,
        "std_dev": 2.2757986281351896
    },
    "macd": {
        "fast_ema": 19,
        "slow_ema": 23,
        "signal_period": 13
    },
    "mean_reversion": {
        "period": 29,
        "std_dev": 2.149
    },
    "momentum_breakout": {
        "period": 15,
        "momentum_period": 19
    },
    "kdj": {
        "period": 21
    },
    "swing_point": {
        "left_bars": 4,
        "right_bars": 3,
        "tolerance_pct": 0.0006056955205083707,
        "num_swings": 2
    },
    "daily_breakout": {
        "bars_count": 1392
    },
    "wave_theory": {
        "ema_short": 8,
        "ema_medium": 11,
        "ema_long": 39,
        "wave_period": 19,
        "range_period": 15,
        "adx_period": 24,
        "momentum_period": 10,
        "range_threshold": 0.0023249568571626195,
        "adx_threshold": 28
    }
}

# 市场趋势判断权重配置
TREND_INDICATOR_WEIGHTS = {
    "price_breakout": 0.2506453627492913,
    "volume_confirmation": 0.2851951176719032,
    "momentum oscillator": 0.3677,
    "moving_average": 0.353451489159348
}

# 趋势判断阈值
TREND_THRESHOLDS = {
    "strong_trend": 0.4040165929726079,
    "weak_trend": 0.2963326526031991,
    "volume_spike": 1.5234871986051206,
    "oversold": 31,
    "overbought": 75
}


# 动态权重配置（优化器结果 2026-05-10，5万根M1数据）
DEFAULT_WEIGHTS = {
    "ma_cross": 0.4771175228437661,
    "rsi": 0.478479356771334,
    "bollinger": 1.3391582231825303,
    "mean_reversion": 1.9958459833789346,
    "momentum_breakout": 0.7567594572894596,
    "macd": 1.170771336653618,
    "kdj": 0.013296663814979182,
    "swing_point": 0.5194535227966377,
    "daily_breakout": 0.6506588091889913,
    "wave_theory": 1.6765424080384428
}

# 市场状态策略权重配置
MARKET_STATE_WEIGHTS = {
    "uptrend": {
        "ma_cross": 0.4771175228437661,
        "momentum_breakout": 0.7567594572894596,
        "swing_point": 0.5194535227966377,
        "macd": 1.170771336653618,
        "daily_breakout": 0.6506588091889913,
        "rsi": 0.478479356771334,
        "bollinger": 1.3391582231825303,
        "kdj": 0.013296663814979182,
        "mean_reversion": 1.9958459833789346,
        "wave_theory": 1.6765424080384428
    },
    "downtrend": {
        "ma_cross": 0.4771175228437661,
        "momentum_breakout": 0.7567594572894596,
        "swing_point": 0.5194535227966377,
        "macd": 1.170771336653618,
        "daily_breakout": 0.6506588091889913,
        "rsi": 0.478479356771334,
        "bollinger": 1.3391582231825303,
        "kdj": 0.013296663814979182,
        "mean_reversion": 1.9958459833789346,
        "wave_theory": 1.6765424080384428
    },
    "ranging": {
        "rsi": 0.478479356771334,
        "bollinger": 1.3391582231825303,
        "mean_reversion": 1.9958459833789346,
        "kdj": 0.013296663814979182,
        "wave_theory": 1.6765424080384428,
        "ma_cross": 0.4771175228437661,
        "macd": 1.170771336653618,
        "swing_point": 0.5194535227966377,
        "momentum_breakout": 0.7567594572894596,
        "daily_breakout": 0.6506588091889913
    },
    "none": DEFAULT_WEIGHTS
}

# 市场趋势置信度阈值配置
CONFIDENCE_THRESHOLDS = {
    "high_confidence": 0.5050416445534804,
    "medium_confidence": 0.6102467437388168
}
