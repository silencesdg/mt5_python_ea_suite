# 交易配置
SYMBOL = "XAUUSD"
INTERVAL = 60  # 秒
INITIAL_CAPITAL = 10000  # 初始资金

# 时间配置
TIMEFRAME = 1  # M1 (1分钟图)

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
OPTIMIZER_COUNT = 5000  # 优化器数据量



# 信号阈值配置
SIGNAL_THRESHOLDS = {
    "buy_threshold": 0.5,     # 买入信号阈值
    "sell_threshold": -0.5,   # 卖出信号阈值
}

# 风险管理参数
RISK_CONFIG = {
    "stop_loss_pct": -0.001,        # 固定止损：亏损0.1%
    "profit_retracement_pct": 0.10,  # 利润回撤10%止盈
    "min_profit_for_trailing": 0.05,  # 追踪止损激活阈值：利润0.5%
    "take_profit_pct": 0.20,         # 固定止盈：盈利20%
    "max_position_size": 1,         # 最大仓位100%
    "max_daily_loss": -0.10,          # 最大日亏损10%
}

# 市场状态分析参数
MARKET_STATE_CONFIG = {
    "trend_period": 50,
    "retracement_tolerance": 0.30,
    "volume_period": 20,
    "volume_ma_period": 10,
}

# 波浪理论策略配置
WAVE_THEORY_CONFIG = {
    "daily_data_count": 30,  # 日线数据天数
    "ema_short": 5,
    "ema_medium": 13,
    "ema_long": 34,
    "wave_period": 21,
    "range_period": 20,
    "adx_period": 14,
}

# 市场趋势判断权重配置
TREND_INDICATOR_WEIGHTS = {
    "price_breakout": 0.35,      # 价格突破权重
    "volume_confirmation": 0.25, # 成交量确认权重
    "momentum oscillator": 0.20, # 动量震荡指标权重
    "moving_average": 0.20,     # 移动平均线权重
}

# 趋势判断阈值
TREND_THRESHOLDS = {
    "strong_trend": 0.7,         # 强趋势阈值
    "weak_trend": 0.4,           # 弱趋势阈值
    "volume_spike": 1.5,         # 成交量突增倍数
    "oversold": 30,              # 超卖阈值 (RSI)
    "overbought": 70,            # 超买阈值 (RSI)
}

# 动态权重配置（当市场状态判断置信度低时使用的基础权重）
DEFAULT_WEIGHTS = {
    "ma_cross": 1.31,
    "rsi": 0.41,
    "bollinger": 1.19,
    "mean_reversion": 0.75,
    "momentum_breakout": 1.5,
    "macd": 0.8,
    "kdj": 2.02,
    "turtle": 0.11,
    "daily_breakout": 0.28,
    "wave_theory": 0.8,  # 波浪理论策略基础权重
    "risk_management": 2.0,  # 风险管理策略权重
}
