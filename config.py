from strategies import ma_cross, rsi, bollinger, mean_reversion, momentum_breakout, macd, kdj, turtle, daily_breakout, profit_protect, resilient_trend

SYMBOL = "XAUUSD"
INTERVAL = 60  # 秒
INITIAL_CAPITAL = 10000  # 初始资金

# 策略权重配置 (策略实例, 权重)
STRATEGIES = [
    (ma_cross.Strategy(), 2.28),
    (rsi.Strategy(), 1.84),
    (bollinger.Strategy(), 1.46),
    (mean_reversion.Strategy(), 1.90),
    (momentum_breakout.Strategy(), 2.77),
    (macd.Strategy(), 1.70),
    (kdj.Strategy(), 0.65),
    (turtle.Strategy(), 0.68),
    (daily_breakout.Strategy(), 0.66),
    (profit_protect.Strategy(), 0.57),
    (resilient_trend.Strategy(), 1.0), # 新增带容错的趋势策略
]

# 交易信号阈值
BUY_THRESHOLD = 1.5
SELL_THRESHOLD = -1.5
