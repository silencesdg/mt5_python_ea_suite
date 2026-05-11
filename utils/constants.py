# MT5 时间周期常量映射
# 参考: https://www.mql5.com/en/docs/constants/chartconstants/enum_timeframes

PERIOD_M1 = 1
PERIOD_M5 = 5
PERIOD_M15 = 15
PERIOD_M30 = 30
PERIOD_H1 = 16385
PERIOD_H4 = 16386
PERIOD_D1 = 16408
PERIOD_W1 = 32769
PERIOD_MN1 = 49153

# MT5 timeframe → 分钟数
TIMEFRAME_TO_MINUTES = {
    PERIOD_M1: 1,
    PERIOD_M5: 5,
    PERIOD_M15: 15,
    PERIOD_M30: 30,
    PERIOD_H1: 60,
    PERIOD_H4: 240,
    PERIOD_D1: 1440,
    PERIOD_W1: 10080,
    PERIOD_MN1: 43200,
}

# MT5 timeframe → pandas resample rule
RESAMPLE_RULES = {
    PERIOD_M1: '1min',
    PERIOD_M5: '5min',
    PERIOD_M15: '15min',
    PERIOD_M30: '30min',
    PERIOD_H1: '1h',
    PERIOD_H4: '4h',
    PERIOD_D1: '1D',
    PERIOD_W1: '1W',
    PERIOD_MN1: '1ME',
}

# MT5 交易方向常量
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1
