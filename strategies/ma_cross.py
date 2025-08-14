import pandas as pd
from .base_strategy import BaseStrategy
from logger import logger
from config import STRATEGY_CONFIG

class MACrossStrategy(BaseStrategy):
    def __init__(self, data_provider, symbol, timeframe, short_window=None, long_window=None):
        super().__init__(data_provider, symbol, timeframe)
        # 从配置中获取参数，如果传入参数则使用传入的参数
        config = STRATEGY_CONFIG.get('ma_cross', {})
        self.short_window = short_window if short_window is not None else config.get('short_window', 5)
        self.long_window = long_window if long_window is not None else config.get('long_window', 20)

    def generate_signal(self):
        logger.debug(f"--- {self.name} 信号生成开始 ---")
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.long_window + 5) # 获取更多数据以防万一
        
        if rates is None or len(rates) < self.long_window + 1:
            logger.debug(f"数据不足或获取失败。需要: {self.long_window + 1}, 实际: {len(rates) if rates is not None else 0}")
            return 0
        
        logger.debug(f"获取到 {len(rates)} 条数据")
        df = pd.DataFrame(rates)
        
        # 计算移动平均线
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        
        # 提取最后两条数据用于判断
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        logger.debug(f"最新数据点: Close={latest['close']}, Short MA={latest['short_ma']:.2f}, Long MA={latest['long_ma']:.2f}")
        logger.debug(f"前一数据点: Close={previous['close']}, Short MA={previous['short_ma']:.2f}, Long MA={previous['long_ma']:.2f}")

        # 判断金叉
        is_cross_up = latest['short_ma'] > latest['long_ma'] and previous['short_ma'] <= previous['long_ma']
        logger.debug(f"金叉判断 (is_cross_up): {is_cross_up}")
        if is_cross_up:
            logger.info(f"{self.name}: 检测到金叉，生成买入信号")
            return 1
        
        # 判断死叉
        is_cross_down = latest['short_ma'] < latest['long_ma'] and previous['short_ma'] >= previous['long_ma']
        logger.debug(f"死叉判断 (is_cross_down): {is_cross_down}")
        if is_cross_down:
            logger.info(f"{self.name}: 检测到死叉，生成卖出信号")
            return -1
            
        logger.debug(f"--- {self.name} 信号生成结束 (无信号) ---")
        return 0

    def run_backtest(self, df):
        df = df.copy()
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        signals = pd.Series(0, index=df.index)
        signals[(df['short_ma'] > df['long_ma']) & (df['short_ma'].shift(1) <= df['long_ma'].shift(1))] = 1
        signals[(df['short_ma'] < df['long_ma']) & (df['short_ma'].shift(1) >= df['long_ma'].shift(1))] = -1
        return signals