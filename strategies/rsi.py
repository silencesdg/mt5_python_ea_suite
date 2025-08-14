import pandas as pd
from .base_strategy import BaseStrategy
from config import STRATEGY_CONFIG

class RSIStrategy(BaseStrategy):
    def __init__(self, data_provider, symbol, timeframe, period=None, overbought=None, oversold=None):
        super().__init__(data_provider, symbol, timeframe)
        # 从配置中获取参数，如果传入参数则使用传入的参数
        config = STRATEGY_CONFIG.get('rsi', {})
        self.period = period if period is not None else config.get('period', 14)
        self.overbought = overbought if overbought is not None else config.get('overbought', 70)
        self.oversold = oversold if oversold is not None else config.get('oversold', 30)

    def generate_signal(self):
        rates = self.data_provider.get_historical_data(self.symbol, self.timeframe, self.period + 10) # Get more data for stability
        if rates is None or len(rates) < self.period + 1:
            return 0

        df = pd.DataFrame(rates)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        latest_rsi = rsi.iloc[-1]

        if latest_rsi > self.overbought:
            return -1
        
        if latest_rsi < self.oversold:
            return 1
            
        return 0

    def run_backtest(self, df):
        """
        为RSI策略生成回测信号的向量化方法。
        """
        df = df.copy()
        # 计算价格变化
        delta = df['close'].diff()
        
        # 分别计算上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 使用指数移动平均（EMA）计算平均增益和损失，这是RSI的标准算法
        avg_gain = gain.ewm(com=self.period - 1, min_periods=self.period).mean()
        avg_loss = loss.ewm(com=self.period - 1, min_periods=self.period).mean()

        # 计算RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # 根据超买超卖阈值生成信号
        signals = pd.Series(0, index=df.index)
        signals[rsi > self.overbought] = -1  # 超买区域，产生卖出信号
        signals[rsi < self.oversold] = 1    # 超卖区域，产生买入信号
        
        return signals