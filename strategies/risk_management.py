import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from utils import get_rates, has_open_position, get_current_price
from logger import logger
from config import RISK_CONFIG, SYMBOL

class Strategy:
    """
    风险管理策略
    基于市场状态和持仓情况生成风险管理信号
    """
    
    def __init__(self):
        self.symbol = SYMBOL
        self.timeframe = mt5.TIMEFRAME_M1
        self.data_count = 100  # 用于分析的数据量
        
        # 从配置文件读取风险管理参数
        self.stop_loss_pct = RISK_CONFIG.get("stop_loss_pct", -0.001)
        self.profit_retracement_pct = RISK_CONFIG.get("profit_retracement_pct", 0.10)
        self.min_profit_for_trailing = RISK_CONFIG.get("min_profit_for_trailing", 0.05)
        self.take_profit_pct = RISK_CONFIG.get("take_profit_pct", 0.20)
        self.max_daily_loss = RISK_CONFIG.get("max_daily_loss", -0.10)
        
        # 策略参数
        self.volatility_period = 20
        self.trend_period = 50
        self.rsi_period = 14
        
        # 内部状态
        self.positions = {}
        self.daily_pnl = 0.0
        self.last_position_time = None
        self.min_trade_interval = 5  # 最小交易间隔（分钟）
        
    def _calculate_indicators(self, df):
        """
        计算技术指标
        """
        # 计算移动平均线
        df['ma_short'] = df['close'].rolling(10).mean()
        df['ma_long'] = df['close'].rolling(30).mean()
        
        # 计算RSI
        df['rsi'] = self._calculate_rsi(df['close'], self.rsi_period)
        
        # 计算波动率
        df['volatility'] = df['close'].rolling(self.volatility_period).std() / df['close'].rolling(self.volatility_period).mean()
        
        # 计算ATR
        df['atr'] = self._calculate_atr(df, 14)
        
        # 计算价格变化
        df['price_change'] = df['close'].pct_change()
        
        return df
    
    def _calculate_rsi(self, prices, period):
        """
        计算RSI指标
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df, period):
        """
        计算ATR指标
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _get_position_info(self):
        """
        获取当前持仓信息
        """
        return self.positions.get(self.symbol, None)
    
    def _calculate_position_pnl(self, current_price, position_info):
        """
        计算持仓盈亏
        """
        if not position_info:
            return 0.0
            
        entry_price = position_info['entry_price']
        position_type = position_info['position_type']
        
        if position_type == 'long':
            return (current_price - entry_price) / entry_price
        else:
            return (entry_price - current_price) / entry_price
    
    def _check_risk_conditions(self, df):
        """
        检查风险管理条件
        """
        current_price = df['close'].iloc[-1]
        current_volatility = df['volatility'].iloc[-1]
        current_rsi = df['rsi'].iloc[-1]
        current_atr = df['atr'].iloc[-1]
        
        # 检查是否有持仓
        position_info = self._get_position_info()
        
        # 如果没有持仓，检查开仓条件
        if not position_info:
            return self._check_entry_conditions(df, current_price, current_volatility, current_rsi, current_atr)
        
        # 如果有持仓，检查平仓条件
        else:
            return self._check_exit_conditions(df, current_price, position_info)
    
    def _check_entry_conditions(self, df, current_price, volatility, rsi, atr):
        """
        检查开仓条件
        """
        # 检查日亏损限制
        if self.daily_pnl <= self.max_daily_loss:
            logger.info(f"日亏损已达{self.daily_pnl:.2%}，禁止开仓")
            return 0
        
        # 检查最小交易间隔
        if self.last_position_time:
            time_diff = (pd.Timestamp.now() - self.last_position_time).total_seconds() / 60
            if time_diff < self.min_trade_interval:
                return 0
        
        # 基于市场状态的开仓条件
        ma_short = df['ma_short'].iloc[-1]
        ma_long = df['ma_long'].iloc[-1]
        
        # 低波动率且趋势明确时开仓
        if volatility < 0.02:  # 低波动率
            if ma_short > ma_long and rsi < 70:  # 上升趋势且未超买
                logger.info(f"风险管理策略：低波动率上升趋势，买入信号")
                return 1
            elif ma_short < ma_long and rsi > 30:  # 下降趋势且未超卖
                logger.info(f"风险管理策略：低波动率下降趋势，卖出信号")
                return -1
        
        # 高波动率时等待机会
        elif volatility > 0.05:  # 高波动率
            if rsi < 30:  # 超卖反弹机会
                logger.info(f"风险管理策略：高波动率超卖反弹，买入信号")
                return 1
            elif rsi > 70:  # 超买回调机会
                logger.info(f"风险管理策略：高波动率超买回调，卖出信号")
                return -1
        
        return 0
    
    def _check_exit_conditions(self, df, current_price, position_info):
        """
        检查平仓条件
        """
        position_type = position_info['position_type']
        entry_price = position_info['entry_price']
        current_pnl = self._calculate_position_pnl(current_price, position_info)
        
        # 更新最高利润
        self.positions[self.symbol]['peak_profit'] = max(
            position_info.get('peak_profit', 0), 
            current_pnl
        )
        peak_profit = self.positions[self.symbol]['peak_profit']
        
        # 检查止损 - 仅基于买入成本判断亏损，不考虑盈利回撤
        if current_pnl < 0 and current_pnl <= self.stop_loss_pct:
            logger.info(f"风险管理策略：止损触发，当前亏损{current_pnl:.2%}")
            return -1 if position_type == 'long' else 1
        
        # 检查止盈
        if current_pnl >= self.take_profit_pct:
            logger.info(f"风险管理策略：止盈触发，当前盈利{current_pnl:.2%}")
            return -1 if position_type == 'long' else 1
        
        # 检查追踪止损
        if peak_profit > self.min_profit_for_trailing:
            retracement_from_peak = peak_profit - current_pnl
            if peak_profit > 0:
                retracement_pct = retracement_from_peak / peak_profit
                if retracement_pct >= self.profit_retracement_pct:
                    logger.info(f"风险管理策略：追踪止损触发，最高盈利{peak_profit:.2%}，回撤{retracement_pct:.2%}")
                    return -1 if position_type == 'long' else 1
        
        return 0
    
    def generate_signal(self):
        """
        生成实时交易信号
        """
        # 获取当前价格
        current_price = get_current_price(self.symbol)
        if current_price is None:
            return 0
        
        # 获取历史数据
        rates = get_rates(self.symbol, self.timeframe, self.data_count)
        if rates is None or len(rates) < self.trend_period:
            return 0
        
        df = pd.DataFrame(rates)
        df = self._calculate_indicators(df)
        
        # 检查风险管理条件
        signal = self._check_risk_conditions(df)
        
        return signal
    
    def generate_signal_with_sync(self, risk_controller):
        """
        生成实时交易信号（与主风险管理器同步）
        """
        # 先同步状态
        self.sync_with_risk_controller(risk_controller)
        
        # 然后生成信号
        return self.generate_signal()
    
    def run_backtest(self, df):
        """
        回测信号生成
        """
        df = df.copy()
        df = self._calculate_indicators(df)
        
        signals = pd.Series(0, index=df.index)
        
        # 模拟持仓状态
        positions = {}
        daily_pnl = 0.0
        
        for i in range(self.trend_period, len(df)):
            current_price = df['close'].iloc[i]
            current_time = df.index[i]
            
            # 获取当前持仓信息
            position_info = positions.get(self.symbol)
            
            if not position_info:
                # 检查开仓条件
                if self._should_open_position(df.iloc[:i+1], current_price):
                    # 简化的开仓逻辑
                    ma_short = df['ma_short'].iloc[i]
                    ma_long = df['ma_long'].iloc[i]
                    rsi = df['rsi'].iloc[i]
                    
                    if ma_short > ma_long and rsi < 70:
                        signals.iat[i] = 1
                        positions[self.symbol] = {
                            'entry_price': current_price,
                            'position_type': 'long',
                            'peak_profit': 0.0
                        }
                    elif ma_short < ma_long and rsi > 30:
                        signals.iat[i] = -1
                        positions[self.symbol] = {
                            'entry_price': current_price,
                            'position_type': 'short',
                            'peak_profit': 0.0
                        }
            else:
                # 检查平仓条件
                position_type = position_info['position_type']
                entry_price = position_info['entry_price']
                current_pnl = self._calculate_position_pnl(current_price, position_info)
                
                # 更新最高利润
                position_info['peak_profit'] = max(position_info['peak_profit'], current_pnl)
                
                # 检查止损止盈 - 止损仅基于买入成本判断亏损
                if (current_pnl < 0 and current_pnl <= self.stop_loss_pct) or current_pnl >= self.take_profit_pct:
                    signals.iat[i] = -1 if position_type == 'long' else 1
                    positions.pop(self.symbol, None)
                    
                    # 更新日内盈亏
                    daily_pnl += current_pnl
                    if daily_pnl <= self.max_daily_loss:
                        break  # 日亏损达到限制，停止交易
        
        return signals
    
    def _should_open_position(self, df, current_price):
        """
        判断是否应该开仓
        """
        volatility = df['volatility'].iloc[-1]
        
        # 低波动率且未达到日亏损限制
        return volatility < 0.02
    
    def update_position_entry(self, entry_price, position_type="long"):
        """
        更新持仓入场信息
        """
        self.positions[self.symbol] = {
            'entry_price': entry_price,
            'position_type': position_type,
            'peak_profit': 0.0,
            'entry_time': pd.Timestamp.now()
        }
        self.last_position_time = pd.Timestamp.now()
        
        logger.info(f"风险管理策略记录持仓入场：价格={entry_price:.2f}, 类型={position_type}")
    
    def close_position(self, current_price):
        """
        平仓
        """
        position_info = self.positions.get(self.symbol)
        if position_info:
            pnl = self._calculate_position_pnl(current_price, position_info)
            self.daily_pnl += pnl
            
            self.positions.pop(self.symbol, None)
            logger.info(f"风险管理策略平仓：盈亏{pnl:.2%}")
    
    def reset_daily_stats(self):
        """
        重置日内统计
        """
        self.daily_pnl = 0.0
        logger.info("风险管理策略重置日内统计")
    
    def sync_with_risk_controller(self, risk_controller):
        """
        与主风险管理器同步状态
        """
        # 从主风险管理器获取持仓信息
        if hasattr(risk_controller, 'position_manager'):
            position_info = risk_controller.position_manager.get_position_info()
            if position_info:
                self.positions[self.symbol] = position_info
        
        # 同步日内盈亏
        if hasattr(risk_controller, 'daily_pnl'):
            self.daily_pnl = risk_controller.daily_pnl