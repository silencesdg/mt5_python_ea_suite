import time
import signal
import sys
import json
import pandas as pd
from datetime import datetime
from logger import logger
from config import SYMBOL, TIMEFRAME, REALTIME_CONFIG
from core.risk import RiskController
from strategies.wave_theory import WaveTheoryStrategy
from strategies.ma_cross import MACrossStrategy
from strategies.rsi import RSIStrategy
from strategies.bollinger import BollingerStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.macd import MACDStrategy
from strategies.kdj import KDJStrategy
from strategies.turtle import TurtleStrategy
from strategies.daily_breakout import DailyBreakoutStrategy

class RealtimeTrader:
    """实时交易器 (已重构为依赖注入和市场状态参数切换)"""
    
    def __init__(self, data_provider, update_interval=60):
        self.data_provider = data_provider
        self.update_interval = update_interval
        self.running = False
        self.risk_controller = None
        
        self.regime_params = {}
        self.regime_detector = None
        self.current_regime = 'Ranging'  # Default regime
        self.strategy_instances = {}
        self.current_weights = {}
        self.signal_thresholds = {}

        self.strategy_config_key_to_class = {
            "ma_cross": MACrossStrategy,
            "rsi": RSIStrategy,
            "bollinger": BollingerStrategy,
            "mean_reversion": MeanReversionStrategy,
            "momentum_breakout": MomentumBreakoutStrategy,
            "macd": MACDStrategy,
            "kdj": KDJStrategy,
            "turtle": TurtleStrategy,
            "daily_breakout": DailyBreakoutStrategy,
            "wave_theory": WaveTheoryStrategy
        }

    def _initialize(self):
        if not self.data_provider.initialize():
            return False
        
        self.risk_controller = RiskController(self.data_provider)
        
        # Load regime parameters
        try:
            with open('regime_optimal_params.json', 'r') as f:
                self.regime_params = json.load(f)
            logger.info("成功加载市场状态参数文件: regime_optimal_params.json")
        except FileNotFoundError:
            logger.error("错误: 未找到 regime_optimal_params.json。请先运行 regime_optimizer.py。")
            return False
        except json.JSONDecodeError:
            logger.error("错误: regime_optimal_params.json 文件格式不正确。")
            return False

        # Instantiate regime detector
        self.regime_detector = WaveTheoryStrategy(self.data_provider, SYMBOL, TIMEFRAME)
        
        # Create strategy instances
        self._create_strategy_instances()

        self.risk_controller.sync_state()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("实时交易系统初始化完成")
        return True

    def _create_strategy_instances(self):
        for name, strategy_class in self.strategy_config_key_to_class.items():
            # Use the name that corresponds to the keys in the parameter file (e.g., 'MACrossStrategy')
            class_name = strategy_class.__name__
            self.strategy_instances[class_name] = strategy_class(self.data_provider, SYMBOL, TIMEFRAME)
        logger.info(f"创建了 {len(self.strategy_instances)} 个策略实例")

    def _update_parameters_for_regime(self):
        # 1. Determine current regime
        df_history = self.data_provider.get_historical_data(SYMBOL, TIMEFRAME, count=200)
        if df_history is None or df_history.empty:
            logger.warning("无法获取历史数据来判断市场状态，将使用上一个状态。")
            return

        df_history = pd.DataFrame(df_history)
        df_history.set_index(pd.to_datetime(df_history['time'], unit='s'), inplace=True)

        df_with_indicators = self.regime_detector._calculate_indicators(df_history.copy())
        
        last_row = df_with_indicators.iloc[-1]
        new_regime = "Ranging"  # Default
        if not pd.isna(last_row['adx']):
            if last_row['adx'] < self.regime_detector.adx_threshold:
                new_regime = "Ranging"
            elif last_row['ema_short'] > last_row['ema_medium'] > last_row['ema_long']:
                new_regime = "Uptrend"
            else:
                new_regime = "Downtrend"

        # 2. If regime changed, update parameters
        if new_regime != self.current_regime:
            self.current_regime = new_regime
            logger.info(f"市场状态已切换为: {self.current_regime}")
            
            regime_config = self.regime_params.get(self.current_regime)
            if not regime_config:
                logger.error(f"在参数文件中未找到状态 {self.current_regime} 的配置，将使用默认参数。")
                return

            params = regime_config.get('best_parameters', {})
            
            # Group parameters by component (strategy, risk, etc.)
            strategy_params = {strat_name: {} for strat_name in self.strategy_instances.keys()}
            risk_params = {}
            temp_weights = {}
            
            prefix_map = {
                'MACrossStrategy': 'ma_cross_',
                'RSIStrategy': 'rsi_',
                'BollingerStrategy': 'bollinger_',
                'MACDStrategy': 'macd_',
                'MeanReversionStrategy': 'mean_reversion_',
                'MomentumBreakoutStrategy': 'momentum_breakout_',
                'KDJStrategy': 'kdj_',
                'TurtleStrategy': 'turtle_',
                'DailyBreakoutStrategy': 'daily_breakout_',
                'WaveTheoryStrategy': 'wave_'
            }

            for key, value in params.items():
                if key.startswith('weight_'):
                    strategy_name = key.replace('weight_', '')
                    temp_weights[strategy_name] = value
                elif key in ['buy_threshold', 'sell_threshold']:
                    self.signal_thresholds[key] = value
                elif hasattr(self.risk_controller, key):
                    risk_params[key] = value
                else:
                    for strat_name, prefix in prefix_map.items():
                        if key.startswith(prefix):
                            param_name = key.replace(prefix, '')
                            # BUG FIX: Handle the 'wave_period' special case
                            if strat_name == 'WaveTheoryStrategy' and key == 'wave_period':
                                param_name = 'wave_period'
                            strategy_params[strat_name][param_name] = value
                            break

            # Update parameters in a batch
            for strat_name, params_dict in strategy_params.items():
                if params_dict:
                    self.strategy_instances[strat_name].set_params(params_dict)
            
            if risk_params:
                for key, value in risk_params.items():
                    setattr(self.risk_controller, key, value)

            self.current_weights = temp_weights
            logger.info(f"已为 {self.current_regime} 状态加载新参数和权重。")
        
    def _signal_handler(self, signum, frame):
        logger.info(f"接收到信号 {signum}，准备退出...")
        self.stop()

    def _run_cycle(self):
        try:
            self.risk_controller.sync_state()
            
            # Update parameters based on current market regime
            self._update_parameters_for_regime()

            current_price = self.data_provider.get_current_price(SYMBOL)
            if not current_price:
                logger.warning("无法获取当前价格，跳过本次循环")
                return

            signals, weights = [], []
            strategies_with_weights = []
            
            for strat_name, strat_instance in self.strategy_instances.items():
                weight = self.current_weights.get(strat_name, 0.0)
                if weight > 0: # Only calculate signal if weight is positive
                    strategies_with_weights.append((strat_instance, weight))
                    signals.append(strat_instance.generate_signal())
                    weights.append(weight)

            if not strategies_with_weights: return

            # 打印详细信号日志
            logger.info(f"--- 信号计算详情 (状态: {self.current_regime}) ---")
            for i, (strat, weight) in enumerate(strategies_with_weights):
                signal = signals[i]
                weighted_signal = signal * weight
                strat_name = strat.name
                logger.info(f"  策略: {strat_name:<25} | 信号: {signal:6.2f} | 权重: {weight:6.2f} | 加权信号: {weighted_signal:6.2f}")
            logger.info("--------------------")

            weighted_signal_sum = sum(s * w for s, w in zip(signals, weights))
            
            buy_threshold = self.signal_thresholds.get('buy_threshold', 1.5)
            sell_threshold = self.signal_thresholds.get('sell_threshold', -1.5)
            logger.info(f"加权信号: {weighted_signal_sum:.2f} (买入阈值: {buy_threshold}, 卖出阈值: {sell_threshold})")
            
            direction = None
            if weighted_signal_sum > buy_threshold:
                direction = "buy"
            elif weighted_signal_sum < sell_threshold:
                direction = "sell"
            
            if direction:
                logger.info(f"准备执行{direction}交易，信号强度: {weighted_signal_sum:.2f}")
                success = self.risk_controller.process_trading_signal(direction, current_price, weighted_signal_sum)
                if not success:
                    logger.warning(f"{direction}交易执行失败")
                else:
                    logger.info(f"{direction}交易执行成功")
            
            self.risk_controller.monitor_positions(current_price)

            # --- 状态汇总日志 ---
            logger.info("--- 财务状况更新 ---")
            open_positions = self.risk_controller.position_manager.positions
            if not open_positions:
                logger.info("  当前无持仓")
            else:
                logger.info(f"  当前持仓: {len(open_positions)} 个")
                for pos in open_positions:
                    pnl_pct = self.risk_controller.position_manager._calculate_pnl_pct(pos, current_price['last'])
                    holding_time = datetime.fromtimestamp(current_price['time']) - pos['entry_time']
                    holding_minutes = holding_time.total_seconds() / 60
                    logger.info(f"    - Ticket {pos['ticket']}: {pos['position_type']} {pos['symbol']} @ {pos['entry_price']:.2f} | 持仓时间: {holding_minutes:.1f}分钟 | 浮动盈亏: {pnl_pct:.2%}")

            trade_summary = self.risk_controller.position_manager.get_trade_summary()
            if trade_summary and trade_summary['total_trades'] > 0:
                logger.info("  已平仓交易摘要:")
                logger.info(f"    - 总交易: {trade_summary['total_trades']}, 盈利: {trade_summary['winning_trades']}, 亏损: {trade_summary['losing_trades']}, 胜率: {trade_summary['win_rate']:.2f}%")
                logger.info(f"    - 总净盈亏: ${trade_summary['total_profit_loss']:.2f}")

            logger.info(f"  总权益: ${self.risk_controller.position_manager.total_equity:.2f}")
            logger.info("----------------------")

        except Exception as e:
            import traceback
            logger.error(f"交易周期执行失败: {e}\n{traceback.format_exc()}")
            
    def start(self):
        if not self._initialize(): return
            
        logger.info("=== 启动实时交易系统 ===")
        self.running = True
        
        while self.running:
            cycle_start = time.time()
            self._run_cycle()
            cycle_time = time.time() - cycle_start
            wait_time = max(0, self.update_interval - cycle_time)
            if wait_time > 0: time.sleep(wait_time)
                    
    def stop(self):
        logger.info("=== 停止实时交易系统 ===")
        self.running = False
        try:
            if self.risk_controller:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.risk_controller.save_trade_history(f"realtime_trades_{timestamp}")
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
        finally:
            self.data_provider.shutdown()
            logger.info("实时交易系统已停止")
            sys.exit(0)