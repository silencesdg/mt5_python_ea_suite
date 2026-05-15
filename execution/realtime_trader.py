import time
import signal
import sys
from datetime import datetime
from logger import logger
from config import SYMBOL, TIMEFRAME, REALTIME_CONFIG, SIGNAL_THRESHOLDS, RISK_CONFIG, RISK_CONFIG_CONST
from core.risk import RiskController
from execution.weights import DynamicWeightManager

class RealtimeTrader:
    """实时交易器 (已重构为依赖注入)"""
    
    def __init__(self, data_provider, update_interval=60):
        self.data_provider = data_provider
        self.update_interval = update_interval
        self.running = False
        self.risk_controller = None
        self.weight_manager = None
        self._cycle_count = 0
        
    def _initialize(self):
        if not self.data_provider.initialize():
            return False
        
        self.risk_controller = RiskController(self.data_provider)
        self.weight_manager = DynamicWeightManager(self.data_provider)
        self.risk_controller.sync_state()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # ★ 杠杆自适应：打印有效值
        try:
            acct = self.data_provider.get_account_info()
            lev = acct.leverage if hasattr(acct, 'leverage') else acct.get('leverage', 2000)
        except Exception:
            lev = 2000
        ratio = lev / 100.0

        # ★ 启动参数一览
        logger.info("=" * 50)
        logger.info(f"品种: {SYMBOL} | 周期: M{TIMEFRAME} | 间隔: {self.update_interval}s | 杠杆: {lev}x")
        logger.info(f"风控: 止损={RISK_CONFIG['stop_loss_pct']*ratio:.1%} | "
                   f"止盈={RISK_CONFIG['take_profit_pct']*ratio:.1%} | "
                   f"拖尾激活={RISK_CONFIG['min_profit_for_trailing']*ratio:.1%} | "
                   f"拖尾回撤={RISK_CONFIG['profit_retracement_pct']*ratio:.1%}")
        logger.info(f"信号: 买入阈值={SIGNAL_THRESHOLDS.get('buy_threshold',1.5)} | "
                   f"卖出阈值={SIGNAL_THRESHOLDS.get('sell_threshold',-1.5)}")
        logger.info(f"仓位: 最多多={REALTIME_CONFIG['max_long_positions']} 最多空={REALTIME_CONFIG['max_short_positions']} | "
                   f"超时平仓={'开' if RISK_CONFIG_CONST.get('enable_time_based_exit',True) else '关'}")
        logger.info(f"对冲: 信号对冲={'开' if REALTIME_CONFIG.get('hedge_enabled',False) else '关'} | "
                   f"锁仓={'开' if REALTIME_CONFIG.get('lock_enabled',False) else '关'}")
        logger.info("=" * 50)
        return True
        
    def _signal_handler(self, signum, frame):
        logger.info(f"接收信号 {signum}，准备退出...")
        self.stop()

    def _run_cycle(self):
        try:
            self._cycle_count += 1
            self.risk_controller.sync_state()

            current_price = self.data_provider.get_current_price(SYMBOL)
            if not current_price:
                return

            strategies_with_weights = self.weight_manager.get_current_strategies_and_weights()
            if not strategies_with_weights: return

            signals, weights = [], []
            for strat, weight in strategies_with_weights:
                signals.append(strat.generate_signal())
                weights.append(weight)

            weighted_signal_sum = sum(s * w for s, w in zip(signals, weights))
            buy_threshold = SIGNAL_THRESHOLDS.get('buy_threshold', 1.5)
            sell_threshold = SIGNAL_THRESHOLDS.get('sell_threshold', -1.5)

            direction = None
            if weighted_signal_sum > buy_threshold:
                direction = "buy"
            elif weighted_signal_sum < sell_threshold:
                direction = "sell"

            # 只在信号触发时打印决策依据
            if direction:
                logger.info(f"⚡ 信号触发 | 加权={weighted_signal_sum:.2f} | "
                           f"阈值=[{sell_threshold:.2f}, {buy_threshold:.2f}] | "
                           f"方向={direction.upper()} | 价格={current_price['last']:.2f}")
                self.risk_controller.process_trading_signal(direction, current_price, weighted_signal_sum)

            self.risk_controller.monitor_positions(current_price, weighted_signal=weighted_signal_sum)

            # 每30个周期打印一次状态摘要
            if self._cycle_count % 30 == 0:
                pm = self.risk_controller.position_manager
                n = len(pm.positions)
                summary = pm.get_trade_summary()
                logger.info(f"📊 周期#{self._cycle_count} | 持仓={n} | "
                           f"净值=${pm.total_equity:.2f} | "
                           f"已平{summary['total_trades']}笔 胜率{summary['win_rate']:.0f}% 净${summary['total_profit_loss']:+.2f}")

        except Exception as e:
            import traceback
            logger.error(f"交易周期失败: {e}\n{traceback.format_exc()}")
            
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
                summary = self.risk_controller.position_manager.get_trade_summary()
                if summary['total_trades'] > 0:
                    logger.info(f"📊 本次运行: {summary['total_trades']}笔 胜率{summary['win_rate']:.0f}% 净${summary['total_profit_loss']:+.2f}")
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
        finally:
            self.data_provider.shutdown()
            logger.info("实时交易系统已停止")
            sys.exit(0)
