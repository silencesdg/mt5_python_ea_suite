import time
import signal
import sys
from datetime import datetime
from logger import logger
import config
from core.risk import RiskController
from execution.weights import DynamicWeightManager

class RealtimeTrader:
    """实时交易器 (已重构为依赖注入) — ★ 每周期热加载配置"""
    
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

        # ★ 打印风控参数（保证金%基准）
        try:
            acct = self.data_provider.get_account_info()
            lev = acct.leverage if hasattr(acct, 'leverage') else acct.get('leverage', 2000)
        except Exception:
            lev = 2000
        rl = config.RISK_CONFIG.get('risk_leverage', 100)

        # ★ 启动参数一览
        logger.info("=" * 50)
        logger.info(f"品种: {config.SYMBOL} | 周期: M{config.TIMEFRAME} | 间隔: {self.update_interval}s | 杠杆: {lev}x")
        logger.info(f"风控: 止损={config.RISK_CONFIG['stop_loss_pct']:.0%} | "
                   f"止盈={config.RISK_CONFIG['take_profit_pct']:.0%} | "
                   f"拖尾激活={config.RISK_CONFIG['min_profit_for_trailing']:.0%} | "
                   f"拖尾回撤={config.RISK_CONFIG['profit_retracement_pct']:.0%}"
                   f"  (基准={rl}x)")
        logger.info(f"信号: 买入阈值={config.SIGNAL_THRESHOLDS.get('buy_threshold',1.5)} | "
                   f"卖出阈值={config.SIGNAL_THRESHOLDS.get('sell_threshold',-1.5)}")
        logger.info(f"仓位: 最多多={config.REALTIME_CONFIG['max_long_positions']} 最多空={config.REALTIME_CONFIG['max_short_positions']} | "
                   f"超时平仓={'开' if config.RISK_CONFIG_CONST.get('enable_time_based_exit',True) else '关'}")
        logger.info(f"对冲: 信号对冲={'开' if config.REALTIME_CONFIG.get('hedge_enabled',False) else '关'} | "
                   f"锁仓={'开' if config.REALTIME_CONFIG.get('lock_enabled',False) else '关'}")
        logger.info("=" * 50)
        return True
        
    def _signal_handler(self, signum, frame):
        logger.info(f"接收信号 {signum}，准备退出...")
        self.stop()

    def _run_cycle(self):
        try:
            self._cycle_count += 1
            
            # ★ 热加载配置：cron 优化器改完 config.py 后自动生效
            config.reload()
            
            self.risk_controller.sync_state()

            current_price = self.data_provider.get_current_price(config.SYMBOL)
            if not current_price:
                return

            strategies_with_weights = self.weight_manager.get_current_strategies_and_weights()
            if not strategies_with_weights: return

            signals, weights = [], []
            for strat, weight in strategies_with_weights:
                signals.append(strat.generate_signal())
                weights.append(weight)

            weighted_signal_sum = sum(s * w for s, w in zip(signals, weights))
            buy_threshold = config.SIGNAL_THRESHOLDS.get('buy_threshold', 1.5)
            sell_threshold = config.SIGNAL_THRESHOLDS.get('sell_threshold', -1.5)

            direction = None
            if weighted_signal_sum > buy_threshold:
                direction = "buy"
            elif weighted_signal_sum < sell_threshold:
                direction = "sell"

            # ★ 同向门槛递增：已有N单同向时，第N+1单需要更强信号
            if direction:
                # ★ 适应度门槛：回测亏钱就不开新单
                last_fitness = getattr(config, 'LAST_OPTIMIZATION_FITNESS', 0)
                min_fitness = config.RISK_CONFIG.get('min_backtest_fitness', 250)
                if last_fitness < min_fitness:
                    if self._cycle_count % 30 == 0:
                        logger.warning(f"⚠️ 回测适应度{last_fitness:.0f}<{min_fitness}，暂停开仓（监控持仓中）")
                    direction = None

            if direction:
                alpha = config.RISK_CONFIG.get('entry_escalation_alpha', 1.2)
                pm = self.risk_controller.position_manager
                n_long = sum(1 for p in pm.positions if p['position_type'] == 'long')
                n_short = sum(1 for p in pm.positions if p['position_type'] == 'short')

                if direction == 'buy':
                    adjusted = buy_threshold * (alpha ** n_long)
                    if weighted_signal_sum <= adjusted:
                        logger.debug(f"BUY信号{weighted_signal_sum:.2f}<调整阈值{adjusted:.2f}(已有{n_long}多单), 忽略")
                        direction = None
                elif direction == 'sell':
                    adjusted = sell_threshold * (alpha ** n_short)
                    if weighted_signal_sum >= adjusted:
                        logger.debug(f"SELL信号{weighted_signal_sum:.2f}>调整阈值{adjusted:.2f}(已有{n_short}空单), 忽略")
                        direction = None

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
                self.risk_controller.position_manager.cleanup_peak_data()
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
