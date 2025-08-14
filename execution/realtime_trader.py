import time
import signal
import sys
from datetime import datetime
from logger import logger
from config import SYMBOL, TIMEFRAME, REALTIME_CONFIG, SIGNAL_THRESHOLDS
from core.risk import RiskController
from execution.dynamic_weights import DynamicWeightManager

class RealtimeTrader:
    """实时交易器 (已重构为依赖注入)"""
    
    def __init__(self, data_provider, update_interval=60):
        self.data_provider = data_provider
        self.update_interval = update_interval
        self.running = False
        self.risk_controller = None
        self.weight_manager = None
        
    def _initialize(self):
        if not self.data_provider.initialize():
            return False
        
        self.risk_controller = RiskController(self.data_provider)
        self.weight_manager = DynamicWeightManager(self.data_provider)
        
        self.risk_controller.sync_state()

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.info("实时交易系统初始化完成")
        return True
        
    def _signal_handler(self, signum, frame):
        logger.info(f"接收到信号 {signum}，准备退出...")
        self.stop()

    def _run_cycle(self):
        try:
            self.risk_controller.sync_state()

            current_price = self.data_provider.get_current_price(SYMBOL)
            if not current_price:
                logger.warning("无法获取当前价格，跳过本次循环")
                return

            strategies_with_weights = self.weight_manager.get_current_strategies_and_weights()
            if not strategies_with_weights: return

            signals, weights = [], []
            for strat, weight in strategies_with_weights:
                signals.append(strat.generate_signal())
                weights.append(weight)
            
            # 打印详细信号日志
            logger.info("--- 信号计算详情 ---")
            for i, (strat, weight) in enumerate(strategies_with_weights):
                signal = signals[i]
                weighted_signal = signal * weight
                strat_name = strat.name
                logger.info(f"  策略: {strat_name:<25} | 信号: {signal:6.2f} | 权重: {weight:6.2f} | 加权信号: {weighted_signal:6.2f}")
            logger.info("--------------------")

            weighted_signal_sum = sum(s * w for s, w in zip(signals, weights))
            
            buy_threshold = SIGNAL_THRESHOLDS.get('buy_threshold', 1.5)
            sell_threshold = SIGNAL_THRESHOLDS.get('sell_threshold', -1.5)
            logger.info(f"加权信号: {weighted_signal_sum:.2f} (买入阈值: {buy_threshold}, 卖出阈值: {sell_threshold})")
            # logger.info(f"信号比较: {weighted_signal_sum} > {buy_threshold} = {weighted_signal_sum > buy_threshold}")
            # logger.info(f"信号比较: {weighted_signal_sum} < {sell_threshold} = {weighted_signal_sum < sell_threshold}")
            
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
                    # 计算持仓时间
                    holding_time = current_price['time'] - pos['entry_time']
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

