#!/usr/bin/env python3
"""每日自动优化 — 遗传算法跑参数 → 写入config → 重启实盘EA

通过 cron 调用: python3 scripts/daily_optimize.py
"""

import sys, os, json, re, time, signal, shutil
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from logger import setup_logger
setup_logger("INFO")
from logger import logger

CONFIG_PATH = PROJECT_DIR / "config.py"
BACKUP_DIR = PROJECT_DIR / "config_backups"
RESTART_SIGNAL = PROJECT_DIR / ".restart_signal"


def run_optimizer():
    """运行遗传算法优化，返回 best_params dict"""
    from execution.optimize import run_optimizer as _run
    logger.info("🧬 开始遗传算法优化...")
    best_params, fitness = _run()
    logger.info(f"✅ 优化完成 适应度={fitness:.2f}")
    return best_params


def backup_config():
    """备份当前 config.py"""
    BACKUP_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"config_{ts}.py"
    shutil.copy(CONFIG_PATH, dst)
    logger.info(f"📦 已备份配置: {dst}")


def update_config(best_params: dict):
    """将优化结果写回 config.py"""
    content = CONFIG_PATH.read_text(encoding="utf-8")

    # ═══ RISK_CONFIG ═══
    risk_map = {
        "stop_loss_pct": "stop_loss_pct",
        "profit_retracement_pct": "profit_retracement_pct",
        "min_profit_for_trailing": "min_profit_for_trailing",
        "take_profit_pct": "take_profit_pct",
        "max_holding_minutes": "max_holding_minutes",
        "min_profit_for_time_exit": "min_profit_for_time_exit",
    }
    for opt_key, cfg_key in risk_map.items():
        if opt_key in best_params:
            val = best_params[opt_key]
            content = re.sub(
                rf'("{cfg_key}":\s*)[\d.\-e]+',
                rf'\g<1>{val}',
                content
            )

    # ═══ SIGNAL_THRESHOLDS ═══
    if "buy_threshold" in best_params:
        content = re.sub(
            r'("buy_threshold":\s*)[\d.\-e]+',
            rf'\g<1>{best_params["buy_threshold"]}',
            content
        )
    if "sell_threshold" in best_params:
        content = re.sub(
            r'("sell_threshold":\s*)[\d.\-e]+',
            rf'\g<1>{best_params["sell_threshold"]}',
            content
        )

    # ═══ MARKET_STATE_CONFIG ═══
    market_map = {
        "market_trend_period": "trend_period",
        "market_retracement_tolerance": "retracement_tolerance",
        "market_volume_period": "volume_period",
        "market_volume_ma_period": "volume_ma_period",
    }
    for opt_key, cfg_key in market_map.items():
        if opt_key in best_params:
            val = int(best_params[opt_key]) if "period" in opt_key else best_params[opt_key]
            content = re.sub(
                rf'("{cfg_key}":\s*)[\d.\-e]+',
                rf'\g<1>{val}',
                content
            )

    # ═══ STRATEGY_CONFIG (strategy params) ═══
    strategy_param_map = {
        # MACrossStrategy
        "ma_cross_short_window": ("ma_cross", "short_window"),
        "ma_cross_long_window": ("ma_cross", "long_window"),
        # RSIStrategy
        "rsi_period": ("rsi", "period"),
        "rsi_overbought": ("rsi", "overbought"),
        "rsi_oversold": ("rsi", "oversold"),
        # BollingerStrategy
        "bollinger_period": ("bollinger", "period"),
        "bollinger_std_dev": ("bollinger", "std_dev"),
        # MACDStrategy
        "macd_fast_ema": ("macd", "fast_ema"),
        "macd_slow_ema": ("macd", "slow_ema"),
        "macd_signal_period": ("macd", "signal_period"),
        # MeanReversionStrategy
        "mean_reversion_period": ("mean_reversion", "period"),
        "mean_reversion_std_dev": ("mean_reversion", "std_dev"),
        # MomentumBreakoutStrategy
        "momentum_breakout_period": ("momentum_breakout", "period"),
        "momentum_breakout_momentum_period": ("momentum_breakout", "momentum_period"),
        # KDJStrategy
        "kdj_period": ("kdj", "period"),
        # TurtleStrategy
        "turtle_period": ("turtle", "period"),
        # DailyBreakoutStrategy
        "daily_breakout_bars_count": ("daily_breakout", "bars_count"),
        # WaveTheoryStrategy
        "wave_ema_short": ("wave_theory", "ema_short"),
        "wave_ema_medium": ("wave_theory", "ema_medium"),
        "wave_ema_long": ("wave_theory", "ema_long"),
        "wave_period": ("wave_theory", "wave_period"),
        "wave_range_period": ("wave_theory", "range_period"),
        "wave_adx_period": ("wave_theory", "adx_period"),
        "wave_momentum_period": ("wave_theory", "momentum_period"),
        "wave_range_threshold": ("wave_theory", "range_threshold"),
        "wave_adx_threshold": ("wave_theory", "adx_threshold"),
    }
    for opt_key, (section, key) in strategy_param_map.items():
        if opt_key in best_params:
            val = best_params[opt_key]
            if isinstance(val, float) and abs(val - round(val)) < 1e-6:
                val = int(round(val))
            content = re.sub(
                rf'("{key}":\s*)[\d.\-e]+',
                rf'\g<1>{val}',
                content,
                count=1,
            )

    # ═══ DEFAULT_WEIGHTS ═══
    weight_map = {
        "weight_MACrossStrategy": "ma_cross",
        "weight_RSIStrategy": "rsi",
        "weight_BollingerStrategy": "bollinger",
        "weight_MeanReversionStrategy": "mean_reversion",
        "weight_MomentumBreakoutStrategy": "momentum_breakout",
        "weight_MACDStrategy": "macd",
        "weight_KDJStrategy": "kdj",
        "weight_TurtleStrategy": "turtle",
        "weight_DailyBreakoutStrategy": "daily_breakout",
        "weight_WaveTheoryStrategy": "wave_theory",
    }
    for opt_key, cfg_key in weight_map.items():
        if opt_key in best_params:
            val = best_params[opt_key]
            content = re.sub(
                rf'("{cfg_key}":\s*)[\d.\-e]+',
                rf'\g<1>{val}',
                content
            )

    # ═══ TREND_INDICATOR_WEIGHTS ═══
    trend_weight_map = {
        "trend_price_breakout_weight": "price_breakout",
        "trend_volume_confirmation_weight": "volume_confirmation",
        "trend_momentum_oscillator_weight": "momentum_oscillator",
        "trend_moving_average_weight": "moving_average",
    }
    for opt_key, cfg_key in trend_weight_map.items():
        if opt_key in best_params:
            val = best_params[opt_key]
            content = re.sub(
                rf'("{cfg_key}":\s*)[\d.\-e]+',
                rf'\g<1>{val}',
                content
            )

    # ═══ TREND_THRESHOLDS ═══
    trend_thresh_map = {
        "trend_strong_threshold": "strong_trend",
        "trend_weak_threshold": "weak_trend",
        "trend_volume_spike": "volume_spike",
        "trend_oversold": "oversold",
        "trend_overbought": "overbought",
    }
    for opt_key, cfg_key in trend_thresh_map.items():
        if opt_key in best_params:
            val = best_params[opt_key]
            if "sold" in opt_key or "bought" in opt_key:
                val = int(round(val))
            content = re.sub(
                rf'("{cfg_key}":\s*)[\d.\-e]+',
                rf'\g<1>{val}',
                content
            )

    # ═══ CONFIDENCE_THRESHOLDS ═══
    conf_map = {
        "confidence_high": "high_confidence",
        "confidence_medium": "medium_confidence",
    }
    for opt_key, cfg_key in conf_map.items():
        if opt_key in best_params:
            val = best_params[opt_key]
            content = re.sub(
                rf'("{cfg_key}":\s*)[\d.\-e]+',
                rf'\g<1>{val}',
                content
            )

    CONFIG_PATH.write_text(content, encoding="utf-8")
    logger.info("✏️  配置已更新")


def restart_ea():
    """杀掉旧 EA → 清日志 → 启动新 EA（应用优化后的配置）"""
    import subprocess
    logger.info("🔄 重启 EA...")

    # 杀旧进程
    subprocess.run(["pkill", "-f", "python.*run/realtime.py"], capture_output=True)
    import time; time.sleep(2)
    subprocess.run(["pkill", "-9", "-f", "python.*run/realtime.py"], capture_output=True)
    time.sleep(1)

    # 清空旧交易记录
    for f in PROJECT_DIR.glob("realtime_trades_*"):
        f.unlink(missing_ok=True)

    # 清日志
    log_file = PROJECT_DIR / "logs" / "strategy.log"
    log_file.write_text("")

    # 启动新 EA（后台）
    log = open(log_file, "a")
    proc = subprocess.Popen(
        [sys.executable, "run/realtime.py"],
        cwd=str(PROJECT_DIR),
        stdout=log, stderr=subprocess.STDOUT,
    )
    pid_file = PROJECT_DIR / ".ea_pid"
    pid_file.write_text(str(proc.pid))
    logger.info(f"✅ EA 已重启 PID={proc.pid}")


def main():
    logger.info("=" * 60)
    logger.info("📅 每日自动优化启动")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. 备份当前配置
    backup_config()

    # 2. 运行优化
    try:
        best_params = run_optimizer()
    except Exception as e:
        logger.error(f"优化失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 3. 写入 config.py
    update_config(best_params)

    # 4. 触发重启
    restart_ea()

    logger.info("✅ 每日优化流程完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
