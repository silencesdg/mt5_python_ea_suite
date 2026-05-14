#!/bin/bash
# 重启 EA：杀旧进程 → 应用新配置 → 启动新进程
# 由 daily_optimize.py 或 cron 调用

set -e
PROJECT_DIR="/home/songkl/mt5_python_ea_suite"
cd "$PROJECT_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 重启 EA..."

# 1. 杀掉所有 realtime.py 进程
pkill -f "python.*run/realtime.py" 2>/dev/null || true
sleep 2

# 强制清理残留
pkill -9 -f "python.*run/realtime.py" 2>/dev/null || true
sleep 1

# 2. 清空旧交易记录（新配置从零开始）
rm -f realtime_trades_*.json realtime_trades_*.csv

# 3. 清空日志
> logs/strategy.log

# 4. 启动新 EA
nohup python3 run/realtime.py >> /dev/null 2>&1 &
NEW_PID=$!
echo $NEW_PID > .ea_pid

echo "[$(date '+%Y-%m-%d %H:%M:%S')] EA 已重启 PID=$NEW_PID"
