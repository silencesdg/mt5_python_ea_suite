#!/bin/bash
# cron 环境修复：设置正确的 PYTHONPATH 后运行优化器
export PYTHONPATH="/home/songkl/.hermes/profiles/bot1/home/.local/lib/python3.13/site-packages:$PYTHONPATH"
cd /home/songkl/mt5_python_ea_suite
exec /usr/bin/python3 scripts/daily_optimize.py "$@"
