#!/bin/bash
# 每日自动优化 + 重启 EA
# 设置 PYTHONPATH 确保 cron 环境下能找到 user-site packages（如 deap, moocore 等）
export PYTHONPATH="$HOME/.local/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages:$PYTHONPATH"
cd /home/songkl/mt5_python_ea_suite
exec python3 scripts/daily_optimize.py "$@"
