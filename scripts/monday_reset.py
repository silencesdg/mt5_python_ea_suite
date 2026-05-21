#!/usr/bin/env python3
"""开市后重启 EA（不平仓不清理数据，让持仓自然运行）"""
import time, sys, os, subprocess, requests

API = "http://192.168.1.5:5555/api"
PROJECT_DIR = "/home/songkl/mt5_python_ea_suite"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# 1. 等市场开市（重试最多30次，间隔10秒）
log("等待市场开市...")
for i in range(30):
    try:
        r = requests.get(f"{API}/account", timeout=5)
        if r.status_code == 200:
            acct = r.json()
            if acct.get("trade_allowed"):
                log("✅ 市场已开市")
                break
    except:
        pass
    time.sleep(10)
else:
    log("❌ 等待超时，市场未开市")
    sys.exit(1)

# 2. 杀掉旧 EA，启动新 EA
subprocess.run(["pkill", "-f", "python.*run/realtime.py"], capture_output=True)
time.sleep(2)
subprocess.run(["pkill", "-9", "-f", "python.*run/realtime.py"], capture_output=True)
time.sleep(1)

log_file = os.path.join(PROJECT_DIR, "logs", "strategy.log")
with open(log_file, "a") as f:
    proc = subprocess.Popen(
        [sys.executable, "run/realtime.py"],
        cwd=PROJECT_DIR,
        stdout=f, stderr=subprocess.STDOUT,
    )
log(f"✅ EA 已重启 PID={proc.pid}")
