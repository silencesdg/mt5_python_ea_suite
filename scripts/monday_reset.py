#!/usr/bin/env python3
"""开市后全平持仓 + 清峰值 + 重启 EA"""
import requests, json, time, sys, os, subprocess

API = "http://192.168.1.5:5555/api"
SYMBOL = "XAUUSDz"
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

# 2. 获取所有持仓并平仓
log("获取持仓...")
r = requests.get(f"{API}/positions/{SYMBOL}")
positions = r.json().get("positions", [])
log(f"共 {len(positions)} 单")

total_pnl = 0
for p in positions:
    resp = requests.post(f"{API}/close",
        json={"ticket": p["ticket"], "symbol": SYMBOL, "volume": p["volume"]})
    ok = resp.json().get("success", False)
    total_pnl += p["profit"]
    log(f"{'✅' if ok else '❌'} {p['ticket']} ${p['profit']:+.2f}")
log(f"总盈亏: ${total_pnl:+.2f}")

# 3. 清峰值数据
peak_file = os.path.join(PROJECT_DIR, "position_peaks.json")
if os.path.exists(peak_file):
    os.remove(peak_file)
    log("✅ 峰值数据已清除")

# 4. 杀掉旧 EA，启动新 EA
subprocess.run(["pkill", "-9", "-f", "python.*run/realtime.py"], capture_output=True)
time.sleep(2)

log_file = os.path.join(PROJECT_DIR, "logs", "strategy.log")
with open(log_file, "a") as f:
    proc = subprocess.Popen(
        [sys.executable, "run/realtime.py"],
        cwd=PROJECT_DIR,
        stdout=f, stderr=subprocess.STDOUT,
    )
log(f"✅ EA 已重启 PID={proc.pid}")
