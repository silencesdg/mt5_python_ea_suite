@echo off
REM MT5 EA Suite 开机自启脚本
REM 1. 启动 MT5 终端
REM 2. 等待 MT5 初始化
REM 3. 启动代理服务

echo [%date% %time%] 启动 MT5 ...
start "" "E:\Program Files\MetaTrader 5\terminal64.exe"

echo [%date% %time%] 等待 MT5 初始化 (30秒) ...
timeout /t 30 /nobreak >nul

echo [%date% %time%] 启动代理服务 ...
cd /d "D:\projects\mt5_python_ea_suite"
python -m run.server

