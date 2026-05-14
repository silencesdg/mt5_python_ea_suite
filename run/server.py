#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MT5 代理服务 — python -m run.server"""

import sys
import os
import threading
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.data import LiveDataProvider
from core.data.serializers import (
    serialize_account_info,
    serialize_position,
    serialize_symbol_info,
    serialize_order_result,
    serialize_rates,
)
from config import SERVER_HOST, SERVER_PORT
from logger import setup_logger, logger

setup_logger("INFO")

# 全局 MT5 连接和线程锁
_provider = LiveDataProvider()
_lock = threading.Lock()


def _ensure_connected():
    """检查 MT5 连接，断开时自动重连"""
    import MetaTrader5 as mt5
    if mt5.terminal_info() is not None:
        return True
    logger.warning("MT5 连接已断开，尝试重连...")
    with _lock:
        ok = _provider.initialize()
    if ok:
        logger.info("MT5 重连成功")
    else:
        logger.error("MT5 重连失败")
    return ok


@asynccontextmanager
async def lifespan(app: FastAPI):
    with _lock:
        if not _provider.initialize():
            logger.error("MT5 初始化失败，服务启动中止")
            sys.exit(1)
    logger.info(f"MT5 代理服务启动于 {SERVER_HOST}:{SERVER_PORT}")
    yield
    with _lock:
        _provider.shutdown()
    logger.info("MT5 代理服务已关闭")


app = FastAPI(title="MT5 Proxy API", lifespan=lifespan)


# ── 请求模型 ──

class OrderRequest(BaseModel):
    symbol: str
    order_type: str
    volume: float
    sl: float = None       # ★ 硬止损价格
    tp: float = None       # ★ 硬止盈价格

class CloseRequest(BaseModel):
    ticket: int
    symbol: str
    volume: float


# ── 端点 ──

@app.post("/api/initialize")
def api_initialize():
    with _lock:
        return {"success": _provider.initialize()}


@app.post("/api/shutdown")
def api_shutdown():
    with _lock:
        _provider.shutdown()
    return {"success": True}


@app.get("/api/price/{symbol}")
def api_price(symbol: str):
    _ensure_connected()
    with _lock:
        data = _provider.get_current_price(symbol)
    if data is None:
        return JSONResponse(status_code=503, content={"error": "无法获取价格"})
    # time 转为 Unix 时间戳
    data['time'] = int(data['time'].timestamp()) if hasattr(data['time'], 'timestamp') else int(data['time'])
    return data


@app.get("/api/historical/{symbol}/{timeframe}/{count}")
def api_historical(symbol: str, timeframe: int, count: int):
    _ensure_connected()
    with _lock:
        rates = _provider.get_historical_data(symbol, timeframe, count)
    if rates is None:
        return JSONResponse(status_code=503, content={"error": "无法获取历史数据"})
    return {"rates": serialize_rates(rates)}


@app.get("/api/account")
def api_account():
    _ensure_connected()
    with _lock:
        info = _provider.get_account_info()
    if info is None:
        return JSONResponse(status_code=503, content={"error": "无法获取账户信息"})
    return serialize_account_info(info)


@app.get("/api/positions/{symbol}")
def api_positions(symbol: str):
    _ensure_connected()
    with _lock:
        positions = _provider.get_positions(symbol)
    if positions is None:
        return {"positions": []}
    return {"positions": [serialize_position(p) for p in positions]}


@app.get("/api/symbol/{symbol}")
def api_symbol(symbol: str):
    _ensure_connected()
    with _lock:
        info = _provider.get_symbol_info(symbol)
    if info is None:
        return JSONResponse(status_code=503, content={"error": "无法获取品种信息"})
    return serialize_symbol_info(info)


@app.post("/api/order")
def api_order(req: OrderRequest):
    _ensure_connected()
    with _lock:
        result = _provider.send_order(req.symbol, req.order_type, req.volume,
                                      sl=req.sl, tp=req.tp)
    if result is None:
        return JSONResponse(status_code=503, content={"error": "下单失败"})
    return serialize_order_result(result)


@app.post("/api/close")
def api_close(req: CloseRequest):
    _ensure_connected()
    with _lock:
        success = _provider.close_position(req.ticket, req.symbol, req.volume)
    return {"success": success}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
