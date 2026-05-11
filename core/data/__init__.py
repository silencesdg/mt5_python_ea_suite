from core.data.abc import DataProvider
from core.data.backtest import BacktestDataProvider
from core.data.multi_tf import MultiTimeframeDataStore
from core.data.remote import RemoteDataProvider

# LiveDataProvider 和 DryRunDataProvider 依赖 MetaTrader5，
# 仅在本地模式需要时延迟导入
_LiveDataProvider = None
_DryRunDataProvider = None

def LiveDataProvider(*args, **kwargs):
    global _LiveDataProvider
    if _LiveDataProvider is None:
        from core.data.live import LiveDataProvider as LDP
        _LiveDataProvider = LDP
    return _LiveDataProvider(*args, **kwargs)

def DryRunDataProvider(*args, **kwargs):
    global _DryRunDataProvider
    if _DryRunDataProvider is None:
        from core.data.dryrun import DryRunDataProvider as DDP
        _DryRunDataProvider = DDP
    return _DryRunDataProvider(*args, **kwargs)
