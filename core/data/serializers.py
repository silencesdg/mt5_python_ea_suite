import numpy as np


def _to_native(val):
    """numpy 类型转 Python 原生类型"""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _mt5_to_dict(obj):
    """将 MT5 对象转为 dict，兼容 _asdict() / _fields / __dict__"""
    if hasattr(obj, '_asdict'):
        return {k: _to_native(v) for k, v in obj._asdict().items()}
    if hasattr(obj, '_fields'):
        return {f: _to_native(getattr(obj, f)) for f in obj._fields}
    if hasattr(obj, '__dict__'):
        return {k: _to_native(v) for k, v in obj.__dict__.items()}
    return {}


def serialize_account_info(info):
    return _mt5_to_dict(info) if info else None


def serialize_position(pos):
    return _mt5_to_dict(pos) if pos else None


def serialize_symbol_info(info):
    return _mt5_to_dict(info) if info else None


def serialize_order_result(result):
    return _mt5_to_dict(result) if result else None


def serialize_rates(rates):
    if rates is None:
        return None
    names = rates.dtype.names
    return [{name: _to_native(row[name]) for name in names} for row in rates]
