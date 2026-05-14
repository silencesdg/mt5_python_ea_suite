import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.data.abc import DataProvider


class AttrDict(dict):
    """支持属性访问的 dict，兼容 isinstance(x, dict) 检查"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class RemoteDataProvider(DataProvider):
    """远端数据提供者 — 通过 HTTP 调用 MT5 代理服务"""

    def __init__(self, host="127.0.0.1", port=5555):
        self.base_url = f"http://{host}:{port}/api"
        self._session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retry))

    @property
    def is_live(self):
        return True

    def initialize(self):
        try:
            resp = self._session.post(f"{self.base_url}/initialize", timeout=10)
            return resp.json().get("success", False)
        except Exception:
            return False

    def shutdown(self):
        try:
            self._session.post(f"{self.base_url}/shutdown", timeout=5)
        except Exception:
            pass

    def get_current_price(self, symbol):
        try:
            resp = self._session.get(f"{self.base_url}/price/{symbol}", timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            data['time'] = pd.to_datetime(data['time'], unit='s')
            return data
        except Exception:
            return None

    def get_historical_data(self, symbol, timeframe, count, **kwargs):
        try:
            resp = self._session.get(
                f"{self.base_url}/historical/{symbol}/{timeframe}/{count}",
                timeout=30,
            )
            if resp.status_code != 200:
                return None
            return resp.json()["rates"]
        except Exception:
            return None

    def get_account_info(self):
        try:
            resp = self._session.get(f"{self.base_url}/account", timeout=10)
            if resp.status_code != 200:
                return None
            return AttrDict(resp.json())
        except Exception:
            return None

    def get_positions(self, symbol):
        try:
            resp = self._session.get(f"{self.base_url}/positions/{symbol}", timeout=10)
            if resp.status_code != 200:
                return None
            return [AttrDict(p) for p in resp.json()["positions"]]
        except Exception:
            return None

    def get_symbol_info(self, symbol):
        try:
            resp = self._session.get(f"{self.base_url}/symbol/{symbol}", timeout=10)
            if resp.status_code != 200:
                return None
            return AttrDict(resp.json())
        except Exception:
            return None

    def send_order(self, symbol, order_type, volume, sl=None, tp=None):
        try:
            body = {"symbol": symbol, "order_type": order_type, "volume": volume}
            if sl is not None:
                body["sl"] = sl
            if tp is not None:
                body["tp"] = tp
            resp = self._session.post(
                f"{self.base_url}/order",
                json=body,
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            return AttrDict(resp.json())
        except Exception:
            return None

    def close_position(self, ticket, symbol, volume):
        try:
            resp = self._session.post(
                f"{self.base_url}/close",
                json={"ticket": ticket, "symbol": symbol, "volume": volume},
                timeout=10,
            )
            if resp.status_code != 200:
                return False
            return resp.json().get("success", False)
        except Exception:
            return False
