"""策略注册表 — 单例，消除代码中多处的策略列表重复

用法:
    registry = StrategyRegistry()
    registry.register('ma_cross', MACrossStrategy)
    registry.instantiate_all(SYMBOL, TIMEFRAME)  # 或传入参数字典
    config_key = registry.class_name_to_config_key('MACrossStrategy')
"""

from config import STRATEGY_CONFIG


class StrategyRegistry:
    """策略注册表单例 — 所有策略信息的唯一数据源"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies = {}    # config_key → StrategyClass
            cls._instance._instances = {}     # config_key → instance
        return cls._instance

    def register(self, config_key: str, strategy_class: type) -> None:
        self._strategies[config_key] = strategy_class

    def instantiate_all(self, symbol: str, timeframe: int,
                        params_dict: dict = None,
                        data_provider=None) -> dict:
        """实例化所有已注册策略

        Args:
            data_provider: 回测传None（run_backtest不需要），实盘传DataProvider实例
        Returns:
            {config_key: strategy_instance}
        """
        if params_dict is None:
            params_dict = STRATEGY_CONFIG

        instances = {}
        for key, cls in self._strategies.items():
            params = params_dict.get(key, {})
            instances[key] = cls(data_provider, symbol, timeframe, **params)
        self._instances = instances
        return instances

    @property
    def instances(self) -> dict:
        return self._instances

    @property
    def all_strategies(self) -> dict:
        """返回 {config_key: StrategyClass}"""
        return dict(self._strategies)

    def class_name_to_config_key(self, class_name: str) -> str | None:
        for key, cls in self._strategies.items():
            if cls.__name__ == class_name:
                return key
        return None

    def config_key_to_class_name(self, config_key: str) -> str:
        return self._strategies[config_key].__name__

    def get_instance(self, config_key: str):
        return self._instances.get(config_key)

    def __contains__(self, config_key: str) -> bool:
        return config_key in self._strategies
