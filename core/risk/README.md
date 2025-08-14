# 风险管理模块

此模块包含所有风险管理相关的组件：

- `market_state.py`: 市场状态分析器，基于resilient_trend逻辑
- `position_manager.py`: 仓位管理器，基于profit_protect逻辑  
- `__init__.py`: 风险管理控制器，统一接口