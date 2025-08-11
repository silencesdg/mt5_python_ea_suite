# 架构重构说明

## 重构概述

本次重构将原有的单一策略架构分离为**策略层**和**风险管理层**，实现了更清晰的功能分离和动态权重配置。

## 新架构结构

### 1. 策略层 (strategies/)
包含纯粹的交易策略，每个策略只负责生成买入/卖出信号：
- `ma_cross.py` - 均线交叉策略
- `rsi.py` - RSI超买超卖策略  
- `bollinger.py` - 布林带策略
- `macd.py` - MACD策略
- `kdj.py` - KDJ策略
- `turtle.py` - 海龟交易法则
- `daily_breakout.py` - 日内突破策略
- `mean_reversion.py` - 均值回归策略
- `momentum_breakout.py` - 动量突破策略

### 2. 风险管理层 (risk_management/)
- `market_state.py` - 市场状态分析器（基于原resilient_trend）
- `position_manager.py` - 仓位管理器（基于原profit_protect）
- `__init__.py` - 风险管理控制器

### 3. 动态权重系统
- `dynamic_weights.py` - 动态权重管理器

## 核心改进

### 1. 市场状态判断
- 将`resilient_trend`重构为市场状态分析器
- 判断市场状态：`uptrend`、`downtrend`、`ranging`、`none`
- 提供置信度评估（0.0-1.0）

### 2. 动态权重配置
根据市场状态自动调整策略权重：
- **上升趋势**：侧重趋势跟踪策略（均线交叉、动量突破）
- **下降趋势**：侧重趋势跟踪和反转策略
- **震荡市场**：侧重均值回归和超买超卖策略
- **无明确趋势**：使用平衡的权重配置

### 3. 综合风险管理
- **止损机制**：固定止损10%
- **止盈机制**：固定止盈20% + 追踪止损
- **仓位控制**：最大单笔仓位10%
- **日内限制**：最大日亏损5%

## 使用方法

### 1. 运行回测
```bash
python main.py
```
回测会使用当前市场状态对应的权重配置进行测试。

### 2. 运行实盘交易
```bash
# 在main.py中取消注释
run_realtime()
```
实盘交易会实时分析市场状态并动态调整权重。

### 3. 运行优化
```bash
# 在main.py中取消注释  
run_optimizer()
```
优化器会寻找最佳的基础权重配置。

## 配置说明

### config.py 新配置项
```python
# 风险管理参数
RISK_CONFIG = {
    "stop_loss_pct": -0.10,        # 固定止损：亏损10%
    "profit_retracement_pct": 0.30,  # 利润回撤30%止盈
    "min_profit_for_trailing": 0.05,  # 追踪止损激活阈值：利润5%
    "take_profit_pct": 0.20,         # 固定止盈：盈利20%
    "max_position_size": 0.1,         # 最大仓位10%
    "max_daily_loss": -0.05,          # 最大日亏损5%
}

# 市场状态分析参数
MARKET_STATE_CONFIG = {
    "trend_period": 50,
    "retracement_tolerance": 0.30,
}

# 默认权重（低置信度时使用）
DEFAULT_WEIGHTS = {
    "ma_cross": 1.5,
    "rsi": 1.5,
    "bollinger": 1.5,
    # ... 其他策略
}
```

## 优势

1. **更清晰的功能分离**：策略专注于信号生成，风险管理专注于风险控制
2. **动态适应性**：根据市场状态自动调整策略权重
3. **更强的风险控制**：多层次的风险管理机制
4. **更好的可扩展性**：易于添加新策略和风险管理规则
5. **更稳定的性能**：通过权重分散降低单一策略风险

## 兼容性

- 保持与现有回测和优化系统的兼容性
- 策略接口保持不变，现有策略无需修改
- 配置文件结构更新，但提供了向后兼容