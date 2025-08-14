# MetaTrader 5 智能交易系统 (EA套件)

基于Python的MetaTrader 5智能交易系统，通过多策略组合和动态权重管理实现自动化量化交易。

## 🚀 核心特性

- **多策略组合**: 集成10种量化策略，通过加权投票生成稳健交易信号
- **动态权重管理**: 根据市场状态动态调整策略权重，提升适应性
- **完整风控体系**: 追踪止损、时间退出、资金管理等多层次风险控制
- **遗传算法优化**: 自动寻找最佳策略参数和权重组合
- **多模式运行**: 支持回测、实盘交易、模拟运行三种模式
- **架构重构**: 采用依赖注入设计，提高代码可维护性和扩展性

## 📁 项目结构

```
mt5_python_ea_suite/
├── core/                          # 核心组件
│   ├── data_providers.py         # 数据提供者（实盘/回测/模拟）
│   ├── utils.py                   # 工具函数
│   └── risk/                      # 风险管理模块
│       ├── position_manager.py    # 持仓管理器
│       ├── market_state.py        # 市场状态分析
│       └── __init__.py
├── execution/                     # 执行模块
│   ├── realtime_trader.py         # 实时交易器
│   ├── backtest_engine.py         # 回测引擎
│   ├── dynamic_weights.py         # 动态权重管理
│   └── __init__.py
├── strategies/                    # 交易策略
│   ├── ma_cross.py               # 均线交叉策略
│   ├── rsi.py                    # RSI策略
│   ├── bollinger.py              # 布林带策略
│   ├── macd.py                   # MACD策略
│   ├── kdj.py                    # KDJ策略
│   ├── turtle.py                 # 海龟交易法则
│   ├── mean_reversion.py         # 均值回归策略
│   ├── momentum_breakout.py      # 动量突破策略
│   ├── daily_breakout.py         # 日内突破策略
│   ├── wave_theory.py            # 波浪理论策略
│   └── base_strategy.py          # 策略基类
├── config.py                     # 配置文件
├── main.py                       # 主入口（优化器）
├── start_backtest.py            # 回测启动脚本
├── start_realtime.py            # 实时交易启动脚本
├── optimizer.py                 # 遗传算法优化器
├── logger.py                    # 日志系统
└── requirements.txt              # 依赖包
```

## 🛠️ 安装与配置

### 环境要求

- Python 3.7+
- MetaTrader 5 终端
- 网络连接

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd mt5_python_ea_suite
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv myenv
   # Windows
   myenv\Scripts\activate
   # macOS/Linux
   source myenv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置MT5终端**
   - 确保MT5终端已登录
   - 启用自动交易（点击"自动交易"按钮或按Ctrl+E）
   - 确保账户允许自动交易

## 🎯 策略说明

### 技术指标策略
- **均线交叉 (MACrossStrategy)**: 基于快慢均线交叉的买卖信号
- **相对强弱指数 (RSIStrategy)**: 使用RSI指标判断超买超卖
- **布林带 (BollingerStrategy)**: 基于价格与布林带的相对位置
- **MACD策略**: MACD指标的金叉死叉信号
- **KDJ策略**: KDJ指标的超买超卖信号

### 价格行为策略
- **海龟交易法则 (TurtleStrategy)**: 基于价格突破的顺势交易
- **均值回归 (MeanReversionStrategy)**: 价格偏离均值时的回归交易
- **动量突破 (MomentumBreakoutStrategy)**: 基于动量指标的突破信号
- **日内突破 (DailyBreakoutStrategy)**: 基于日内价格区间的突破

### 复杂策略
- **波浪理论 (WaveTheoryStrategy)**: 结合EMA、ADX、动量等多指标的趋势分析

## 🚦 运行模式

### 1. 回测模式

```bash
# 运行历史回测
python start_backtest.py
```

回测结果将生成：
- `backtest_trades.csv`: 交易记录CSV格式
- `backtest_trades.json`: 交易记录JSON格式
- 详细的性能报告（胜率、盈亏比、最大回撤等）

### 2. 实时交易模式

```bash
# 启动实时交易（模拟运行）
python start_realtime.py

# 启动实盘交易（需要确认）
python start_realtime.py
# 输入 'YES' 确认实盘交易
```

### 3. 参数优化模式

```bash
# 运行遗传算法优化
python optimizer.py
```

优化结果将生成：
- `optimization_results_YYYYMMDD_HHMMSS.json`: 详细优化结果
- `optimization_best_params_YYYYMMDD_HHMMSS.csv`: 最佳参数
- `optimization_history_YYYYMMDD_HHMMSS.csv`: 优化历史
- `optimization_report_YYYYMMDD_HHMMSS.txt`: 优化报告

## ⚙️ 关键配置

### 交易参数
```python
# 基础配置
SYMBOL = "XAUUSD"              # 交易品种
TIMEFRAME = 1                  # 时间周期（1分钟）
INITIAL_CAPITAL = 20000        # 初始资金
SPREAD = 16                    # 点差

# 回测配置
BACKTEST_COUNT = 30000         # 回测数据量
BACKTEST_START_DATE = "2025-05-01"
BACKTEST_END_DATE = "2025-08-01"
```

### 风险管理
```python
RISK_CONFIG = {
    "stop_loss_pct": -0.01,           # 止损阈值（-1%）
    "take_profit_pct": 0.20,          # 止盈阈值（20%）
    "profit_retracement_pct": 0.10,    # 追踪止损回撤阈值（10%）
    "min_profit_for_trailing": 0.01,  # 启动追踪止损的最小盈利（1%）
    "max_holding_minutes": 60,        # 最大持仓时间（分钟）
    "min_profit_for_time_exit": 0.001 # 时间退出最小盈利（0.1%）
}
```

### 信号阈值
```python
SIGNAL_THRESHOLDS = {
    "buy_threshold": 1.5,      # 买入信号阈值
    "sell_threshold": -1.5      # 卖出信号阈值
}
```

## 🛡️ 风险控制

### 多层次风控体系

1. **止损保护**
   - 固定百分比止损
   - 追踪止损（基于盈利回撤）
   - 动态止损位调整

2. **时间管理**
   - 最大持仓时间限制
   - 盈利未达标强制平仓
   - 市场状态适应性调整

3. **资金管理**
   - 初始资金分配
   - 多空仓位资金分配
   - 动态权益更新

4. **持仓限制**
   - 最大持仓数量控制
   - 交易方向限制（多/空/双向）
   - 信号强度过滤

### 动态权重管理

系统根据市场状态动态调整策略权重：
- **趋势识别**: 价格突破、成交量确认、动量指标
- **市场状态**: 强趋势、弱趋势、震荡市
- **置信度评估**: 高置信度、中等置信度
- **权重调整**: 根据市场状态调整各策略权重

## 📊 性能监控

### 实时监控
- 当前持仓状态
- 浮动盈亏计算
- 策略信号强度
- 账户权益更新

### 历史记录
- 交易记录CSV/JSON格式导出
- 性能指标统计
- 参数优化历史
- 峰值盈利数据持久化

## 🔧 故障排除

### 常见问题

1. **MT5连接失败**
   - 确保MT5终端已启动并登录
   - 检查网络连接
   - 验证MT5终端版本

2. **自动交易被禁用**
   - 点击MT5终端"自动交易"按钮
   - 按Ctrl+E快捷键
   - 检查账户权限

3. **订单执行失败**
   - 检查账户资金
   - 验证品种信息
   - 确认交易时间

4. **数据获取失败**
   - 检查日期范围设置
   - 验证MT5服务器连接
   - 调整数据量设置

### 日志文件

- `logs/error.log`: 错误日志
- `logs/strategy.log`: 策略执行日志
- `position_peaks.json`: 持仓峰值数据

## 📈 优化建议

1. **参数优化**
   - 使用遗传算法优化策略参数
   - 定期重新优化以适应市场变化
   - 保存最佳参数组合

2. **策略调整**
   - 根据市场情况调整策略权重
   - 添加新的策略模块
   - 优化信号阈值

3. **风险控制**
   - 定期检查止损设置
   - 监控最大回撤
   - 调整资金分配比例

## ⚠️ 风险提示

- **实盘交易前务必充分测试**
- **建议先使用模拟运行模式**
- **不要投入超过风险承受能力的资金**
- **定期监控系统运行状态**
- **保持策略的持续优化**

## 📝 更新日志

### v2.0 (重构版)
- 架构重构，采用依赖注入设计
- 新增动态权重管理模块
- 完善风险控制体系
- 优化持仓管理和数据持久化
- 改进信号组合逻辑

### v1.0
- 基础多策略组合框架
- 遗传算法参数优化
- 回测和实时交易功能
- 基本风险控制

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进系统！

## 📄 许可证

本项目仅供学习和研究使用，实盘交易风险自负。