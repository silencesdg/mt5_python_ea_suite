# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

基于Python的MetaTrader 5智能交易系统(EA)套件，集成10种量化策略，通过加权投票和阈值过滤组合信号，支持动态权重管理（根据市场状态调整策略权重）。

## 实际入口点

`main.py` 为空壳，实际入口为：

```bash
python start_backtest.py     # 回测模式
python start_realtime.py     # 实时交易（默认模拟，需输入YES确认实盘）
python optimizer.py          # 遗传算法参数优化
```

## 核心架构

### 依赖注入设计

`core/data_providers.py` 定义了 `DataProvider` 抽象基类（9个抽象方法），三种实现：

- `LiveDataProvider` — 真实MT5 API调用
- `DryRunDataProvider` — 模拟交易（委托LiveDataProvider获取价格，不发送真实订单）
- `BacktestDataProvider` — 回测（维护DataFrame + `current_index`，通过 `tick()` 推进）

所有策略和执行模块接收 `DataProvider` 接口，不直接调用MT5。这是理解整个系统解耦的关键。

### 风险管理门面

`core/risk/__init__.py` 中的 `RiskController` 是门面类，组合了：
- `PositionManager` (`position_manager.py`) — 开仓、监控、止损/止盈/追踪止损/时间退出、资金管理
- `MarketStateAnalyzer` (`market_state.py`) — 四维度趋势检测（价格突破、成交量确认、动量、均线），返回 `("uptrend"/"downtrend"/"ranging"/"none", confidence)`

### 动态权重系统

`execution/dynamic_weights.py` 中的 `DynamicWeightManager`：
1. 使用 `MarketStateAnalyzer` 判断当前市场状态
2. 根据状态从 `config.py:MARKET_STATE_WEIGHTS` 获取对应权重
3. 按置信度决定使用市场状态权重还是 `DEFAULT_WEIGHTS`
4. 返回 `[(策略实例, 权重), ...]` 列表供信号组合使用

### 两阶段回测设计

`start_backtest.py` 的回测采用两阶段：
1. **预生成阶段**：遍历所有策略 `run_backtest(df)` 生成信号列，加权求和后通过 `SIGNAL_THRESHOLDS` 阈值过滤得到最终信号序列
2. **模拟执行阶段**：单遍 `iterrows` 逐行推进，通过 `RiskController` 模拟开仓/监控/平仓

### 信号组合逻辑

```
加权和 = sum(各策略信号 * 权重)
最终信号 = 1  (买入)   if 加权和 > buy_threshold (1.5)
            -1 (卖出)   if 加权和 < sell_threshold (-1.5)
            0  (无信号)  otherwise
```

阈值在 `config.py:SIGNAL_THRESHOLDS` 中配置。

## 策略开发规范

所有策略继承 `strategies/base_strategy.py` 的 `BaseStrategy`：

```python
class BaseStrategy:
    def __init__(self, data_provider, symbol, timeframe):  # 接收 data_provider 而非直接访问 MT5
    def generate_signal(self) -> int:    # 实时信号：通过 self.data_provider 获取当前价格
    def run_backtest(self, df) -> pd.Series:  # 向量化回测：接收 DataFrame，返回 -1/0/1 序列
    def _log_signal(self, signal, reason):    # 基类提供，记录买卖信号到日志
```

**添加新策略步骤：**
1. 在 `strategies/` 下创建新文件，继承 `BaseStrategy`
2. 实现 `generate_signal()` 和 `run_backtest(self, df)`
3. 在 `config.py:STRATEGY_CONFIG` 添加参数字典，在 `config.py:DEFAULT_WEIGHTS` 和 `config.py:MARKET_STATE_WEIGHTS` 添加权重
4. 在 `execution/dynamic_weights.py` 的 `strategy_blueprints` 列表和 `optimizer.py` 的导入中注册

## 配置系统

`config.py` 包含20+个配置字典，核心：

| 配置 | 用途 |
|---|---|
| `SYMBOL`, `TIMEFRAME`, `INITIAL_CAPITAL` | 基础交易参数（当前: XAUUSD, M1, 20000） |
| `STRATEGY_CONFIG` | 各策略的参数字典（键名为策略简称） |
| `DEFAULT_WEIGHTS` | 默认策略权重（优化后） |
| `MARKET_STATE_WEIGHTS` | 各市场状态(up/down/ranging)下的策略权重 |
| `SIGNAL_THRESHOLDS` | 买卖信号阈值 |
| `RISK_CONFIG` | 止损/止盈/追踪止损/持仓时间等风控参数 |
| `MARKET_STATE_CONFIG` | 趋势检测参数 |
| `GENETIC_OPTIMIZER_CONFIG` | 遗传算法参数（种群、代数、交叉/变异概率） |
| `BACKTEST_CONFIG` / `REALTIME_CONFIG` | 回测/实盘专项配置 |

## 依赖

```
MetaTrader5  # MT5 Python API
pandas
deap         # 遗传算法框架
tqdm         # 进度条
```

## 语言要求

- 所有代码注释、文档、日志必须使用中文
- 日志文件: `logs/strategy.log`, `logs/error.log`
- 日志器使用单例模式（`logger.py`）
