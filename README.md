# MT5 Python EA Suite (MetaTrader 5 量化交易策略套件)

这是一个基于Python与MetaTrader 5 (MT5)平台交互的、经过重构的现代化量化交易策略套件。它提供了一个清晰、可扩展的框架，用于开发、回测和运行由多个交易策略组成的投资组合。

---

## 核心概念

- **多策略组合**: 系统的核心思想不是依赖于单一策略，而是将多个不同类型的策略（如趋势跟踪、均值回归等）组合起来，通过动态加权生成一个更稳健的交易决策。

- **动态权重**: 系统能根据对市场状态（如趋势、震荡）的分析，动态地调整不同策略的权重，在趋势行情中倚重趋势策略，在震荡行情中侧重于摆动策略。

- **依赖注入架构**: 项目采用依赖注入的设计模式，将核心交易逻辑与数据来源彻底分离。无论是处理实时行情、模拟数据还是历史数据，核心逻辑都保持不变，极大地保证了回测与实盘的一致性，并方便了未来的扩展。

- **配置即代码**: 所有的系统行为，从交易参数到策略组合，都通过 `config.py` 文件进行集中配置，清晰直观。

---

## 系统架构

```
/ (根目录)
├── core/                   # 核心逻辑模块
│   ├── __init__.py
│   ├── data_providers.py   # 数据提供者 (实盘/模拟/回测)
│   ├── risk/               # 风险管理 (包含PositionManager)
│   └── utils.py            # 通用工具函数
├── execution/              # 执行逻辑模块
│   ├── __init__.py
│   ├── backtest_engine.py  # 回测引擎
│   ├── dynamic_weights.py  # 动态权重管理器
│   └── realtime_trader.py  # 实时交易处理器
├── strategies/             # 策略库
│   ├── __init__.py
│   ├── base_strategy.py
│   └── ... (具体策略)
├── logs/                   # 日志目录
├── start_realtime.py       # [入口] 启动实时/模拟交易
├── start_backtest.py       # [入口] 启动回测
├── config.py               # [核心] 全局配置文件
├── requirements.txt        # Python依赖
└── README.md               # 项目说明
```

---

## 如何运行

#### 1. 环境准备

- 确保您的电脑上已经安装了MetaTrader 5交易终端。
- 在MT5中，进入 `工具 > 选项 > EA交易`，确保勾选了“允许算法交易”和“允许DLL导入”。
- 安装Python 3.x并创建虚拟环境。
- **安装依赖**: 
  ```bash
  pip install -r requirements.txt
  ```

#### 2. 核心配置 (`config.py`)

这是您与系统交互的唯一配置中心。在这里您可以调整几乎所有的参数，例如：
- `INITIAL_CAPITAL`: 初始资金。
- `USE_DATE_RANGE` / `BACKTEST_COUNT`: 选择回测模式（按日期或按K线数）。
- `REALTIME_CONFIG`: **实盘/模拟交易的配置**，包括`dry_run`模式的开关。
- `RISK_CONFIG`: 止盈、止损等风险参数。
- `CAPITAL_ALLOCATION`: 多、空方向的资金分配比例。
- `DEFAULT_WEIGHTS`: 策略的基础权重。

#### 3. 运行回测

直接运行回测启动脚本。所有配置将从 `config.py` 读取。

```bash
python start_backtest.py
```
回测结束后，会打印详细的性能报告，并将完整的交易记录保存到 `backtest_trades.csv` 和 `backtest_trades.json` 文件中。

#### 4. 运行实盘或模拟交易

在 `config.py` 中，将 `REALTIME_CONFIG` 里的 `dry_run` 设置为 `False`（实盘）或 `True`（模拟），然后运行：

```bash
python start_realtime.py
```
程序会根据您的配置，以实盘或模拟模式启动。

---

## 如何添加一个新策略

新架构下添加策略非常简单：

1.  在 `strategies/` 目录下，创建一个新的Python文件，例如 `my_strategy.py`。
2.  在该文件中，创建一个继承自 `base_strategy.BaseStrategy` 的策略类。
3.  实现您的 `__init__` 和 `generate_signal` (用于实盘) 或 `run_backtest` (用于回测) 方法。
4.  在 `execution/dynamic_weights.py` 文件的 `__init__` 方法中，将您的新策略添加到 `self.strategy_instances` 字典中。
5.  在 `config.py` 的 `DEFAULT_WEIGHTS` 中，为您的新策略添加一个基础权重。

完成！您的新策略已经无缝集成到整个系统中。