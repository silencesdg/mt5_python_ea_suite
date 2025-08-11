# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概述

这是一个基于Python的MetaTrader 5智能交易系统(EA)套件，用于量化交易策略。系统通过加权投票和阈值过滤组合多个交易策略，生成稳健的交易决策。

## 架构设计

### 核心组件

- **主入口**: `main.py` 提供三种执行模式：
  - `run_backtest()`: 使用当前配置权重进行单次回测
  - `run_optimizer()`: 遗传算法优化寻找最佳权重
  - `run_realtime()`: 与MT5集成的实时交易

- **配置文件**: `config.py` 包含：
  - `STRATEGIES`: (策略实例, 权重) 元组列表
  - 交易参数（品种、时间周期、初始资金）

- **回测引擎**: `backtest.py` 处理：
  - 历史数据策略执行
  - 使用加权和的信号组合
  - 防止未来函数的收益计算

- **遗传优化器**: `optimizer.py`:
  - 使用DEAP库进行遗传算法
  - 优化策略权重以获得最大收益
  - 支持并行处理提高性能

- **工具函数**: `utils.py` 提供MT5连接管理和交易功能
- **日志系统**: `logger.py` 处理控制台/文件双日志输出，支持UTF-8编码

### 策略架构

`strategies/` 目录中的所有策略都遵循一致的模式：

```python
class Strategy:
    def __init__(self):  # 初始化参数
    def _calculate_indicators(self, df):  # 计算技术指标
    def generate_signal(self):  # 实时交易信号生成
    def run_backtest(self, df):  # 回测信号生成
```

### 信号组合逻辑

系统通过以下方式组合策略信号：
1. 加权求和：`sum(信号 * 权重，针对所有策略)`
2. 信号决策：加权和大于0买入，小于0卖出
3. 返回用于回测的组合信号序列

## 开发命令

### 环境设置
```bash
# 创建并激活虚拟环境
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS/Linux
source myenv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行模式
```bash
# 运行单次回测（使用config.py权重）
python main.py

# 运行遗传优化（寻找最佳权重）
python main.py  # 取消注释 run_optimizer() 调用

# 运行实时交易（使用config.py权重）
python main.py  # 取消注释 run_realtime() 调用
```

## 关键开发模式

### 添加新策略
1. 在 `strategies/` 目录创建新文件
2. 实现必需的Strategy类方法
3. 添加导入和策略实例到 `config.py:STRATEGIES`

### 配置管理
- 策略权重需要从优化器结果手动更新
- 所有交易参数集中在 `config.py` 中

### 错误处理
- 策略执行中的全面异常处理
- 日志文件使用UTF-8编码
- 操作前进行MT5连接验证

### 性能考虑
- 优化器预加载历史数据避免重复I/O
- 遗传算法启用并行处理
- 日志器使用单例模式防止重复处理器

## 语言要求

- **所有开发和文档必须使用中文**（根据GEMINI.md）
- 错误日志在 `logs/error.log`
- 策略日志在 `logs/strategy.log`