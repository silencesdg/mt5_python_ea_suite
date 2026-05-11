#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""优化器入口 — python -m run.optimize"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.optimize import run_optimizer
from logger import logger


def main():
    print("=" * 60)
    print("MT5 智能交易系统 - 遗传算法参数优化")
    print("=" * 60)

    try:
        best_params, best_fitness = run_optimizer()
        print(f"\n优化完成！最佳适应度: {best_fitness:.2f}")
    except Exception as e:
        import traceback
        logger.error(f"优化器运行出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
