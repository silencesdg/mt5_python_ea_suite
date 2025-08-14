
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# 从 regime_optimizer 脚本中导入状态分类函数
from regime_optimizer import classify_market_regimes

# --- 配置 ---
DATA_FILE = "full_historical_data.parquet"
OUTPUT_HTML_FILE = "regime_visualization.html"

# --- 主函数 ---
def create_regime_candlestick_chart():
    """
    加载数据，分类市场状态，并生成一个按状态染色的交互式K线图。
    """
    # 1. 加载数据
    if not os.path.exists(DATA_FILE):
        print(f"错误: 数据文件 {DATA_FILE} 不存在。请先运行 data_downloader.py。")
        return

    print(f"正在从 {DATA_FILE} 加载数据...")
    full_df = pd.read_parquet(DATA_FILE)
    print(f"数据加载完成，共 {len(full_df)} 条记录。")

    # 2. 分类市场状态
    # 这个函数会返回一个带有 'regime' 列的新DataFrame
    df_classified = classify_market_regimes(full_df)

    print("\n--- 开始生成交互式K线图 ---")

    # 3. 创建Plotly图表对象
    fig = go.Figure()

    # 4. 为每种市场状态分别创建一个K线图层 (Trace)
    # 这是处理大数据和分类染色的最高效方法
    # 我们通过将不属于当前状态的数据设置为NaN来“隐藏”它们，从而只绘制我们想要的K线
    
    regime_colors = {
        "Uptrend": "red",
        "Downtrend": "green",
        "Ranging": "black"
    }

    for regime, color in regime_colors.items():
        print(f"正在为 {regime} 状态创建图层...")
        
        # 创建一个临时DataFrame用于绘图
        df_trace = df_classified.copy()
        
        # 将不属于当前状态的K线数据设置为空值(NaN)
        df_trace.loc[df_trace['regime'] != regime, ['open', 'high', 'low', 'close']] = np.nan
        
        # 添加K线图层
        fig.add_trace(go.Candlestick(
            x=df_trace.index,
            open=df_trace['open'],
            high=df_trace['high'],
            low=df_trace['low'],
            close=df_trace['close'],
            name=regime,
            increasing_line_color=color, # 上涨部分
            decreasing_line_color=color  # 下跌部分
        ))

    # 5. 更新图表布局和样式
    print("正在配置图表样式...")
    fig.update_layout(
        title={
            'text': "市场状态可视化 (XAUUSD - M1)",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="日期",
        yaxis_title="价格",
        legend_title="市场状态",
        template="plotly_dark",  # 使用深色主题
        xaxis_rangeslider_visible=False  # **关键：为提升性能，禁用底部范围滑块**
    )

    # 6. 保存为HTML文件
    try:
        fig.write_html(OUTPUT_HTML_FILE)
        print(f"\n--- 图表生成成功！ ---")
        print(f"已保存到: {os.path.abspath(OUTPUT_HTML_FILE)}")
        print("请用您的浏览器打开此文件以进行交互式分析。")
    except Exception as e:
        print(f"保存HTML文件时出错: {e}")

if __name__ == "__main__":
    create_regime_candlestick_chart()
