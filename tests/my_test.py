# %% ======================= init =======================
from pathlib import Path
os.chdir(Path(__vsc_ipynb_file__).parent)

from dotenv import load_dotenv
# 加载项目根目录的.env
load_dotenv() 

import sys
project_root = Path(__vsc_ipynb_file__).parent.parent
sys.path.append(str(project_root))

# %% ======================= bark notification =======================
import requests
import os

key = os.getenv("BARK_KEY", "")
title = "脚本提醒"
content = "任务执行完成！"
url = f"https://api.day.app/{key}/{title}/{content}"
requests.get(url)

# %% ======================= back test =======================
import pandas as pd
from scripts.data_pipeline.adjust import forward_adjust
from scripts.data_pipeline.backtest import run_backtest

daily = pd.read_parquet('../data/daily/ts_code=000001.SZ/data.parquet')
xdxr  = pd.read_parquet('../data/xdxr/ts_code=000001.SZ/data.parquet')
adj   = forward_adjust(daily, xdxr)   # 前复权
r     = run_backtest(adj)             # 用复权价回测
print(r.summary())

# %% ======================= 历史 K 线指标 =======================
import pandas as pd
from scripts.data_pipeline.indicators import compute_all
from scripts.data_pipeline.screener.conditions import (
    golden_cross_series,
    hammer_series,
    kdj_golden_cross_series,
)
from scripts.data_pipeline.signals import trend_snapshot

df = pd.read_parquet('../data/daily/ts_code=601088.SH/data.parquet')
df = df.sort_values('trade_date').reset_index(drop=True)

ind = compute_all(df, timeframe='daily')

cutoff = (
    pd.to_datetime(ind['trade_date'].max(), format='%Y%m%d') - pd.DateOffset(months=12)
).strftime('%Y%m%d')
recent = ind['trade_date'] >= cutoff  # 'YYYYMMDD' 字符串按字典序比较即时间序

macd_cross = golden_cross_series(ind)
kdj_cross  = kdj_golden_cross_series(ind)
hammer     = hammer_series(ind)

print('趋势判断：')
print(trend_snapshot(df).to_string())
print()
print('MACD 金叉（截止', ind['trade_date'].max(), '）')
print(ind[macd_cross & recent][['trade_date', 'close', 'DIF', 'DEA']])
print()
print('KDJ 金叉')
print(ind[kdj_cross & recent][['trade_date', 'close', 'K', 'D']])
print()
print('锤头线（下锤线）')
print(ind[hammer & recent][['trade_date', 'open', 'high', 'low', 'close']])

# %% ======================= 1. 下载：tdx_client.TdxDownloader =======================
from scripts.data_pipeline.tdx_client import TdxDownloader

dl = TdxDownloader(Path("../data"))

daily   = dl.download_daily("000001")          # 日K全历史(自动翻页),落盘并返回
print(daily)
# minute  = dl.download_minute("000001", freq=5) # 5分钟线,带 trade_time 列
# dl.download_minute("000001", freq=15)
# dl.download_minute("000001", freq=30)
# dl.download_minute("000001", freq=60)
# xdxr    = dl.download_xdxr("000001")           # 除权除息
# snap    = dl.snapshot("000001")                # 实时快照(hq); snapshot("AAPL") 走 exhq
# print(snap)

# sec   = dl.download_security_list(0)               # 全市场枚举快照(0=SZ / 1=SH)
# idx   = dl.download_index("000001", market=1)      # 指数 K 线；market 必须显式传(000001=上证指数,SH)
# tick  = dl.download_tick("000001", 20240610)       # 指定日分笔成交(YYYYMMDD 或 YYYY-MM-DD)
# tick0 = dl.download_tick_today("000001")           # 当日分笔成交(盘中可能不完整)
# mt    = dl.download_minute_time("000001", 20240610)# 指定日分时(每个交易日 ≈ 240 点)
# mt0   = dl.download_minute_time_today("000001")    # 当日分时
# fin   = dl.download_company_finance("000001")      # F10 主要财务指标(long 格式)
# cap   = dl.download_finance_capital("000001")      # 股本结构快照(单行)

# %% ======================= 2. 指标计算：indicators.compute_all =======================
from scripts.data_pipeline.indicators import compute_all
import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_parquet('../data/daily/ts_code=601088.SH/data.parquet').sort_values('trade_date').reset_index(drop=True)
ind = compute_all(df, timeframe="daily", shares=1e9)  # shares 可选,用于换手率
print(ind)

# %% ======================= 3. 选股：screener =======================
from scripts.data_pipeline.screener.run_screener import screen
from scripts.data_pipeline.screener.conditions import golden_cross, rsi_oversold, kdj_golden_cross, near_boll_lower

# 只找最新交易日 且 符合条件的股票
result = screen(
    ["000001"],
    [golden_cross, rsi_oversold, kdj_golden_cross, near_boll_lower],
    data_root="data",
    max_bars=200,          # 每个周期最多取的 K 线根数，特殊情况调大
)
print(result)

# %% ======================= 5. 通达信 MCP（实时概念/资金/涨停数据） =======================
from scripts.tdx_mcp import TdxMcpClient

client = TdxMcpClient()                      # 读环境变量 TDX_API_KEY
result = client.query("人工智能概念板块成分股 今日涨跌幅", size=50)
print(result.ok(), result.total)
print(result.to_dicts())                     # list[dict]，字段名 → 值

# 自动翻页（合并多页，最多 max_pages 页）
result_all = client.query_all("DeepSeek概念板块成分股", page_size=50, max_pages=20)

# %% ======================= 可视化：收益率（Plotly） =======================
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.data_pipeline.adjust import forward_adjust
import plotly.io as pio


code = "000001"
daily = pd.read_parquet(f'../data/daily/ts_code={code}.SZ/data.parquet')
xdxr  = pd.read_parquet(f'../data/xdxr/ts_code={code}.SZ/data.parquet')

# 关键：用「前复权价」算收益，否则除权除息会在价格上制造假跳空，收益算出来是错的
adj = forward_adjust(daily, xdxr).sort_values('trade_date').reset_index(drop=True)
adj['ret'] = adj['close'].pct_change()                  # 日收益率
adj['cum'] = (1 + adj['ret'].fillna(0)).cumprod() - 1   # 累计收益率

ACCENT = "#2563eb"   # 单一主色（顺序编码 = 一个色相由浅到深）
MUTED  = "#64748b"

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.04,
    subplot_titles=(f"{code} 累计收益率", "日收益率"),
)

fig.add_trace(
    go.Scatter(x=adj['datetime'], y=adj['cum'], name="累计收益",
               line=dict(color=ACCENT, width=2),
               fill='tozeroy', fillcolor="rgba(37,99,235,0.12)",
               hovertemplate="%{x|%Y-%m-%d}<br>累计收益 %{y:.2%}<extra></extra>"),
    row=1, col=1,
)
fig.add_trace(
    go.Bar(x=adj['datetime'], y=adj['ret'], name="日收益",
           marker_color=MUTED, marker_line_width=0,
           hovertemplate="%{x|%Y-%m-%d}<br>日收益 %{y:.2%}<extra></extra>"),
    row=2, col=1,
)

fig.update_layout(
    template="plotly_white",   # 深色换 "plotly_dark"
    height=640, showlegend=False, hovermode="x",
    margin=dict(l=8, r=8, t=40, b=8),
)
fig.update_yaxes(tickformat=".0%", row=1, col=1)
fig.update_yaxes(tickformat=".1%", row=2, col=1)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="rgba(100,116,139,0.15)",
                 zeroline=True, zerolinecolor="rgba(100,116,139,0.35)")

# 浏览器打开图表
pio.renderers.default = "browser"
fig.show()

# 顺带：K 线（A股习惯：红涨绿跌）
fig_k = go.Figure(go.Candlestick(
    x=adj['datetime'], open=adj['open'], high=adj['high'],
    low=adj['low'], close=adj['close'],
    increasing_line_color="#d92d20", decreasing_line_color="#0ca678",
))
fig_k.update_layout(template="plotly_white", height=520, xaxis_rangeslider_visible=False)
fig_k.show()

# 导出（可选）
# fig.write_html("returns.html")   # 交互式，可直接发人
# fig.write_image("returns.png")   # 静态图（需 pip install kaleido）


# %%
