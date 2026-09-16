# %% ======================= init =======================
from pathlib import Path
os.chdir(Path(__vsc_ipynb_file__).parent)

from dotenv import load_dotenv
# 加载项目根目录的.env
load_dotenv() 

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

