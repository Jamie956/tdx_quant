# ============================== 1. 下载：tdx_client.TdxDownloader
# from pathlib import Path
# from scripts.data_pipeline.tdx_client import TdxDownloader

# dl = TdxDownloader(Path("data"))

# daily   = dl.download_daily("000001")          # 日K全历史(自动翻页),落盘并返回
# minute  = dl.download_minute("000001", freq=5) # 5分钟线,带 trade_time 列
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

# ============================== 2. 指标计算：indicators.compute_all
# from scripts.data_pipeline.indicators import compute_all
# import numpy as np
# import pandas as pd

# pd.set_option('display.max_columns', None)

# def _make_daily_df(rows: int = 120) -> pd.DataFrame:
#     rng = np.random.default_rng(seed=42)
#     base = 10.0 + np.arange(rows, dtype=float) * 0.1
#     wiggle = rng.normal(scale=0.05, size=rows)
#     close = base + wiggle
#     high = close + rng.uniform(0.05, 0.3, size=rows)
#     low = close - rng.uniform(0.05, 0.3, size=rows)
#     open_ = close + rng.normal(scale=0.1, size=rows)
#     vol = rng.uniform(1e5, 1e7, size=rows)
#     amount = vol * close
#     return pd.DataFrame({
#         'open': open_,
#         'high': high,
#         'low': low,
#         'close': close,
#         'vol': vol,
#         'amount': amount,
#     })

# ind = compute_all(_make_daily_df(), timeframe="daily", shares=1e9)  # shares 可选,用于换手率
# # ind 在副本上附加全部指标列
# print(ind)

# ============================== 3. 选股：screener
# from scripts.data_pipeline.screener.run_screener import screen
# from scripts.data_pipeline.screener.conditions import golden_cross, rsi_oversold

# result = screen(
#     ["000001", "600000", "000002"],
#     [golden_cross, rsi_oversold],
#     data_root="data",
#     max_bars=200,          # 每个周期最多取的 K 线根数，特殊情况调大
# )
# # 列: ts_code, timeframe, close, hit_count, matched, latest_trade_date
# # 每个 (股票, 周期) 一行；按 hit_count 降序
# print(result)


# ============================== 5. 通达信 MCP（实时概念/资金/涨停数据）
# from scripts.tdx_mcp import TdxMcpClient

# client = TdxMcpClient()                      # 读环境变量 TDX_API_KEY
# result = client.query("人工智能概念板块成分股 今日涨跌幅", size=50)
# print(result.ok(), result.total)
# print(result.to_dicts())                     # list[dict]，字段名 → 值

# 自动翻页（合并多页，最多 max_pages 页）
# result_all = client.query_all("DeepSeek概念板块成分股", page_size=50, max_pages=20)

