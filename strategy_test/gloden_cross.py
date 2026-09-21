from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.data_pipeline.strategy import (
    attribute_history,
    g,
    order_target,
    order_value,
    run_daily,
    run_strategy,
    set_benchmark,
    set_option,
)

def initialize(context):
    set_benchmark('000300.SH')           # 聚宽 000300.XSHG 的本地等价
    set_option('use_real_price', True)   # True=不复权（忠于原策略）
    g.security = '000001.SZ'             # 平安银行（原 000001.XSHE）
    # 聚宽的 handle_data 是每根 bar 自动调用；本地用 run_daily('open') 逐日等价
    run_daily(handle_data, time='open')

def handle_data(context):
    # 取最近 10 根日线收盘价（截止前一交易日，无未来函数）
    prices = attribute_history(g.security, 10, '1d', ['close'])
    if len(prices) < 10:
        return
    close = prices['close']
    ma5 = close.iloc[-5:].mean()   # data[s].mavg(5) 的等价
    ma10 = close.mean()            # data[s].mavg(10)

    cash = context.portfolio.cash  # 聚宽 available_cash 的本地等价
    if ma5 > ma10 and cash > 0:
        order_value(g.security, cash)
    elif ma5 < ma10:
        order_target(g.security, 0)


if __name__ == '__main__':
    run_strategy(
        initialize, security='000001.SZ',
        start='2015-01-01', fill_price='open',
        initial_capital=1_000_000.0,
    )
