from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ''}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np

from scripts.data_pipeline.strategy import (
    OrderCost,
    attribute_history,
    g,
    get_price,
    log,
    order_target,
    order_target_value,
    run_daily,
    run_strategy,
    set_benchmark,
    set_option,
    set_order_cost,
)

def _ols(high, low):
    """一元 OLS：high ~ low 的斜率 beta 与 R2（等价 statsmodels.OLS，纯 numpy）。"""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    # mean(): 平均值
    hc = high - high.mean()
    lc = low - low.mean()
    denom = float((lc ** 2).sum())
    if denom == 0:
        return np.nan, np.nan
    # 计算斜率
    beta = float((hc * lc).sum() / denom)
    # `np.corrcoef(high, low)`：计算`high`和`low`两个序列的**皮尔逊相关系数矩阵**
        #           0(high)	        1(low)
        # 0 (high)	1	            r(high,low)
        # 1 (low)	r(high,low)	    1
    # `[0,1]`：取出矩阵中 `high ↔ low` 的相关系数 r
    # R²：衡量高低价点贴不贴近这条回归线，r2 越高，高低点越贴合回归线，可信度越高
    r2 = float(np.corrcoef(high, low)[0, 1] ** 2)
    return beta, r2


def set_parameter(context):
    g.N = 18
    g.M = 1100
    g.init = True
    g.security = '000300.SH'
    g.buy = 0.7
    g.sell = -0.7
    g.ans = []
    g.ans_rightdev = []

    # 预计算回测开始前的历史 RSRS 斜率（get_price 取到 context.previous_date）
    prices = get_price(g.security, '2005-01-05', context.previous_date, '1d', ['high', 'low'])
    highs = prices['high']
    lows = prices['low']
    for i in range(g.N - 1, len(highs)):
        beta, r2 = _ols(highs.iloc[i - g.N + 1:i + 1], lows.iloc[i - g.N + 1:i + 1])
        g.ans.append(beta)
        g.ans_rightdev.append(r2)

    if not g.ans:
        log.warning(
            '预计算为空：start=%s 早于数据首日或无足够历史，RSRS 将不产生任何信号',
            context.previous_date,
        )
    elif len(g.ans) < g.M:
        log.warning(
            '预计算 %d 根斜率 < M=%d，标准化窗口未满，早期信号可能失真（建议 start 设在 2010 之后）',
            len(g.ans), g.M,
        )


def before_market_open(context):
    g.days = getattr(g, 'days', 0) + 1


def market_open(context):
    if not g.ans:
        return

    beta = 0.0
    r2 = 0.0
    if g.init:
        g.init = False
    else:
        # 用最近 N 根（截止前一交易日）更新斜率，逐日追加
        prices = attribute_history(context.security, g.N, '1d', ['high', 'low'])
        beta, r2 = _ols(prices['high'], prices['low'])
        g.ans.append(beta)
        g.ans_rightdev.append(r2)

    # 取最近 M 个历史斜率，计算**当前斜率在历史分布中的标准化 Z 值**，还叠加了 beta 和 r² 做加权修正
    # 标准化 + 右偏加权（np.std 默认总体标准差，对齐聚宽）
    # `[-g.M:]` 切片：取**最后 M 个元素**（python 切片，负号代表从末尾往前数）
    section = g.ans[-g.M:]
    # \(\mu\) = 最近 M 个历史斜率的**平均值**
    mu = float(np.mean(section))
    # \(\sigma\) = 最近 M 个历史斜率的**标准差**，衡量历史斜率波动大小
    sigma = float(np.std(section))
    # `zscore`：当前斜率，**高于历史均值 2 个标准差**
    # `section[-1]`：**最新这一根 K 线的斜率 beta**（当前值）
    zscore = (section[-1] - mu) / sigma if sigma > 0 else 0.0
    zscore_rightdev = zscore * beta * r2

    if zscore_rightdev > g.buy:
        log.info('%s 全仓 (rsrs=%.3f)', context.current_dt.strftime('%Y-%m-%d'), zscore_rightdev)
        order_target_value(context.security, context.portfolio.total_value)
    elif zscore_rightdev < g.sell and len(context.portfolio.positions) > 0:
        log.info('%s 空仓 (rsrs=%.3f)', context.current_dt.strftime('%Y-%m-%d'), zscore_rightdev)
        order_target(context.security, 0)


def initialize(context):
    set_option('use_real_price', True)
    set_benchmark('000300.SH')
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003,
                             close_commission=0.0003, min_commission=5))
    set_parameter(context)
    run_daily(before_market_open, time='before_open', reference_security='000300.SH')
    run_daily(market_open, time='open', reference_security='000300.SH')

def main() -> None:
    result = run_strategy(
        initialize, security='000300.SH',
        start='2015-01-01', fill_price='open'
    )


if __name__ == '__main__':
    main()
