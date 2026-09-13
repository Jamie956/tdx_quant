from __future__ import annotations

import pandas as pd

from scripts.data_pipeline.indicators import compute_all


def trend_snapshot(df: pd.DataFrame, *, slope_bars: int = 5) -> pd.Series:
    """Return a multi-field verdict on the latest bar's trend.

    Medium-term trend uses ``close`` vs ``MA20`` plus the ``MA20`` slope over
    ``slope_bars``; short-term uses ``DIF`` vs ``DEA``. ``df`` must be a
    time-ascending raw OHLCV frame (indicators are computed here).
    """
    ind = compute_all(df, timeframe='daily')
    last = ind.iloc[-1]
    close, ma20, ma60 = last['close'], last['MA20'], last['MA60']
    # ma20_slope 最新一根 MA20 减去 5 根之前的 MA20，衡量「MA20 这 5 根 K 线涨了多少
    ma20_slope = (
        last['MA20'] - ind['MA20'].iloc[-1 - slope_bars]
        if len(ind) > slope_bars
        else float('nan')
    )
    dif, dea = last['DIF'], last['DEA']

    mid_bull = close > ma20 and ma20_slope > 0
    mid_bear = close < ma20 and ma20_slope < 0
    short_bull = dif > dea

    if mid_bull and short_bull:
        trend = '上涨（中期多头 + 短期多头）'
    elif mid_bull:
        trend = '上涨（中期多头，短期回调）'
    elif mid_bear and not short_bull:
        trend = '下跌（中期空头 + 短期空头）'
    elif mid_bear:
        trend = '下跌（中期空头，短期反弹）'
    else:
        trend = '震荡/不明'

    return pd.Series({
        'trade_date': last['trade_date'] if 'trade_date' in ind.columns else None,
        'close': close,
        'ma20': round(ma20, 3),
        'ma20_slope': round(ma20_slope, 3),
        'ma60': round(ma60, 3),
        'dif': round(dif, 4),
        'dea': round(dea, 4),
        'trend': trend,
    })
