from __future__ import annotations

import pandas as pd

# 通达信 xdxr 里「除权除息」事件的分类码。
EX_DIVIDEND_CATEGORY = 1


def forward_adjust(daily: pd.DataFrame, xdxr: pd.DataFrame) -> pd.DataFrame:
    """Return ``daily`` with OHLC prices 前复权-adjusted for corporate actions.

    前复权以最新价为基准：把除权除息日之前的历史价格按比例缩放，使价格序列连续
    （除权当天不再有跳空缺口），且最新价保持不变。分红/送转/配股都按通达信惯例以
    「每 10 股」为单位（``fenhong`` / ``songzhuangu`` / ``peigu`` 除以 10）。

    ``daily`` 与 ``xdxr`` 均需含 ``trade_date``（YYYYMMDD）。返回的是副本，原
    DataFrame 不被修改；只调整 open/high/low/close，vol/amount 原样保留。
    """
    daily = daily.sort_values('trade_date').reset_index(drop=True).copy()
    events = xdxr[xdxr['category'] == EX_DIVIDEND_CATEGORY].copy()
    if events.empty:
        return daily

    daily['_key'] = pd.to_datetime(daily['trade_date'].astype(str), format='%Y%m%d')
    events = events.sort_values('trade_date').reset_index(drop=True)
    events['_key'] = pd.to_datetime(events['trade_date'].astype(str), format='%Y%m%d')

    cash = events['fenhong'].fillna(0.0) / 10.0        # 每股现金红利
    bonus = events['songzhuangu'].fillna(0.0) / 10.0   # 每股送转股
    rights = events['peigu'].fillna(0.0) / 10.0        # 每股配股
    rights_price = events['peigujia'].fillna(0.0)      # 配股价（元/股）

    # 除权参考价 = (前收盘 - 每股红利 + 配股价×配股比例) / (1 + 送转比例 + 配股比例)
    pre_close = pd.merge_asof(
        events[['_key']],
        daily[['_key', 'close']],
        on='_key', direction='backward', allow_exact_matches=False,
    )['close']

    events['factor'] = (
        (pre_close - cash + rights_price * rights) / (1.0 + bonus + rights)
    ) / pre_close

    # 丢弃无法锚定前收盘的事件（例如早于第一条 K 线）：它对收益无影响（只是整体缩放）。
    events = events[events['factor'].notna()].reset_index(drop=True)

    # adj[t] = 所有 ex_date > t 的事件因子的连乘 = total / (ex_date <= t 的连乘)
    total_prod = events['factor'].prod()
    events['cum_prod'] = events['factor'].cumprod()

    merged = pd.merge_asof(
        daily,
        events[['_key', 'cum_prod']],
        on='_key', direction='backward',
    )
    adj = (total_prod / merged['cum_prod'].fillna(1.0)).to_numpy()

    for col in ('open', 'high', 'low', 'close'):
        daily[col] = daily[col].to_numpy() * adj

    return daily.drop(columns=['_key'])
