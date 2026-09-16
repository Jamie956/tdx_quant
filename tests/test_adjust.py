from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.data_pipeline.adjust import forward_adjust


def _daily(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        'trade_date': [f'2023010{i}' for i in range(2, 2 + len(closes))],
        'open': closes, 'high': closes, 'low': closes, 'close': closes,
        'vol': [100] * len(closes),
    })


def test_forward_adjust_flattens_cash_dividend_drop() -> None:
    daily = _daily([10.0, 9.0, 9.0])
    xdxr = pd.DataFrame({
        'trade_date': ['20230103'], 'category': [1],
        'fenhong': [10.0], 'songzhuangu': [0.0], 'peigu': [0.0], 'peigujia': [0.0],
    })
    adj = forward_adjust(daily, xdxr)
    # 派息 1 元/股，除权日从 10 跌到 9；前复权后应抹平为 [9, 9, 9]，且最新价不变。
    np.testing.assert_allclose(adj['close'].to_numpy(), [9.0, 9.0, 9.0])


def test_forward_adjust_flattens_bonus_split() -> None:
    daily = _daily([20.0, 10.0, 10.0])
    xdxr = pd.DataFrame({
        'trade_date': ['20230103'], 'category': [1],
        'fenhong': [0.0], 'songzhuangu': [10.0], 'peigu': [0.0], 'peigujia': [0.0],
    })
    adj = forward_adjust(daily, xdxr)
    # 每10股送转10股（1:1 拆股），前复权后历史价减半 → 抹平为 [10, 10, 10]。
    np.testing.assert_allclose(adj['close'].to_numpy(), [10.0, 10.0, 10.0])


def test_forward_adjust_no_events_is_unchanged() -> None:
    daily = _daily([10.0, 11.0, 12.0])
    xdxr = pd.DataFrame(
        columns=['trade_date', 'category', 'fenhong', 'songzhuangu', 'peigu', 'peigujia'],
    )
    adj = forward_adjust(daily, xdxr)
    np.testing.assert_allclose(adj['close'].to_numpy(), [10.0, 11.0, 12.0])
