from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.data_pipeline.signals import trend_snapshot


def _frame(close: np.ndarray) -> pd.DataFrame:
    """Synthetic time-ascending OHLCV frame with a daily ``trade_date`` column."""
    n = len(close)
    vol = np.full(n, 1e6)
    return pd.DataFrame({
        'open': close,
        'high': close + 0.1,
        'low': close - 0.1,
        'close': close,
        'vol': vol,
        'amount': close * vol,
        'trade_date': pd.date_range('2023-01-02', periods=n, freq='D').strftime('%Y%m%d'),
    })


def test_trend_snapshot_uptrend() -> None:
    up = np.linspace(10.0, 30.0, 120)
    snap = trend_snapshot(_frame(up))
    assert snap['trend'].startswith('上涨')


def test_trend_snapshot_downtrend() -> None:
    down = np.linspace(30.0, 10.0, 120)
    snap = trend_snapshot(_frame(down))
    assert snap['trend'].startswith('下跌')
