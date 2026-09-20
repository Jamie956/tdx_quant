from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.data_pipeline.strategy.engine import DataLoader, normalize_security


def _frame(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range('2023-01-02', periods=n, freq='D')
    close = np.arange(10.0, 10.0 + n)
    return pd.DataFrame({
        'open': close - 0.1,
        'high': close + 0.2,
        'low': close - 0.2,
        'close': close,
        'vol': np.full(n, 1e6),
        'amount': close * 1e6,
        'trade_date': dates.strftime('%Y%m%d'),
    })


def _loader() -> DataLoader:
    loader = DataLoader(frames={'000001.SZ': _frame()})
    loader.load('000001.SZ')
    return loader


def test_normalize_security() -> None:
    assert normalize_security('000300.XSHG') == '000300.SH'
    assert normalize_security('000001.XSHE') == '000001.SZ'
    assert normalize_security('000300.SH') == '000300.SH'
    assert normalize_security('600000') == '600000.SH'


def test_get_price_slice_range() -> None:
    loader = _loader()
    out = loader.get_price('000001.SZ', start='2023-01-03', end='2023-01-05',
                           fields=['close'], asof='2023-01-09')
    assert list(out.index) == ['20230103', '20230104', '20230105']
    assert list(out.columns) == ['close']


def test_attribute_history_ends_at_asof() -> None:
    loader = _loader()
    out = loader.attribute_history('000001.SZ', 3, fields=['close'], asof='2023-01-06')
    assert list(out.index) == ['20230104', '20230105', '20230106']


def test_attribute_history_no_lookahead_on_asof_none() -> None:
    loader = _loader()
    out = loader.attribute_history('000001.SZ', 5, fields=['close'], asof=None)
    assert out.empty  # 无 prestart 时不得看到任何 bar


def test_get_price_caps_at_asof() -> None:
    loader = _loader()
    # 显式 end 晚于 asof：应被 asof 截断
    out = loader.get_price('000001.SZ', start='2023-01-02', end='2023-01-09',
                           fields=['close'], asof='2023-01-05')
    assert list(out.index) == ['20230102', '20230103', '20230104', '20230105']
