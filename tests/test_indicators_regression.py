from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.data_pipeline.indicators.regression import calc_rsrs


def _ohlc_df(n: int = 30) -> pd.DataFrame:
    rng = np.arange(n, dtype=float)
    return pd.DataFrame({
        'high': 3.0 + 2.0 * rng + np.sin(rng),
        'low': 1.0 + rng + 0.1 * np.cos(rng),
    })


def test_rsrs_columns() -> None:
    out = calc_rsrs(_ohlc_df())
    assert list(out.columns) == ['RSRS_BETA', 'RSRS_R2', 'RSRS', 'RSRS_RIGHTDEV']


def test_rsrs_perfect_linear() -> None:
    low = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    df = pd.DataFrame({'high': 1.0 + 2.0 * low, 'low': low})
    out = calc_rsrs(df, n=3, m=3)
    # high == 1 + 2*low -> slope 2, R2 == 1 for every valid window
    assert out['RSRS_BETA'].iloc[2] == pytest.approx(2.0)
    assert out['RSRS_R2'].iloc[2] == pytest.approx(1.0)


def test_rsrs_beta_matches_numpy_ols() -> None:
    df = _ohlc_df(20)
    n = 5
    out = calc_rsrs(df, n=n, m=10)
    expected = []
    for i in range(len(df)):
        if i < n - 1:
            expected.append(np.nan)
        else:
            low = df['low'].iloc[i - n + 1:i + 1].to_numpy()
            high = df['high'].iloc[i - n + 1:i + 1].to_numpy()
            expected.append(np.polyfit(low, high, 1)[0])  # slope of high ~ low
    np.testing.assert_allclose(out['RSRS_BETA'].to_numpy(), np.array(expected), equal_nan=True)


def test_rsrs_warmup_nan() -> None:
    df = _ohlc_df(20)
    out = calc_rsrs(df, n=5, m=10)
    assert out['RSRS_BETA'].iloc[:4].isna().all()
    low = df['low'].iloc[:5].to_numpy()
    high = df['high'].iloc[:5].to_numpy()
    assert out['RSRS_BETA'].iloc[4] == pytest.approx(np.polyfit(low, high, 1)[0])


def test_rsrs_r2_is_corr_squared() -> None:
    df = _ohlc_df(20)
    out = calc_rsrs(df, n=5)
    corr = df['high'].rolling(5).corr(df['low'])
    np.testing.assert_allclose(out['RSRS_R2'].to_numpy(), (corr ** 2).to_numpy(), equal_nan=True)


def test_rsrs_zscore_population_std() -> None:
    df = _ohlc_df(40)
    out = calc_rsrs(df, n=3, m=8)
    beta = out['RSRS_BETA']
    mu = beta.rolling(8).mean()
    sigma = beta.rolling(8).std(ddof=0)
    expected = (beta - mu) / sigma
    np.testing.assert_allclose(out['RSRS'].to_numpy(), expected.to_numpy(), equal_nan=True)


def test_rsrs_rightdev_formula() -> None:
    df = _ohlc_df(40)
    out = calc_rsrs(df, n=3, m=8)
    expected = out['RSRS'] * out['RSRS_BETA'] * out['RSRS_R2']
    np.testing.assert_allclose(out['RSRS_RIGHTDEV'].to_numpy(), expected.to_numpy(), equal_nan=True)
