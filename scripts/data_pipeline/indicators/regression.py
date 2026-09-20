from __future__ import annotations

import pandas as pd


def calc_rsrs(
    df: pd.DataFrame,
    n: int = 18,
    m: int = 1100,
) -> pd.DataFrame:
    """RSRS (resistance/support relative strength) timing indicator.

    Ported from the JoinQuant strategy in ``tests/rsrs.py``: regress ``high`` on
    ``low`` over the last ``n`` bars (OLS with intercept), take the slope (beta),
    standardize the beta series over the last ``m`` bars into a z-score, and
    finally right-deviation-weight it by ``beta * R2`` — the actual trade signal.

    Pure pandas: OLS slope == Cov(low, high) / Var(low), R2 == corr(high, low)^2.
    Warmup rows (fewer than ``n`` or ``m`` observations) are NaN.
    """
    highs = df['high']
    lows = df['low']

    # rolling cov/var both use ddof=1, which cancels in the ratio -> exact OLS slope
    beta = highs.rolling(n).cov(lows) / lows.rolling(n).var()
    r2 = highs.rolling(n).corr(lows) ** 2

    # original uses np.std (population std, ddof=0) for the standardization
    mu = beta.rolling(m).mean()
    sigma = beta.rolling(m).std(ddof=0)
    rsrs = (beta - mu) / sigma
    rightdev = rsrs * beta * r2

    return pd.DataFrame({
        'RSRS_BETA': beta,
        'RSRS_R2': r2,
        'RSRS': rsrs,
        'RSRS_RIGHTDEV': rightdev,
    })
