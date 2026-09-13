from __future__ import annotations

import pandas as pd

from scripts.data_pipeline.indicators.volume import calc_volume_ratio


def _cross_above(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """Boolean Series: True on bars where ``fast`` crosses above ``slow``.

    A bar is a cross iff the previous bar had ``fast <= slow`` and this bar has
    ``fast > slow``. NaN anywhere in the involved values yields False for that
    bar (``>``/``<=`` are False on NaN, so no explicit NaN guard is needed).
    """
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def golden_cross_series(df: pd.DataFrame) -> pd.Series:
    """Boolean Series marking every bar where MACD DIF crosses above DEA.

    Vectorized counterpart to :func:`golden_cross` — use this to locate the
    historical dates of MACD golden crosses, not just the latest bar. Returns
    an all-False Series (same index) when the required columns are missing.
    """
    if 'DIF' not in df.columns or 'DEA' not in df.columns:
        return pd.Series(False, index=df.index)
    return _cross_above(df['DIF'], df['DEA'])


def kdj_golden_cross_series(df: pd.DataFrame) -> pd.Series:
    """Boolean Series marking every bar where KDJ K crosses above D."""
    if 'K' not in df.columns or 'D' not in df.columns:
        return pd.Series(False, index=df.index)
    return _cross_above(df['K'], df['D'])


def golden_cross(df: pd.DataFrame) -> bool:
    """MACD golden cross on the latest bar: DIF crosses ABOVE DEA.

    True iff ``DIF[-1] > DEA[-1]`` and ``DIF[-2] <= DEA[-2]``. Returns False
    when there are fewer than two bars or any involved value is NaN.
    """
    if len(df) < 2:
        return False
    return bool(golden_cross_series(df).iloc[-1])


def kdj_golden_cross(df: pd.DataFrame) -> bool:
    """KDJ golden cross on the latest bar: K crosses ABOVE D."""
    if len(df) < 2:
        return False
    return bool(kdj_golden_cross_series(df).iloc[-1])


def volume_breakout(df: pd.DataFrame, n: int = 5, k: float = 2) -> bool:
    """放量突破: today's n-day volume ratio > k AND close > MA20.

    The n-day volume ratio is computed inline (honouring ``n``) rather than
    read from ``VOL_RATIO`` (which uses the daily-config default of 5).
    """
    if len(df) < 2 or 'MA20' not in df.columns or 'close' not in df.columns:
        return False
    ratio = calc_volume_ratio(df, n).iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    close = df['close'].iloc[-1]
    if pd.isna(ratio) or pd.isna(ma20) or pd.isna(close):
        return False
    return bool(ratio > k and close > ma20)


def rsi_oversold(df: pd.DataFrame, threshold: float = 30) -> bool:
    """Latest RSI6 below ``threshold``."""
    if 'RSI6' not in df.columns or len(df) == 0:
        return False
    rsi = df['RSI6'].iloc[-1]
    if pd.isna(rsi):
        return False
    return bool(rsi < threshold)


def near_boll_lower(df: pd.DataFrame) -> bool:
    """Latest close touches or falls below the BOLL lower band."""
    if 'BOLL_DN' not in df.columns or 'close' not in df.columns or len(df) == 0:
        return False
    close = df['close'].iloc[-1]
    dn = df['BOLL_DN'].iloc[-1]
    if pd.isna(dn) or pd.isna(close):
        return False
    return bool(close <= dn)


def hammer_series(
    df: pd.DataFrame,
    *,
    lower_body_ratio: float = 2.0,
    upper_body_ratio: float = 1.0,
) -> pd.Series:
    """Boolean Series marking hammer candles (锤头线 / 下锤线).

    A hammer has a long lower shadow (>= ``lower_body_ratio`` × the real body)
    and a small upper shadow (<= ``upper_body_ratio`` × the real body), i.e. the
    body sits near the top of the candle — a bullish-reversal signal after a
    decline. Requires OHLC columns; returns an all-False Series otherwise.
    """
    if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
        return pd.Series(False, index=df.index)
    body = (df['close'] - df['open']).abs()
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    return (
        (lower_shadow >= lower_body_ratio * body)
        & (upper_shadow <= upper_body_ratio * body)
        & (lower_shadow > 0)
    )


def hammer(
    df: pd.DataFrame,
    *,
    lower_body_ratio: float = 2.0,
    upper_body_ratio: float = 1.0,
) -> bool:
    """Latest bar is a hammer (锤头线): long lower shadow, small upper shadow."""
    if len(df) == 0:
        return False
    return bool(
        hammer_series(
            df,
            lower_body_ratio=lower_body_ratio,
            upper_body_ratio=upper_body_ratio,
        ).iloc[-1]
    )


def uptrend(df: pd.DataFrame, *, slope_bars: int = 5) -> bool:
    """Latest bar is in a medium-term uptrend: close above MA20 and MA20 rising."""
    if 'close' not in df.columns or 'MA20' not in df.columns or len(df) < slope_bars + 1:
        return False
    close, ma20 = df['close'].iloc[-1], df['MA20'].iloc[-1]
    slope = ma20 - df['MA20'].iloc[-1 - slope_bars]
    return bool(close > ma20 and slope > 0)


def downtrend(df: pd.DataFrame, *, slope_bars: int = 5) -> bool:
    """Latest bar is in a medium-term downtrend: close below MA20 and MA20 falling."""
    if 'close' not in df.columns or 'MA20' not in df.columns or len(df) < slope_bars + 1:
        return False
    close, ma20 = df['close'].iloc[-1], df['MA20'].iloc[-1]
    slope = ma20 - df['MA20'].iloc[-1 - slope_bars]
    return bool(close < ma20 and slope < 0)


# Registry mapping CLI names to the condition callables.
CONDITIONS = {
    'golden_cross': golden_cross,
    'kdj_golden_cross': kdj_golden_cross,
    'volume_breakout': volume_breakout,
    'rsi_oversold': rsi_oversold,
    'near_boll_lower': near_boll_lower,
    'hammer': hammer,
    'uptrend': uptrend,
    'downtrend': downtrend,
}
