from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.data_pipeline.backtest import run_backtest


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


def test_run_backtest_completes_a_round_trip() -> None:
    # rise then fall -> a golden cross followed by a death cross
    close = np.concatenate([np.linspace(10.0, 20.0, 60), np.linspace(20.0, 12.0, 60)])
    result = run_backtest(_frame(close))
    assert result.metrics['n_trades'] >= 1
    assert len(result.trades) == result.metrics['n_trades']


def test_run_backtest_equity_length_and_columns() -> None:
    close = np.linspace(10.0, 20.0, 120)
    result = run_backtest(_frame(close))
    assert len(result.equity) == len(close)
    assert list(result.trades.columns) == [
        'entry_date', 'exit_date', 'entry_price', 'exit_price', 'return_pct', 'days_held',
    ]
    for key in ('n_trades', 'total_return', 'annualized_return', 'benchmark_return',
                'max_drawdown', 'sharpe', 'win_rate', 'avg_trade_return'):
        assert key in result.metrics


def test_run_backtest_benchmark_is_buy_and_hold() -> None:
    close = np.linspace(10.0, 20.0, 120)
    result = run_backtest(_frame(close), initial_capital=50_000.0)
    ret = pd.Series(close).pct_change().fillna(0.0)
    expected = 50_000.0 * (1.0 + ret).cumprod()
    np.testing.assert_allclose(result.benchmark.to_numpy(), expected.to_numpy())
