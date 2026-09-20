from __future__ import annotations

import numpy as np
import pandas as pd

from examples.rsrs_timing import initialize
from scripts.data_pipeline.strategy import run_strategy


def _index_frame(n: int = 1200) -> pd.DataFrame:
    # low 有意义的 18 日方差 + 与 low 频率不同的 gap 变化，使 RSRS 斜率在 1 附近
    # 双向摆动（既触发买入也触发卖出），从而产生可撮合的 round trip。
    rng = np.random.default_rng(42)
    i = np.arange(n, dtype=float)
    low = 1000.0 + 30.0 * np.sin(i / 30.0) + 0.05 * i
    high = low + 15.0 + 10.0 * np.sin(i / 13.0 + 1.0) + rng.normal(0.0, 1.0, n)
    close = (high + low) / 2.0
    dates = pd.bdate_range('2006-01-05', periods=n)
    return pd.DataFrame({
        'open': close - 1.0,
        'high': high,
        'low': low,
        'close': close,
        'vol': np.full(n, 1e6),
        'amount': close * 1e6,
        'trade_date': dates.strftime('%Y%m%d'),
    })


def test_rsrs_example_runs_end_to_end() -> None:
    df = _index_frame(1200)
    start = df['trade_date'].iloc[1000]  # 前 1000 根用于预计算，后 200 根回测

    result = run_strategy(
        initialize, security='000300.SH', frames={'000300.SH': df},
        start=start, fill_price='open', initial_capital=1_000_000.0,
    )

    assert isinstance(result.equity, pd.Series)
    assert len(result.equity) == 200
    assert len(result.benchmark) == 200
    assert list(result.trades.columns) == [
        'entry_date', 'exit_date', 'entry_price', 'exit_price', 'return_pct', 'days_held',
    ]
    for key in ('n_trades', 'total_return', 'annualized_return', 'benchmark_return',
                'max_drawdown', 'sharpe', 'win_rate', 'avg_trade_return'):
        assert key in result.metrics
    # 有历史预计算 + 足够长的回测区间，策略应至少触发过买入信号并持有过仓位
    # （净值偏离初始资金；买卖撮合逻辑已在 test_strategy_engine 单独覆盖）
    assert result.equity.nunique() > 1
