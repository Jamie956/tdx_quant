from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.data_pipeline.strategy import (
    OrderCost,
    attribute_history,
    order_target,
    run_daily,
    run_strategy,
    set_order_cost,
)


def _frame(n: int = 60, close=None) -> pd.DataFrame:
    if close is None:
        close = np.linspace(10.0, 20.0, n)
    else:
        close = np.asarray(close, dtype=float)
    dates = pd.date_range('2023-01-02', periods=n, freq='D')
    return pd.DataFrame({
        'open': close - 0.1,
        'high': close + 0.2,
        'low': close - 0.2,
        'close': close,
        'vol': np.full(n, 1e6),
        'amount': close * 1e6,
        'trade_date': dates.strftime('%Y%m%d'),
    })


def test_single_round_trip() -> None:
    df = _frame(60)
    n = len(df)
    state = {'n': 0}

    def initialize(ctx):
        set_order_cost(OrderCost(open_commission=0, close_commission=0, close_tax=0, min_commission=0))
        run_daily(on_open, time='open')

    def on_open(ctx):
        state['n'] += 1
        if state['n'] == 1:
            order_target(ctx.security, 1000)   # 满仓 1000 股
        elif state['n'] == n:
            order_target(ctx.security, 0)      # 清仓

    result = run_strategy(initialize, security='000001.SZ', frames={'000001.SZ': df},
                          fill_price='open', initial_capital=100_000.0)
    assert len(result.equity) == n
    assert len(result.trades) == 1
    assert result.trades.iloc[0]['entry_price'] == pytest.approx(df['open'].iloc[0])
    assert result.trades.iloc[0]['exit_price'] == pytest.approx(df['open'].iloc[-1])
    for key in ('n_trades', 'total_return', 'annualized_return', 'benchmark_return',
                'max_drawdown', 'sharpe', 'win_rate', 'avg_trade_return'):
        assert key in result.metrics


def test_fill_price_open_vs_close() -> None:
    df = _frame(60)
    n = len(df)

    def run(price):
        state = {'n': 0}

        def initialize(ctx):
            set_order_cost(OrderCost(0, 0, 0, 0))
            run_daily(on_open, time='open')

        def on_open(ctx):
            state['n'] += 1
            if state['n'] == 1:
                order_target(ctx.security, 1000)
            elif state['n'] == n:
                order_target(ctx.security, 0)

        return run_strategy(initialize, security='000001.SZ', frames={'000001.SZ': df},
                            fill_price=price, initial_capital=100_000.0)

    r_open = run('open')
    r_close = run('close')
    # 成交价约定直接体现在买卖价上：open 用开盘价、close 用收盘价
    assert r_open.trades.iloc[0]['entry_price'] == pytest.approx(df['open'].iloc[0])
    assert r_open.trades.iloc[0]['exit_price'] == pytest.approx(df['open'].iloc[-1])
    assert r_close.trades.iloc[0]['entry_price'] == pytest.approx(df['close'].iloc[0])
    assert r_close.trades.iloc[0]['exit_price'] == pytest.approx(df['close'].iloc[-1])


def test_no_lookahead_attribute_history() -> None:
    df = _frame(30)
    violations = []

    def initialize(ctx):
        run_daily(on_open, time='open')

    def on_open(ctx):
        h = attribute_history(ctx.security, 1, '1d', ['close'])
        if len(h) and ctx.prev_close.get(ctx.security) is not None:
            got = float(h['close'].iloc[-1])
            # 最近 1 根应为前一交易日收盘，绝不该是当日价
            if abs(got - ctx.prev_close[ctx.security]) > 1e-9:
                violations.append(str(ctx.current_dt))

    run_strategy(initialize, security='000001.SZ', frames={'000001.SZ': df})
    assert violations == []


def test_empty_strategy_equity_and_benchmark() -> None:
    close = np.linspace(10.0, 20.0, 30)
    df = _frame(30, close=close)

    def initialize(ctx):
        run_daily(lambda ctx: None, time='open')

    result = run_strategy(initialize, security='000001.SZ', frames={'000001.SZ': df},
                          initial_capital=100_000.0)
    # 空仓：净值恒为 initial_capital
    assert result.equity.iloc[-1] == pytest.approx(100_000.0)
    # 基准为买入持有
    assert result.benchmark.iloc[-1] == pytest.approx(100_000.0 * close[-1] / close[0])
    assert result.metrics['n_trades'] == 0
