from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.data_pipeline.indicators import compute_all
from scripts.data_pipeline.screener.conditions import (
    death_cross_series,
    golden_cross_series,
)

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Result of :func:`run_backtest`."""

    equity: pd.Series        # strategy equity curve (indexed by trade_date)
    benchmark: pd.Series     # buy-and-hold equity curve
    trades: pd.DataFrame     # one row per completed round trip
    metrics: dict

    def summary(self) -> str:
        m = self.metrics
        return '\n'.join([
            f"交易次数        {m['n_trades']}",
            f"策略总收益      {m['total_return']:.2%}",
            f"年化收益        {m['annualized_return']:.2%}",
            f"基准(买入持有)  {m['benchmark_return']:.2%}",
            f"最大回撤        {m['max_drawdown']:.2%}",
            f"夏普比率        {m['sharpe']:.2f}",
            f"胜率            {m['win_rate']:.2%}",
            f"单笔平均收益    {m['avg_trade_return']:.2%}",
        ])


def run_backtest(df: pd.DataFrame, *, initial_capital: float = 100_000.0) -> BacktestResult:
    """Backtest a MACD golden-cross / death-cross strategy on one stock.

    Buys at the close of a golden-cross bar and sells at the close of the next
    death-cross bar (close-to-close, no commission or slippage). ``df`` must be
    a time-ascending raw OHLCV frame.
    """
    ind = compute_all(df, timeframe='daily').reset_index(drop=True)
    # 日收市价
    close = ind['close']
    # pct_change()：计算百分比变化
    # fillna(0.0): NAN -> 0.0
    # ret 每日涨跌幅
    ret = close.pct_change().fillna(0.0)

    # Long while DIF > DEA, applied one bar later so the signal cannot peek at
    # the return it would earn on that same bar (no lookahead).
    # astype(int)：true -> 1, false -> 0
    # position 1 持有 0 空仓
    # 在金叉上的时候，值为 1，表示这个时间持仓
    position = (ind['DIF'] > ind['DEA']).astype(int).shift(1).fillna(0).astype(int)
    # 持仓时有涨跌幅，不持仓时值为 0
    strat_ret = position * ret

    # cumprod()：把每天的收益连乘，算出你的总资产变化
    # equity 触发条件时的持仓累计收益，净值曲线，每一天的总资产
    equity = initial_capital * (1.0 + strat_ret).cumprod()
    # 全时段累计收益
    benchmark = initial_capital * (1.0 + ret).cumprod()

    trades = _extract_trades(ind)
    metrics = _metrics(equity, benchmark, strat_ret, trades, initial_capital)

    if 'trade_date' in ind.columns:
        equity.index = ind['trade_date']
        benchmark.index = ind['trade_date']

    return BacktestResult(equity=equity, benchmark=benchmark, trades=trades, metrics=metrics)


def _extract_trades(ind: pd.DataFrame) -> pd.DataFrame:
    """Pair each golden cross with the next death cross into a round trip.

    Per-trade return is ``close[sell] / close[buy] - 1`` — the same close-to-close
    convention the equity curve uses (the one-bar execution lag telescopes away).
    """
    # 日收市价
    close = ind['close']
    # to_numpy()：把列转成一个数组
    # np.where()：数组元素为 True 的索引组成一个数组
    # buy_idx：买入的索引
    buy_idx = np.where(golden_cross_series(ind).to_numpy())[0]
    # sell_idx：卖出的索引
    sell_idx = np.where(death_cross_series(ind).to_numpy())[0]

    rows: list[dict] = []
    j = 0
    for g in buy_idx:
        # j：卖了几多次
        while j < len(sell_idx) and sell_idx[j] <= g:
            j += 1
        if j >= len(sell_idx):
            break  # no matching sell -> open position at end (not a completed trade)
        # d 卖出当天的索引
        d = int(sell_idx[j])
        rows.append({
            # 买入日
            'entry_date': ind['trade_date'].iloc[g] if 'trade_date' in ind.columns else g,
            # 卖出日
            'exit_date': ind['trade_date'].iloc[d] if 'trade_date' in ind.columns else d,
            # 买入价
            'entry_price': float(close.iloc[g]),
            # 卖出价
            'exit_price': float(close.iloc[d]),
            # 单笔收益 = 卖出价/买入价 - 1
            'return_pct': float(close.iloc[d] / close.iloc[g] - 1.0),
            'days_held': d - g,
        })
        j += 1

    columns = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'return_pct', 'days_held']
    return pd.DataFrame(rows, columns=columns)


def _metrics(equity, benchmark, strat_ret, trades, initial_capital) -> dict:
    n = len(equity)
    # 使用策略赚了多少
    total = equity.iloc[-1] / initial_capital - 1.0
    # 不使用策略赚了多少
    bench_total = benchmark.iloc[-1] / initial_capital - 1.0
    # 1.0 + total：总资产倍数
    # **：次方
    # TRADING_DAYS / n = 252 / 126 = 2：126 天是半年；一年等于2 个半年
    # - 1.0：扣掉本金
    annualized = (1.0 + total) ** (TRADING_DAYS / n) - 1.0 if n > 0 else 0.0
    # max_dd 最大回撤
    # equity.cummax() 累计最大值：记录到当天为止，曾经到达过的最高净值
    # equity / equity.cummax()：当前净值 ÷ 历史最高净值
    max_dd = (equity / equity.cummax() - 1.0).min()
    # 收益率标准差
    std = strat_ret.std()
    # strat_ret.mean()：日平均收益率
    # np.sqrt(TRADING_DAYS)：根号 252，波动率按根号时间放大
    sharpe = float(strat_ret.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float('nan')

    rets = trades['return_pct'] if len(trades) else pd.Series([], dtype=float)
    # (rets > 0).mean()：对 0/1 序列求平均值
    win_rate = float((rets > 0).mean()) if len(rets) else float('nan')

    return {
        'n_trades': int(len(trades)),
        'total_return': float(total),
        'annualized_return': float(annualized),
        'benchmark_return': float(bench_total),
        'max_drawdown': float(max_dd),
        'sharpe': sharpe,
        'win_rate': win_rate,
        'avg_trade_return': float(rets.mean()) if len(rets) else float('nan'),
    }
