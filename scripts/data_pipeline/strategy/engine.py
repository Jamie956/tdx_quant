from __future__ import annotations

import bisect
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from scripts.data_pipeline.adjust import forward_adjust
from scripts.data_pipeline.backtest import BacktestResult, _metrics
from scripts.data_pipeline.fetch_realtime_watchlist import infer_hq_market
from scripts.data_pipeline.strategy import runtime
from scripts.data_pipeline.strategy.context import (
    Context,
    Portfolio,
    Position,
    execute_order,
)

from scripts.data_pipeline.strategy.plot import plot_equity


_SUFFIX_MAP = {'.XSHG': '.SH', '.XSHE': '.SZ'}


def normalize_security(code: str) -> str:
    """归一化证券代码：``000300.XSHG`` / ``000300.SH`` / ``000300`` -> ``000300.SH``。"""
    s = str(code).strip().upper()
    for old, new in _SUFFIX_MAP.items():
        if s.endswith(old):
            return s[: -len(old)] + new
    if s.endswith(('.SH', '.SZ')):
        return s
    if s.isdigit() and len(s) == 6:
        market = infer_hq_market(s)
        return f"{s}.{'SH' if market == 1 else 'SZ'}"
    raise ValueError(f'cannot normalize security code: {code!r}')


def _norm_date(value) -> str | None:
    """把 date/datetime/'YYYY-MM-DD'/'YYYYMMDD' 统一成 'YYYYMMDD'。"""
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime('%Y%m%d')
    return str(value).replace('-', '')


def _to_date(s: str) -> date | None:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8])) if s else None


def _to_dt(s: str) -> datetime | None:
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8])) if s else None


class DataLoader:
    """加载并缓存证券日线（原始 + 前复权），提供 as-of 切片查询。"""

    def __init__(self, *, data_root: str = 'data', frames: dict | None = None) -> None:
        self.data_root = Path(data_root)
        self.frames = frames
        self.raw: dict[str, pd.DataFrame] = {}   # code -> 按 trade_date 索引
        self.adj: dict[str, pd.DataFrame] = {}
        self.dates: dict[str, list] = {}
        self.is_stock: dict[str, bool] = {}

    def load(self, code: str) -> None:
        code = normalize_security(code)
        if code in self.raw:
            return
        if self.frames is not None:
            raw, adj, is_stock = self._from_frames(code)
        else:
            raw, adj, is_stock = self._from_parquet(code)
        self.raw[code] = raw
        self.adj[code] = adj
        self.dates[code] = list(raw.index)
        self.is_stock[code] = is_stock

    def _from_frames(self, code: str):
        if code not in self.frames:
            raise KeyError(f'no frame provided for {code!r}')
        df = self._prepare(self.frames[code])
        return df, df, True  # frames 模式视为个股（佣金/印花税），is_stock=True

    def _from_parquet(self, code: str):
        daily = self.data_root / 'daily' / f'ts_code={code}' / 'data.parquet'
        index = self.data_root / 'index_daily' / f'ts_code={code}' / 'data.parquet'
        if daily.exists():
            raw = self._prepare(pd.read_parquet(daily))
            xdxr_path = self.data_root / 'xdxr' / f'ts_code={code}' / 'data.parquet'
            if xdxr_path.exists():
                adj = self._prepare(forward_adjust(raw.reset_index(), pd.read_parquet(xdxr_path)))
            else:
                adj = raw
            return raw, adj, True
        if index.exists():
            raw = self._prepare(pd.read_parquet(index))
            return raw, raw, False
        raise FileNotFoundError(
            f'no data for {code!r} under {self.data_root}/daily or {self.data_root}/index_daily'
        )

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['trade_date'] = df['trade_date'].astype(str)
        return df.sort_values('trade_date').reset_index(drop=True).set_index('trade_date')

    def _frame(self, code: str, real_price: bool) -> pd.DataFrame:
        return self.raw[code] if real_price else self.adj[code]

    def _asof_pos(self, code: str, asof) -> int:
        key = _norm_date(asof)
        if key is None:
            return -1  # 无可见历史（首日无 prestart 时不得看到任何 bar）
        return bisect.bisect_right(self.dates[code], key) - 1

    def get_price(
        self, code: str, *, start=None, end=None, fields=None, asof=None, real_price: bool = False,
    ) -> pd.DataFrame:
        """返回 ``[start, end]`` 区间（含端点）内、且不晚于 ``asof`` 的 bar。"""
        code = normalize_security(code)
        df = self._frame(code, real_price)
        ds = self.dates[code]
        lo = bisect.bisect_left(ds, _norm_date(start)) if start else 0
        hi = (bisect.bisect_right(ds, _norm_date(end)) - 1) if end else len(ds) - 1
        hi = min(hi, self._asof_pos(code, asof))
        if lo > hi:
            out = df.iloc[0:0]
        else:
            out = df.iloc[lo:hi + 1]
        return self._select(out, fields)

    def attribute_history(
        self, code: str, count: int, *, fields=None, asof=None, real_price: bool = False,
    ) -> pd.DataFrame:
        """返回截止 ``asof`` 的最近 ``count`` 根 bar（不含当日）。"""
        code = normalize_security(code)
        df = self._frame(code, real_price)
        pos = self._asof_pos(code, asof)
        if pos < 0:
            out = df.iloc[0:0]
        else:
            out = df.iloc[max(0, pos - count + 1):pos + 1]
        return self._select(out, fields)

    @staticmethod
    def _select(df: pd.DataFrame, fields) -> pd.DataFrame:
        if fields is None:
            cols = [c for c in ('open', 'high', 'low', 'close', 'vol', 'amount',
                                'up_count', 'down_count') if c in df.columns]
        else:
            cols = list(fields)
        return df[cols]

    def price(self, code: str, date_str: str | None, field: str, *, real_price: bool) -> float | None:
        if date_str is None:
            return None
        df = self._frame(normalize_security(code), real_price)
        try:
            return float(df.loc[date_str, field])
        except KeyError:
            return None


def _pair_trades(fills: list, date_pos: dict) -> pd.DataFrame:
    """把逐笔成交按 FIFO 撮合成一次「开→平」round trip。"""
    lots: list[list] = []  # [entry_date, entry_price, remaining_shares, entry_pos]
    rows: list[dict] = []
    for d, side, shares, price in fills:
        pos = date_pos[d]
        if side == 'buy':
            lots.append([d, price, shares, pos])
        else:
            remaining = shares
            while remaining > 0 and lots:
                lot = lots[0]
                take = min(remaining, lot[2])
                rows.append({
                    'entry_date': lot[0],
                    'exit_date': d,
                    'entry_price': lot[1],
                    'exit_price': price,
                    'return_pct': price / lot[1] - 1.0,
                    'days_held': pos - lot[3],
                })
                lot[2] -= take
                remaining -= take
                if lot[2] == 0:
                    lots.pop(0)
    columns = ['entry_date', 'exit_date', 'entry_price', 'exit_price', 'return_pct', 'days_held']
    return pd.DataFrame(rows, columns=columns)


def _benchmark_curve(loader: DataLoader, code: str, loop_dates: list, capital: float) -> pd.Series:
    close = loader.raw[code]['close'].astype(float)
    aligned = close.reindex(loop_dates).ffill().bfill()
    baseline = aligned.iloc[0]
    return pd.Series(capital * aligned / baseline, index=loop_dates, name='benchmark')


def run_strategy(
    initialize,
    *,
    data_root: str = 'data',
    security: str,
    benchmark: str | None = None,
    start=None,
    end=None,
    initial_capital: float = 1_000_000.0,
    adjust: str = 'forward',
    fill_price: str = 'open',
    lot_size: int = 100,
    t_plus_1: bool = True,
    frames: dict | None = None,
) -> BacktestResult:
    """运行一个聚宽式策略并返回 :class:`BacktestResult`。

    ``initialize`` 在回测开始前调用一次；其内部通过 ``run_daily`` 注册的回调按
    before_open → open → after_close 顺序逐日执行。信号只可见 ≤ 前一交易日的数据，
    订单按 ``fill_price``（open/close）成交，无未来函数。
    """
    sec = normalize_security(security)
    if fill_price not in ('open', 'close'):
        raise ValueError(f'fill_price must be "open" or "close", got {fill_price!r}')

    loader = DataLoader(data_root=data_root, frames=frames)
    loader.load(sec)
    loader.load(normalize_security(benchmark) if benchmark else sec)

    runtime._reset()
    runtime._seed_default_options(adjust)

    portfolio = Portfolio(cash=initial_capital, starting_cash=initial_capital,
                          total_value=initial_capital)
    ctx = Context(security=sec, benchmark=normalize_security(benchmark) if benchmark else sec,
                  portfolio=portfolio, data=loader)

    full = list(loader.dates[sec])
    lo = bisect.bisect_left(full, _norm_date(start)) if start else 0
    hi = bisect.bisect_right(full, _norm_date(end)) if end else len(full)
    loop_dates = full[lo:hi]
    if not loop_dates:
        raise ValueError('empty trading calendar for the given start/end')
    prestart = full[lo - 1] if lo > 0 else None
    date_pos = {d: i for i, d in enumerate(loop_dates)}

    ctx.current_dt = _to_dt(prestart)
    ctx.previous_date = _to_date(prestart)
    runtime._set_context(ctx)
    try:
        initialize(ctx)
    finally:
        runtime._clear_context()

    # set_benchmark 可在 initialize 内覆盖 benchmark
    if runtime._benchmark is not None:
        ctx.benchmark = normalize_security(runtime._benchmark)
        loader.load(ctx.benchmark)

    order_cost = runtime._order_cost
    schedule = runtime._schedule
    real_price = runtime._use_real_price()

    fills: list = []
    equity_vals: list = []

    for i, d in enumerate(loop_dates):
        prev_str = loop_dates[i - 1] if i > 0 else prestart
        ctx.current_dt = _to_dt(d)
        ctx.previous_date = _to_date(prev_str)
        ctx.pending_orders = []
        ctx.prev_close = {
            sec: loader.price(sec, prev_str, 'close', real_price=real_price),
        }
        if t_plus_1:
            portfolio.start_of_day()

        runtime._set_context(ctx)
        try:
            for fn in schedule['before_open']:
                fn(ctx)
            for fn in schedule['open']:
                fn(ctx)

            fill = loader.price(sec, d, fill_price, real_price=real_price)
            if fill is not None:
                for code, shares in ctx.pending_orders:
                    pos = portfolio.positions.get(code)
                    if pos is None:
                        if shares <= 0:
                            continue
                        pos = Position(code=code)
                        portfolio.positions[code] = pos
                    filled = execute_order(
                        portfolio, pos, shares, fill,
                        cost=order_cost, is_stock=loader.is_stock[sec], lot_size=lot_size,
                    )
                    if filled:
                        fills.append((d, 'buy' if filled > 0 else 'sell', abs(filled), fill))

            for fn in schedule['after_close']:
                fn(ctx)
        finally:
            runtime._clear_context()

        close = loader.price(sec, d, 'close', real_price=real_price)
        portfolio.mark_to_market({sec: close or 0.0})
        equity_vals.append(portfolio.total_value)

    equity = pd.Series(equity_vals, index=loop_dates, name='equity')
    strat_ret = equity.pct_change().fillna(0.0)
    benchmark_series = _benchmark_curve(loader, ctx.benchmark, loop_dates, initial_capital)
    trades = _pair_trades(fills, date_pos)
    metrics = _metrics(equity, benchmark_series, strat_ret, trades, initial_capital)

    result = BacktestResult(equity=equity, benchmark=benchmark_series, trades=trades, metrics=metrics)
    plot_equity(result, initial_capital=initial_capital, summary=result.summary())
    return result
