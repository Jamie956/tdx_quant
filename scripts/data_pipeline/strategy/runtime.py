from __future__ import annotations

import logging
from types import SimpleNamespace

from scripts.data_pipeline.strategy.context import OrderCost

log = logging.getLogger('strategy')

# 持久全局状态（聚宽 ``g`` 的本地等价）。策略直接 ``from ...strategy import g`` 后读写。
# 注意：引擎重置时只清空 ``__dict__``，不重新绑定名字，保证策略 import 到的引用始终有效。
g = SimpleNamespace()

_current_context = None
_schedule: dict[str, list] = {'before_open': [], 'open': [], 'after_close': []}
_benchmark: str | None = None
_order_cost = OrderCost()
_options: dict = {}


def _reset() -> None:
    global _current_context, _benchmark, _order_cost
    _current_context = None
    _schedule.clear()
    _schedule.update({'before_open': [], 'open': [], 'after_close': []})
    _benchmark = None
    _order_cost = OrderCost()
    _options.clear()
    g.__dict__.clear()


def _seed_default_options(adjust: str) -> None:
    # adjust='none' -> 不复权（use_real_price=True）；否则默认前复权
    _options['use_real_price'] = (adjust == 'none')


def _use_real_price() -> bool:
    return bool(_options.get('use_real_price', False))


def _set_context(ctx) -> None:
    global _current_context
    _current_context = ctx


def _clear_context() -> None:
    global _current_context
    _current_context = None


def _ctx():
    if _current_context is None:
        raise RuntimeError('strategy API called outside the backtest loop (no active context)')
    return _current_context


# ----------------------------------------------------------------------
# 初始化期 API（在 initialize 内调用）
# ----------------------------------------------------------------------
def set_benchmark(security) -> None:
    _benchmark = security


def set_order_cost(cost: OrderCost) -> None:
    global _order_cost
    _order_cost = cost


def set_option(name: str, value) -> None:
    _options[name] = value


def run_daily(func, time: str = 'open', reference_security: str | None = None) -> None:
    if time not in _schedule:
        raise ValueError(f'run_daily time must be one of {tuple(_schedule)}, got {time!r}')
    _schedule[time].append(func)


def send_message(msg: str) -> None:
    log.info('send_message: %s', msg)


# ----------------------------------------------------------------------
# 运行期 API（在回调内调用）
# ----------------------------------------------------------------------
def get_price(security, start_date=None, end_date=None, frequency: str = '1d', fields=None):
    ctx = _ctx()
    return ctx.data.get_price(
        security, start=start_date, end=end_date, fields=fields,
        asof=ctx.previous_date, real_price=_use_real_price(),
    )


def attribute_history(security, count: int, unit: str = '1d', fields=None):
    ctx = _ctx()
    return ctx.data.attribute_history(
        security, count, fields=fields, asof=ctx.previous_date, real_price=_use_real_price(),
    )


def _check_orderable(ctx, security: str) -> None:
    if security != ctx.security:
        raise ValueError(f'single-symbol backtest: cannot order {security!r} (only {ctx.security!r})')


def _position_shares(ctx, code: str) -> int:
    pos = ctx.portfolio.position(code)
    return pos.shares if pos else 0


def order(security, amount: int) -> None:
    ctx = _ctx()
    _check_orderable(ctx, security)
    if amount:
        ctx.pending_orders.append((security, int(amount)))


def order_target(security, amount: int) -> None:
    ctx = _ctx()
    _check_orderable(ctx, security)
    delta = int(amount) - _position_shares(ctx, security)
    if delta:
        ctx.pending_orders.append((security, delta))


def order_value(security, value: float) -> None:
    ctx = _ctx()
    _check_orderable(ctx, security)
    price = ctx.prev_close.get(security)
    if not price or price <= 0:
        return
    shares = int(float(value) / price) // 100 * 100
    if shares:
        ctx.pending_orders.append((security, shares))


def order_target_value(security, value: float) -> None:
    ctx = _ctx()
    _check_orderable(ctx, security)
    price = ctx.prev_close.get(security)
    if not price or price <= 0:
        return
    target = int(float(value) / price) // 100 * 100
    delta = target - _position_shares(ctx, security)
    if delta:
        ctx.pending_orders.append((security, delta))
