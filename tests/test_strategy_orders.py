from __future__ import annotations

import pytest

from scripts.data_pipeline.strategy.context import (
    OrderCost,
    Portfolio,
    Position,
    execute_order,
)


def test_buy_rounds_down_to_lot() -> None:
    p = Portfolio(cash=100_000.0, starting_cash=100_000.0)
    pos = Position(code='X')
    filled = execute_order(
        p, pos, 250, 10.0,
        cost=OrderCost(open_commission=0, close_commission=0, min_commission=0),
        is_stock=True,
    )
    assert filled == 200
    assert pos.shares == 200
    assert p.cash == pytest.approx(100_000.0 - 200 * 10.0)


def test_buy_charges_commission() -> None:
    p = Portfolio(cash=100_000.0, starting_cash=100_000.0)
    pos = Position(code='X')
    cost = OrderCost(open_commission=0.001, min_commission=0)
    execute_order(p, pos, 100, 10.0, cost=cost, is_stock=True)
    assert p.cash == pytest.approx(100_000.0 - 1000.0 - 1.0)  # gross + 1 元佣金


def test_min_commission_floor() -> None:
    p = Portfolio(cash=100_000.0, starting_cash=100_000.0)
    pos = Position(code='X')
    cost = OrderCost(open_commission=0.0003, min_commission=5.0)
    # 100 股 * 10 元 * 0.0003 = 0.3 元 < 5 元 -> 收 5 元
    execute_order(p, pos, 100, 10.0, cost=cost, is_stock=True)
    assert p.cash == pytest.approx(100_000.0 - 1000.0 - 5.0)


def test_sell_respects_available_and_tax() -> None:
    p = Portfolio(cash=0.0, starting_cash=0.0)
    pos = Position(code='X', shares=500, available_shares=300)
    cost = OrderCost(close_commission=0, close_tax=0.001, min_commission=0)
    filled = execute_order(p, pos, -500, 10.0, cost=cost, is_stock=True)
    assert filled == -300  # 只能卖可卖量 300
    assert pos.shares == 200
    assert p.cash == pytest.approx(300 * 10.0 - 300 * 10.0 * 0.001)


def test_index_has_no_close_tax() -> None:
    p = Portfolio(cash=0.0, starting_cash=0.0)
    pos = Position(code='X', shares=100, available_shares=100)
    cost = OrderCost(close_commission=0, close_tax=0.001, min_commission=0)
    execute_order(p, pos, -100, 10.0, cost=cost, is_stock=False)
    assert p.cash == pytest.approx(1000.0)  # 指数卖出无印花税


def test_start_of_day_unlocks_shares() -> None:
    pos = Position(code='X', shares=100, available_shares=0)
    p = Portfolio(cash=0.0, starting_cash=0.0, positions={'X': pos})
    p.start_of_day()
    assert pos.available_shares == 100
