from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderCost:
    """交易成本：开/平仓佣金 + 印花税（仅个股卖出）+ 单笔最低佣金。"""
    open_commission: float = 0.0003
    close_commission: float = 0.0003
    close_tax: float = 0.001
    min_commission: float = 5.0


@dataclass
class Position:
    code: str
    shares: int = 0            # 总持仓股数
    available_shares: int = 0  # 可卖股数（T+1：当日买入次日可卖）
    avg_cost: float = 0.0      # 加权平均买入价（不含手续费）


@dataclass
class Portfolio:
    cash: float
    starting_cash: float
    positions: dict = field(default_factory=dict)
    total_value: float = 0.0

    def position(self, code: str) -> Position | None:
        return self.positions.get(code)

    def mark_to_market(self, prices: dict) -> None:
        """按 ``prices``（code -> 最新价）重估总资产。"""
        value = self.cash
        for code, pos in self.positions.items():
            value += pos.shares * float(prices.get(code, 0.0))
        self.total_value = value

    def start_of_day(self) -> None:
        """T+1：每个交易日开盘前把全部持仓标记为可卖。"""
        for pos in self.positions.values():
            pos.available_shares = pos.shares


@dataclass
class Context:
    """回测运行上下文（聚宽 ``context`` 的本地等价）。"""
    security: str
    benchmark: str
    portfolio: Portfolio
    data: object = None                 # DataLoader
    current_dt: object = None           # 当前交易日 (datetime)
    previous_date: object = None        # 上一交易日 (date)
    # 引擎内部使用的当日状态
    pending_orders: list = field(default_factory=list)  # [(code, signed_shares), ...]
    prev_close: dict = field(default_factory=dict)      # code -> 上一交易日收盘价


def _commission(cost: OrderCost, price: float, shares: int, is_open: bool) -> float:
    rate = cost.open_commission if is_open else cost.close_commission
    return max(cost.min_commission, float(price) * shares * rate)


def execute_order(
    portfolio: Portfolio,
    position: Position,
    order_shares: int,
    price: float,
    *,
    cost: OrderCost,
    is_stock: bool,
    lot_size: int = 100,
) -> int:
    """按 ``price`` 执行一笔带符号股数 ``order_shares`` 的市价单。

    买入按整手向下取整、卖出受可卖量约束；佣金/印花税/最低佣金按 ``cost`` 扣减。
    返回实际成交股数（带符号，0 表示未成交）。
    """
    if order_shares > 0:
        shares = (order_shares // lot_size) * lot_size
        if shares <= 0:
            return 0
        gross = float(price) * shares
        fee = _commission(cost, price, shares, is_open=True)
        portfolio.cash -= gross + fee
        total = position.shares + shares
        if total > 0:
            position.avg_cost = (
                position.avg_cost * position.shares + float(price) * shares
            ) / total
        position.shares = total
        # available_shares 不变：当日买入 T+1 才可卖
        return shares

    shares = min(-order_shares, position.available_shares, position.shares)
    if shares <= 0:
        return 0
    gross = float(price) * shares
    fee = _commission(cost, price, shares, is_open=False)
    tax = gross * cost.close_tax if is_stock else 0.0
    portfolio.cash += gross - fee - tax
    position.shares -= shares
    position.available_shares -= shares
    if position.shares == 0:
        position.avg_cost = 0.0
    return -shares
