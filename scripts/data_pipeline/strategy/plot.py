"""回测结果可视化（plotly，自包含 HTML，无 CDN）。

用法：
    from scripts.data_pipeline.strategy.plot import plot_equity
    plot_equity(result, initial_capital=1_000_000)
"""
from __future__ import annotations

import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from scripts.data_pipeline.backtest import BacktestResult

STRATEGY = "#d92d20"                 # 策略净值
BENCH = "#64748b"                    # 基准
GRID = "rgba(100,116,139,0.15)"


def _to_datetime_index(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index, format='%Y%m%d')
    return s


def plot_equity(
    result: BacktestResult,
    *,
    initial_capital: float | None = None,
    title: str = '策略净值 vs 买入持有',
    summary: str | None = None,
    output: str = 'data/backtest_equity.html',
    open_browser: bool = True,
) -> go.Figure:
    """画策略累计收益 vs 基准（买入持有）。

    ``result`` 是 ``BacktestResult``；``initial_capital`` 缺省时从基准曲线首点
    （即初始资金）推导。落盘自包含 HTML 并可选自动打开，返回 figure 供继续定制。
    """
    capital = (
        float(initial_capital)
        if initial_capital is not None
        else float(result.benchmark.iloc[0])
    )

    eq = _to_datetime_index(result.equity / capital - 1.0)
    bench = _to_datetime_index(result.benchmark / capital - 1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq, name='策略', line=dict(color=STRATEGY, width=2),
        hovertemplate='%{x}<br>策略 %{y:.2%}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=bench.index, y=bench, name='基准', line=dict(color=BENCH, width=1.5),
        hovertemplate='%{x}<br>基准 %{y:.2%}<extra></extra>',
    ))

    fig.update_layout(
        template='plotly_white', height=520, hovermode='x unified', title=title,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        margin=dict(l=8, r=8, t=40, b=8),
    )
    fig.update_xaxes(tickformat='%Y-%m')
    fig.update_yaxes(title_text='累计收益', tickformat='.0%', gridcolor=GRID)

    if summary:
        fig.add_annotation(
            text=summary.replace('\n', '<br>'),
            xref='paper', yref='paper',
            x=0.01, y=0.99, xanchor='left', yanchor='top',
            showarrow=False, align='left',
            font=dict(family='Menlo, Monaco, monospace', size=11, color='#374151'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e5e7eb', borderwidth=1, borderpad=8,
        )

    if output:
        fig.write_html(output, include_plotlyjs=True, config={'scrollZoom': True})
        print(f'saved -> {output}')
        if open_browser:
            webbrowser.open(f'file://{Path(output).resolve()}')
    return fig
